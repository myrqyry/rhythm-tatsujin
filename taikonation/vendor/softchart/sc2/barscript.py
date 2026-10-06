"""BarScript encoding layer: SoftChart 2.0 bar-scoped multi-stage stream.

Implements the content layer of SOFTCHART2_DESIGN.md on top of the slot+ S0
command trace (softchart.trace): the trace -> token reordering of §1.1, the
deterministic stage-label derivation of §3 (versioned ``barscript_labels_v1``),
the grammar FSM v2 with the §8.1 plan->decode hard constraints, and the
deterministic re-zip inverse that the UL v2 symbolizer consumes (§1.2).

Per-bar token layout (§1.1)
---------------------------
::

    [BAR] BAR_DENOM_D                                  <- header stage
    DENS_D8_d SPAN_BUDGET_s BAR_SYNC_y MOTIF_m          <- plan stage (4 slots)
    ( SPAN_START? SPAN_END? POS_i )*                    <- skeleton stage
    [REAL] ( TYPE (SPAN_DUR_b)? )*                      <- realization stage

followed by the next ``[BAR]`` or ``EOS``.  POSITION tokens reuse the
``vocab.time0`` family as BAR-LOCAL lattice indices ``i`` with exact position
``i / D`` (v3 slot-mode precedent; keeps hit token counts identical to the
flat interleaved encoding, §1.1 "hit 的 token 總數不變").  ``D`` is the bar's
rational-lattice denominator = LCM of the reduced denominators of the bar's
note positions (SLOTPLUS_DESIGN §2.1 "authored 分割的最小公倍細分"), which is
canonical under authored zero-padding; ``D = 1`` for empty bars; bars needing
``D > MAX_BAR_DENOM`` fail closed (OffLatticeError), never snapped.

Realization emits exactly one TYPE token per skeleton position, in position
order.  A skeleton position preceded by ``SPAN_START`` realizes as a span
class (roll / roll_big / balloon) followed by its duration bucket token
(§1.3 span duration head supervision); a position preceded by ``SPAN_END``
realizes as the ``end`` class; unmarked positions realize as hit classes.

Label derivation (``barscript_labels_v1``, all pure functions of the trace
and a frozen BarscriptSpec; §3)
-------------------------------
- DENS_d8: count of non-``end`` onsets in the bar, binned by
  ``spec.dens_edges`` (bucket = number of edges <= count; the same edges give
  the FSM's [L, U] legal note-count interval, §8.1).
- SPAN_BUDGET: number of span starts in the bar, capped at 3+ .
- Span duration: exact length in quarter-note beats (meter-state weighted
  across bars, tempo-independent), binned on the ``spec.span_dur_edges`` log
  grid.
- SYNC: Longuet-Higgins & Lee note-rest syncopation banded by
  ``spec.sync_edges``.  Deterministic variant frozen here: metric weight
  w(0) = 0, w(beat) = -1, w(other) = -(1 + ceil(log2(q))) with q the reduced
  within-beat denominator; onsets are the bar's non-``end`` notes; rest
  candidates live on the eighth-note grid (precedent: data.py lhl_band
  "simplified to the eighth-note grid") plus the next bar's downbeat when
  that downbeat carries no onset; each onset scores max(w(rest)) - w(onset)
  over rests strictly before the next onset, when positive.
- MOTIF_REF: bar occupancy signature = ordered tuple of exact positions of
  non-``end`` onsets (type-agnostic: occupancy is a placement decision; the
  realizer owns types).  Exact match against bar b-δ for δ in (1, 2, 4, 8)
  ascending -> REPEAT_δ; else normalized Levenshtein distance (over position
  sequences, / max length) minimized over available δ (ties -> smaller δ)
  <= spec.motif_tau -> VAR_δ; else NEW.  Bars with no δ-predecessor are NEW.

Grammar FSM v2 (§1.2, §8.1)
---------------------------
``BarscriptFSM`` returns a per-step legal-token mask over the vocab.  Hard
constraints: header -> plan (4 fixed slots; REPEAT_δ/VAR_δ masked when fewer
than δ bars precede) -> skeleton (positions strictly increasing on the bar's
D grid; DENS bucket bounds the count of non-end positions to [L, U]: [REAL]
is masked below L, new countable positions are masked at U; SPAN_BUDGET is a
consumed budget masking SPAN_START at zero, 3+ = unbounded upper bucket) ->
[REAL] -> exactly N_onset type tokens role-constrained by the skeleton.
Cross-bar open-span carry: while a span is open, SPAN_START and plain
positions are masked (a span may not swallow hits), SPAN_END is the only way
to place content, EOS is masked (no orphan starts), and the state survives
[BAR] boundaries (multi-bar rolls/balloons).  Position masks reserve room so
the DENS lower bound stays reachable; as a documented fail-open escape hatch
(generation robustness, never exercised by gold sequences) [REAL] becomes
legal when no skeleton token could otherwise be emitted.

MOTIF_REF hard constraint (``motif_hard=True``, the default)
------------------------------------------------------------
A committed ``REPEAT_δ`` plan token compiles into the skeleton exactly the
way DENS and SPAN_BUDGET already do: the bar's countable (non-``end``)
positions are forced to reproduce, index for index, the occupancy signature
of bar ``b-δ`` — the same signature the label derivation compares
(``occupancy_signature`` / ``motif_ref``), so a gold ``REPEAT_δ`` bar
satisfies it by construction and teacher forcing is untouched.  ``VAR_δ``
(normalized edit distance <= τ, not an equality) and ``NEW`` stay soft.

The signature is a tuple of EXACT rational positions, so it is
denominator-free: a bar whose ``BAR_DENOM`` differs from its δ-predecessor's
still counts as a REPEAT whenever every predecessor position lands on this
bar's lattice (``bar_labels`` never looks at ``D``; the FSM matches that
reading and only requires representability).

Conflicts are resolved by PREVENTION at the plan stage, which is where the
priority order lives.  ``BAR_DENOM`` (header), ``DENS`` (plan0) and
``SPAN_BUDGET`` (plan1) are all committed before ``MOTIF`` (plan3), so at
plan3 the FSM masks every ``REPEAT_δ`` it could not honour: signature not
representable on this bar's ``D``, ``len(signature)`` outside the DENS
``[L, U]`` interval, or a carried open span that would have to close before
a first target position sitting at index 0.  ``NEW`` is always legal, so the
plan mask is never empty and the skeleton stage can never dead-end.  Within
the skeleton the carry rule keeps absolute priority (a span still may not
swallow hits): while a span is open only ``SPAN_END`` places content, and its
position is additionally bounded to stay strictly before the next forced
target.  ``SPAN_START`` is offered only where the following target leaves a
slot for the close, so committing to a span can never strand the FSM.
Redundant references are also masked: ``REPEAT_δ`` is illegal when a smaller
available δ' has the identical signature, because the realized bar would
relabel as ``REPEAT_δ'`` — gold always emits the minimal δ, so this is a
pure canonicalization.  ``motif_hard=False`` restores the pre-v18 soft
behaviour for ablations.

Scope rulings (documented, with authority)
------------------------------------------
- slot+ header state tokens (TEMPO/MEASURE/DELAY/k families) are the time
  layer's vocabulary (SLOTPLUS_DESIGN §3.2 puts their values on regression /
  classification heads; §9.2 defers their token families).  This content
  layer anchors them at the ``[BAR]`` header insertion point but does not
  invent them: TEMPO/MEASURE/DELAY events pass through encoding untouched
  (they move no lattice position, so the (pos, type) re-zip is exact without
  them) and labels that need meter (beats, LHL) read it from the trace.
- ``barscript_to_events`` restores the (bar, pos, type-class) event stream
  only (the task's re-zip inverse); '9' kusudama collapses into the balloon
  class exactly as in vocab.NOTE_TYPE_MAP, so equality checks against the
  flat encoding and against the trace are at note-class level.
- Span-semantics normalization (coordinator ruling 2026-07-25, full-corpus
  diagnostic ``span_diag.json``: 328 fail-closed charts, 98.2% one pattern;
  BALLOON header counts are the metadata evidence that the game engine treats
  these as decorations of the OPEN span, not new spans — implicit
  close+reopen would break the balloon counts): while a span is open, a
  SAME-FAMILY span-start event is dropped and counted (balloon family:
  balloon/kusudama chars are interchangeable, so any balloon-class start
  inside an open balloon; roll family: exactly roll->roll). A span-end event
  with no open span is likewise dropped and counted (engine ignores it).
  The dropped events vanish from the encoding entirely: lattice denominators,
  DENS counts, occupancy signatures and the re-zip reference stream are all
  computed on the NORMALIZED stream (``normalized_events``); the flat B0
  encoding applies the identical normalization so both arms share one data
  population. Counts return via ``return_stats=True`` /
  ``normalized_events``. Everything else stays fail-closed: cross-family
  restarts (e.g. roll open + balloon start), roll_big->roll_big (unobserved,
  not sanctioned by the ruling — pending adjudication if it ever appears),
  hits strictly inside an open span, and spans left open at end of chart.
  trace.py stays untouched (S0 verbatim contract): normalization lives
  strictly in this encoding layer.
"""

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .trace import NoteEvent, note_class
from .vocab import (MAX_BAR_DENOM, MOTIF_DELTAS, MOTIF_REF_NAMES, N_BAR_SYNC,
                    N_DENS_D8, N_SPAN_BUDGET, N_SPAN_DUR_BUCKETS,
                    NOTE_CLASSES, STAGE_HEADER, STAGE_PLAN, STAGE_REALIZATION,
                    STAGE_SKELETON, VOCAB, WINDOW)

HIT_CLASSES = ("don", "ka", "don_big", "ka_big")
SPAN_CLASSES = ("roll", "roll_big", "balloon")

LABELS_VERSION = "barscript_labels_v1"


class BarscriptError(ValueError):
    """A construct the BarScript layer refuses to guess about."""


class OffLatticeError(BarscriptError):
    """Bar needs a lattice denominator beyond MAX_BAR_DENOM (fail-closed)."""


class SpanStructureError(BarscriptError):
    """Authored span structure is inconsistent (orphan start/end, swallowed
    hit inside an open span). Fail-closed and counted, never repaired."""


class IllegalTokenError(BarscriptError):
    """A token outside the FSM's legal mask was stepped."""


# ---------------------------------------------------------------------------
# spec


def _as_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else x


@dataclass(frozen=True)
class BarscriptSpec:
    """Frozen label-derivation parameters (§3: bucket edges and τ are NOT
    hard-coded; they load from the spec artifact built on the train split).

    Defaults are DRAFT values for local development; the frozen artifact from
    scripts/build_barscript_spec.py supersedes them before training.
    """

    labels_version: str = LABELS_VERSION
    # 7 strictly increasing positive ints -> 8 DENS_d8 buckets over per-bar
    # non-end onset counts; bucket b = number of edges <= count, so bucket b
    # spans [edges[b-1], edges[b]-1] (b=0: [0, edges[0]-1]; top: unbounded).
    dens_edges: tuple = (1, 3, 5, 8, 11, 14, 18)
    # 15 strictly increasing floats (quarter-note beats, log-ish grid) -> 16
    # span-duration buckets; frozen grid anchors on authored p50/p90/p99.
    span_dur_edges: tuple = (0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                             6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
    # 5 strictly increasing positive ints -> 6 LHL bands over per-bar scores.
    sync_edges: tuple = (1, 2, 3, 5, 8)
    # normalized-edit-distance threshold for VAR (frozen after the τ sweep).
    motif_tau: float = 0.25
    motif_deltas: tuple = MOTIF_DELTAS
    provenance: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "dens_edges", _as_tuple(self.dens_edges))
        object.__setattr__(self, "span_dur_edges",
                           _as_tuple(self.span_dur_edges))
        object.__setattr__(self, "sync_edges", _as_tuple(self.sync_edges))
        object.__setattr__(self, "motif_deltas",
                           _as_tuple(self.motif_deltas))
        if self.labels_version != LABELS_VERSION:
            raise BarscriptError(
                f"unsupported labels_version {self.labels_version!r} "
                f"(this code implements {LABELS_VERSION!r})")
        for name, edges, n in (("dens_edges", self.dens_edges, N_DENS_D8),
                               ("span_dur_edges", self.span_dur_edges,
                                N_SPAN_DUR_BUCKETS),
                               ("sync_edges", self.sync_edges, N_BAR_SYNC)):
            if len(edges) != n - 1:
                raise BarscriptError(
                    f"{name} must have {n - 1} edges, got {len(edges)}")
            if any(b <= a for a, b in zip(edges, edges[1:])):
                raise BarscriptError(f"{name} must be strictly increasing")
        if self.dens_edges[0] < 1 or any(
                int(e) != e for e in self.dens_edges):
            raise BarscriptError("dens_edges must be positive integers")
        if self.motif_deltas != MOTIF_DELTAS:
            raise BarscriptError(
                f"motif_deltas is frozen to {MOTIF_DELTAS} by the vocab "
                "MOTIF_REF token family")
        if not (0.0 <= self.motif_tau <= 1.0):
            raise BarscriptError("motif_tau must be in [0, 1]")

    # --- deterministic binning -------------------------------------------
    def dens_bucket(self, count):
        return sum(1 for e in self.dens_edges if e <= count)

    def dens_bounds(self, bucket):
        """Legal non-end onset count interval [L, U] for a DENS bucket
        (U is None for the unbounded top bucket). §8.1 hard constraint."""
        lo = 0 if bucket == 0 else int(self.dens_edges[bucket - 1])
        hi = (None if bucket >= len(self.dens_edges)
              else int(self.dens_edges[bucket]) - 1)
        return lo, hi

    def span_budget_bucket(self, count):
        return min(count, N_SPAN_BUDGET - 1)

    def span_dur_bucket(self, beats):
        return sum(1 for e in self.span_dur_edges if e <= float(beats))

    def sync_band(self, score):
        return min(sum(1 for e in self.sync_edges if e <= score),
                   N_BAR_SYNC - 1)

    # --- (de)serialization ------------------------------------------------
    def to_dict(self):
        return {"labels_version": self.labels_version,
                "dens_edges": list(self.dens_edges),
                "span_dur_edges": list(self.span_dur_edges),
                "sync_edges": list(self.sync_edges),
                "motif_tau": self.motif_tau,
                "motif_deltas": list(self.motif_deltas),
                "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, d):
        if "spec" in d and "labels_version" not in d:
            d = d["spec"]  # accept the build_barscript_spec.py artifact
        known = {"labels_version", "dens_edges", "span_dur_edges",
                 "sync_edges", "motif_tau", "motif_deltas", "provenance"}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# trace -> per-bar content


def _bar_notes(trace):
    """Per-bar ordered [(pos: Fraction, class: str)] over trace.bars."""
    bars = [[] for _ in trace.bars]
    for ev in trace.events:
        if isinstance(ev, NoteEvent):
            cls = note_class(ev.type)
            if cls is None:
                raise BarscriptError(
                    f"bar {ev.bar}: unknown note char {ev.type!r}")
            bars[ev.bar].append((ev.pos, cls))
    return bars


def bar_lattice_denom(notes):
    """Canonical lattice denominator: LCM of reduced position denominators
    (1 for an empty bar). Fail-closed above MAX_BAR_DENOM (never snapped)."""
    d = 1
    for pos, _cls in notes:
        d = d * pos.denominator // math.gcd(d, pos.denominator)
        if d > MAX_BAR_DENOM:
            raise OffLatticeError(
                f"bar lattice denominator {d} exceeds support "
                f"{MAX_BAR_DENOM} (fail-closed)")
    return d


def _same_family_restart(open_cls, cls):
    """Sanctioned decoration pattern (ruling 2026-07-25): balloon-family
    starts are interchangeable inside an open balloon (7/9 chars both map to
    the balloon class); in the roll family exactly roll->roll is sanctioned.
    roll_big->roll_big and all cross-family restarts stay fail-closed."""
    return ((open_cls == "balloon" and cls == "balloon")
            or (open_cls == "roll" and cls == "roll"))


def _normalize_spans(trace, bars_notes):
    """Span-semantics normalization + global span pairing (fail-closed for
    everything the ruling does not sanction; module docstring).

    Returns (norm_bars, spans, start_index, stats): norm_bars mirrors
    bars_notes with ignored events dropped; spans carry exact (bar, pos)
    endpoints and duration in quarter-note beats; start_index maps each kept
    span start's (bar, pos) to its span dict; stats counts
    {"ignored_start", "ignored_end"}.
    """
    beats = [Fraction(4 * b.num, b.den) for b in trace.bars]
    norm_bars = [[] for _ in bars_notes]
    spans = []
    start_index = {}
    stats = {"ignored_start": 0, "ignored_end": 0}
    open_span = None
    for bar_i, notes in enumerate(bars_notes):
        for pos, cls in notes:
            if cls in SPAN_CLASSES:
                if open_span is None:
                    open_span = {"bar": bar_i, "pos": pos, "class": cls}
                    norm_bars[bar_i].append((pos, cls))
                elif _same_family_restart(open_span["class"], cls):
                    stats["ignored_start"] += 1  # decoration; engine ignores
                else:
                    raise SpanStructureError(
                        f"bar {bar_i} pos {pos}: {cls} span start while the "
                        f"{open_span['class']} span opened at bar "
                        f"{open_span['bar']} is open (not a sanctioned "
                        "same-family decoration)")
            elif cls == "end":
                if open_span is None:
                    stats["ignored_end"] += 1  # orphan end; engine ignores
                    continue
                dur = Fraction(0)
                for b in range(open_span["bar"], bar_i + 1):
                    lo = open_span["pos"] if b == open_span["bar"] else Fraction(0)
                    hi = pos if b == bar_i else Fraction(1)
                    dur += (hi - lo) * beats[b]
                span = dict(open_span, end_bar=bar_i, end_pos=pos,
                            dur_beats=dur)
                spans.append(span)
                start_index[(open_span["bar"], open_span["pos"])] = span
                open_span = None
                norm_bars[bar_i].append((pos, cls))
            else:
                if open_span is not None:
                    raise SpanStructureError(
                        f"bar {bar_i} pos {pos}: hit inside the span opened "
                        f"at bar {open_span['bar']} (swallowed hit)")
                norm_bars[bar_i].append((pos, cls))
    if open_span is not None:
        raise SpanStructureError(
            f"span opened at bar {open_span['bar']} pos {open_span['pos']} "
            "never closes (orphan start)")
    return norm_bars, spans, start_index, stats


def normalized_events(trace):
    """Post-normalization (bar, pos, class) stream + normalization stats.

    This is the re-zip REFERENCE stream: both the flat and the BarScript
    encodings reproduce exactly these events. Equal to trace_events(trace)
    with stats == zeros for charts that need no normalization."""
    norm_bars, _spans, _idx, stats = _normalize_spans(trace, _bar_notes(trace))
    events = [(b, pos, cls) for b, notes in enumerate(norm_bars)
              for pos, cls in notes]
    return events, stats


# --- LHL syncopation (barscript_labels_v1 frozen variant) ------------------


def metric_weight(pos, num):
    """LHL metric weight of an in-bar position (Fraction in [0,1)); num =
    metrical beats per bar. 0 = downbeat, -1 = beat, deeper = weaker."""
    if pos == 0:
        return 0
    beat = pos * num
    if beat.denominator == 1:
        return -1
    frac = beat - (beat.numerator // beat.denominator)
    q = frac.denominator  # >= 2
    return -(1 + (q - 1).bit_length())  # 1 + ceil(log2(q))


def lhl_score(onsets, num, next_downbeat_has_onset):
    """Bar-level LHL note-rest syncopation score (non-negative int)."""
    ons = sorted(set(onsets))
    if not ons:
        return 0
    onset_set = set(ons)
    grid = [Fraction(k, 2 * num) for k in range(1, 2 * num)]
    score = 0
    for i, p in enumerate(ons):
        bound = ons[i + 1] if i + 1 < len(ons) else Fraction(1)
        best = None
        for q in grid:
            if p < q < bound and q not in onset_set:
                w = metric_weight(q, num)
                if best is None or w > best:
                    best = w
        if i == len(ons) - 1 and not next_downbeat_has_onset:
            best = 0 if best is None else max(best, 0)
        wp = metric_weight(p, num)
        if best is not None and best > wp:
            score += best - wp
    return score


# --- motif reference -------------------------------------------------------


def occupancy_signature(notes):
    """Ordered tuple of exact positions of the bar's non-end onsets."""
    return tuple(pos for pos, cls in notes if cls != "end")


def norm_edit_distance(a, b):
    """Levenshtein over position sequences, normalized by max length
    (0.0 for two empty sequences)."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (0 if x == y else 1)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def motif_ref(signatures, bar_i, tau, deltas=MOTIF_DELTAS):
    """(name, distance) for bar bar_i against its δ-predecessors.

    Exact match at the smallest available δ wins REPEAT_δ; otherwise the
    minimal normalized edit distance over available δ (ties -> smaller δ)
    yields VAR_δ when <= τ; otherwise NEW. distance is None for NEW-by-
    no-predecessor, 0.0 for REPEAT, the winning distance otherwise.
    """
    sig = signatures[bar_i]
    avail = [d for d in deltas if bar_i - d >= 0]
    if not avail:
        return "new", None
    for d in avail:
        if signatures[bar_i - d] == sig:
            return f"repeat_{d}", 0.0
    best_d, best = None, None
    for d in avail:
        dist = norm_edit_distance(sig, signatures[bar_i - d])
        if best is None or dist < best:
            best_d, best = d, dist
    if best is not None and best <= tau:
        return f"var_{best_d}", best
    return "new", best


# ---------------------------------------------------------------------------
# labels


def _bar_labels_impl(trace, spec):
    """(labels, normalization stats); labels are computed on the NORMALIZED
    note stream (ignored decorations do not shape lattice denominators, DENS
    counts, occupancy signatures or LHL scores)."""
    norm_bars, _spans, start_index, stats = _normalize_spans(
        trace, _bar_notes(trace))
    signatures = [occupancy_signature(n) for n in norm_bars]
    out = []
    for i, notes in enumerate(norm_bars):
        denom = bar_lattice_denom(notes)
        onset_pos = [pos for pos, cls in notes if cls != "end"]
        n_onsets = len(onset_pos)
        next_db = (i + 1 < len(norm_bars)
                   and any(pos == 0 and cls != "end"
                           for pos, cls in norm_bars[i + 1]))
        score = lhl_score(onset_pos, trace.bars[i].num, next_db)
        ref, dist = motif_ref(signatures, i, spec.motif_tau,
                              spec.motif_deltas)
        spans_here = []
        for pos, cls in notes:
            if cls in SPAN_CLASSES:
                sp = start_index[(i, pos)]
                spans_here.append({
                    "pos": pos, "class": cls,
                    "dur_beats": sp["dur_beats"],
                    "dur_bucket": spec.span_dur_bucket(sp["dur_beats"]),
                    "end_bar": sp["end_bar"], "end_pos": sp["end_pos"]})
        out.append({
            "bar": i, "denom": denom, "notes": notes,
            "n_onsets": n_onsets,
            "dens_bucket": spec.dens_bucket(n_onsets),
            "n_span_starts": len(spans_here),
            "span_budget_bucket": spec.span_budget_bucket(len(spans_here)),
            "sync_score": score, "sync_band": spec.sync_band(score),
            "motif_ref": ref, "motif_distance": dist,
            "spans": spans_here,
        })
    return out, stats


def bar_labels(trace, spec):
    """Deterministic per-bar labels (barscript_labels_v1). Pure function of
    (trace, spec); §3. Returns a list with one dict per trace bar (computed
    on the normalized note stream; use normalized_events for the stats)."""
    return _bar_labels_impl(trace, spec)[0]


# ---------------------------------------------------------------------------
# encoding


def _pos_token(pos, denom):
    idx = pos * denom
    if idx.denominator != 1:
        raise OffLatticeError(
            f"position {pos} does not sit on the /{denom} lattice")
    idx = int(idx)
    if not 0 <= idx < min(denom, WINDOW):
        raise BarscriptError(f"position index {idx} out of [0, {denom})")
    return VOCAB.time(idx)


def trace_to_barscript(trace, spec, return_stats=False):
    """Encode a trace as the BarScript stream. Returns (token_ids, stage_ids)
    (parallel lists; stages use vocab.STAGE_*), or (token_ids, stage_ids,
    stats) with ``return_stats=True`` where stats carries the span
    normalization counters. Ends with EOS; the window prefix ([BOS],
    conditions) is the train-side builder's job."""
    labels, stats = _bar_labels_impl(trace, spec)
    v = VOCAB
    tokens, stages = [], []

    def put(tok, stage):
        tokens.append(tok)
        stages.append(stage)

    for lab in labels:
        denom = lab["denom"]
        put(v.bar, STAGE_HEADER)
        put(v.bar_denom[denom], STAGE_HEADER)
        put(v.bar_dens[lab["dens_bucket"]], STAGE_PLAN)
        put(v.span_budget[lab["span_budget_bucket"]], STAGE_PLAN)
        put(v.bar_sync[lab["sync_band"]], STAGE_PLAN)
        put(v.motif[lab["motif_ref"]], STAGE_PLAN)
        for pos, cls in lab["notes"]:
            if cls in SPAN_CLASSES:
                put(v.span_start, STAGE_SKELETON)
            elif cls == "end":
                put(v.span_end, STAGE_SKELETON)
            put(_pos_token(pos, denom), STAGE_SKELETON)
        put(v.real, STAGE_REALIZATION)
        span_iter = iter(lab["spans"])
        for pos, cls in lab["notes"]:
            put(v.note[cls], STAGE_REALIZATION)
            if cls in SPAN_CLASSES:
                put(v.span_dur[next(span_iter)["dur_bucket"]],
                    STAGE_REALIZATION)
    put(v.eos, STAGE_HEADER)
    if return_stats:
        return tokens, stages, stats
    return tokens, stages


def trace_to_flat(trace, return_stats=False):
    """Flat interleaved encoding (the B0 control arm's ordering): per bar
    [BAR] BAR_DENOM_D then (POS, TYPE) pairs in stream order. Same position
    and type token families as BarScript; no plan/skeleton/duration tokens.
    Applies the IDENTICAL span normalization as trace_to_barscript so both
    arms train on one data population."""
    norm_bars, _spans, _idx, stats = _normalize_spans(trace, _bar_notes(trace))
    v = VOCAB
    tokens = []
    for notes in norm_bars:
        denom = bar_lattice_denom(notes)
        tokens += [v.bar, v.bar_denom[denom]]
        for pos, cls in notes:
            tokens += [_pos_token(pos, denom), v.note[cls]]
    tokens.append(v.eos)
    if return_stats:
        return tokens, stats
    return tokens


# ---------------------------------------------------------------------------
# re-zip (decoding back to events)


def _tok_family(tok):
    v = VOCAB
    if tok == v.bar:
        return "bar"
    if tok == v.real:
        return "real"
    if tok == v.eos:
        return "eos"
    if tok == v.span_start:
        return "span_start"
    if tok == v.span_end:
        return "span_end"
    if v.time0 <= tok < v.time0 + WINDOW:
        return "pos"
    if tok in v.id2note:
        return "note"
    for fam in ("bar_denom", "bar_dens", "span_budget", "bar_sync", "motif",
                "span_dur"):
        if tok in getattr(v, fam).values():
            return fam
    return "other"


def barscript_to_events(token_ids):
    """Deterministic re-zip inverse: BarScript tokens -> ordered
    [(bar_index, pos: Fraction, note_class)] (§1.2 "skeleton+realization
    確定性 re-zip 回 (pos,type) 事件流"). Plan tokens are skipped; skeleton
    and realization are zipped 1:1 in position order. Raises BarscriptError
    on malformed structure."""
    v = VOCAB
    events = []
    i, n = 0, len(token_ids)
    bar_i = -1
    denom_rev = {t: d for d, t in v.bar_denom.items()}
    while i < n:
        tok = token_ids[i]
        if tok == v.eos:
            return events
        if tok != v.bar:
            raise BarscriptError(f"token {i}: expected [BAR] or EOS")
        bar_i += 1
        i += 1
        if i >= n or token_ids[i] not in denom_rev:
            raise BarscriptError(f"token {i}: expected BAR_DENOM after [BAR]")
        denom = denom_rev[token_ids[i]]
        i += 1
        # plan stage: skip plan-family tokens
        while i < n and _tok_family(token_ids[i]) in (
                "bar_dens", "span_budget", "bar_sync", "motif"):
            i += 1
        # skeleton stage
        skeleton = []  # (idx, role)
        while i < n:
            fam = _tok_family(token_ids[i])
            if fam == "real":
                i += 1
                break
            role = "hit"
            if fam in ("span_start", "span_end"):
                role = fam
                i += 1
                fam = _tok_family(token_ids[i]) if i < n else None
            if fam != "pos":
                raise BarscriptError(
                    f"token {i}: expected a POSITION token in skeleton")
            idx = token_ids[i] - v.time0
            if idx >= denom or (skeleton and idx <= skeleton[-1][0]):
                raise BarscriptError(
                    f"token {i}: position {idx} not strictly increasing "
                    f"on the /{denom} grid")
            skeleton.append((idx, role))
            i += 1
        else:
            raise BarscriptError("missing [REAL] before end of stream")
        # realization stage: exactly one type per skeleton entry
        for idx, role in skeleton:
            if i >= n or token_ids[i] not in v.id2note:
                raise BarscriptError(
                    f"token {i}: expected a TYPE token in realization")
            cls = v.id2note[token_ids[i]]
            i += 1
            if role == "span_start":
                if cls not in SPAN_CLASSES:
                    raise BarscriptError(
                        f"token {i - 1}: SPAN_START position realized as "
                        f"{cls!r}")
                if i < n and _tok_family(token_ids[i]) == "span_dur":
                    i += 1
            elif role == "span_end":
                if cls != "end":
                    raise BarscriptError(
                        f"token {i - 1}: SPAN_END position realized as "
                        f"{cls!r}")
            elif cls not in HIT_CLASSES:
                raise BarscriptError(
                    f"token {i - 1}: unmarked position realized as {cls!r}")
            events.append((bar_i, Fraction(idx, denom), cls))
    raise BarscriptError("stream ended without EOS")


def flat_to_events(token_ids):
    """Flat-encoding inverse: [(bar_index, pos: Fraction, note_class)]."""
    v = VOCAB
    denom_rev = {t: d for d, t in v.bar_denom.items()}
    events = []
    bar_i, denom = -1, None
    i, n = 0, len(token_ids)
    while i < n:
        tok = token_ids[i]
        if tok == v.eos:
            return events
        if tok == v.bar:
            bar_i += 1
            i += 1
            if i >= n or token_ids[i] not in denom_rev:
                raise BarscriptError(
                    f"token {i}: expected BAR_DENOM after [BAR]")
            denom = denom_rev[token_ids[i]]
            i += 1
            continue
        if _tok_family(tok) != "pos" or denom is None:
            raise BarscriptError(f"token {i}: expected [BAR]/POS/EOS")
        idx = tok - v.time0
        i += 1
        if i >= n or token_ids[i] not in v.id2note:
            raise BarscriptError(f"token {i}: expected TYPE after POSITION")
        events.append((bar_i, Fraction(idx, denom), v.id2note[token_ids[i]]))
        i += 1
    raise BarscriptError("stream ended without EOS")


def trace_events(trace):
    """RAW [(bar, pos, class)] stream straight from the trace (no span
    normalization; the re-zip reference is normalized_events)."""
    return [(ev.bar, ev.pos, note_class(ev.type))
            for ev in trace.events if isinstance(ev, NoteEvent)]


# ---------------------------------------------------------------------------
# UL v2 symbolizer (re-zip consumer)


def ul_symbols(events):
    """Interleaved (IOI, class) symbol stream over a (bar, pos, class) event
    stream — the same normalization as train.py _ngram_symbols (class ids
    0..7; IOI = delta to the previous event's global position; the first
    event carries no IOI symbol), lifted to the exact rational lattice.
    Global position is bar + pos in bar units; any deterministic
    symbolization gives identical statistics for identical event streams."""
    syms = []
    prev = None
    for bar, pos, cls in events:
        g = Fraction(bar) + pos
        if prev is not None:
            syms.append(("ioi", g - prev))
        syms.append(("cls", NOTE_CLASSES.index(cls)))
        prev = g
    return syms


def ul_4gram_counts(events):
    """Counter of 4-grams over the UL symbol stream (the statistic the
    unlikelihood objective conditions on)."""
    from collections import Counter
    syms = ul_symbols(events)
    return Counter(tuple(syms[i:i + 4]) for i in range(len(syms) - 3))


# ---------------------------------------------------------------------------
# grammar FSM v2


class BarscriptFSM:
    """Stepwise legal-next-token mask for the BarScript grammar (§1.2 FSM v2
    row + §8.1 plan->decode hard constraints; cross-bar open-span carry).

    Usage: ``mask = fsm.legal_mask(); fsm.step(tok)``. ``step`` raises
    IllegalTokenError (with a state summary) when tok is outside the mask.
    ``gold_masks`` below batch-precomputes masks for training.
    """

    def __init__(self, spec, vocab=VOCAB, motif_hard=True):
        self.spec = spec
        self.v = vocab
        self.motif_hard = bool(motif_hard)
        self._denom_rev = {t: d for d, t in vocab.bar_denom.items()}
        self._motif_rev = {t: m for m, t in vocab.motif.items()}
        self.reset()

    def reset(self):
        self.phase = "start"  # start|denom|plan0..plan3|skel|skel_pos|real|real_dur
        self.bar_i = -1
        self.denom = None
        self.dens_lo = self.dens_hi = None
        self.budget = None  # remaining SPAN_START budget (None = unbounded)
        self.open_span = False  # cross-bar carry state
        self.last_pos = -1
        self.n_count = 0  # non-end skeleton positions this bar
        self.pending = None  # marker awaiting its position ("span_start"/"span_end")
        self.skeleton = []  # (idx, role) for realization typing
        self.real_i = 0
        self.done = False
        # MOTIF_REF hard constraint state
        self.sigs = []  # per completed bar: occupancy signature (Fractions)
        self.motif_ref = None  # committed MOTIF_REF name for the current bar
        self.repeat_target = None  # forced countable indices (None = soft bar)
        self.repeat_i = 0  # how many forced targets are already placed

    # -- MOTIF_REF occupancy bookkeeping -----------------------------------
    def _record_bar_sig(self):
        """Append the completed bar's occupancy signature (exact rational
        positions of countable skeleton entries).  Mirrors
        ``occupancy_signature`` on the normalized note stream: span starts
        count (they are onsets), span ends do not."""
        sig = tuple(Fraction(idx, self.denom)
                    for idx, role in self.skeleton if role != "span_end")
        while len(self.sigs) < self.bar_i:  # defensive (truncated stream)
            self.sigs.append(())
        if len(self.sigs) == self.bar_i:
            self.sigs.append(sig)
        else:
            self.sigs[self.bar_i] = sig

    def _repeat_indices(self, delta):
        """Forced countable indices for REPEAT_δ on this bar's lattice, or
        None when the reference cannot be honoured under the already
        committed BAR_DENOM / DENS / open-span state (plan-stage
        prevention)."""
        j = self.bar_i - delta
        if j < 0 or j >= len(self.sigs) or self.denom is None:
            return None
        sig = self.sigs[j]
        n = len(sig)
        if n < self.dens_lo or (self.dens_hi is not None and n > self.dens_hi):
            return None
        limit = min(self.denom, WINDOW)
        idxs = []
        prev = -1
        for pos in sig:
            x = pos * self.denom
            if x.denominator != 1:
                return None  # off this bar's lattice
            i = int(x)
            if not (0 <= i < limit) or i <= prev:
                return None
            idxs.append(i)
            prev = i
        if self.open_span and idxs and idxs[0] == 0:
            return None  # the carried span has nowhere to close first
        return idxs

    def _repeat_redundant(self, delta):
        """True when a smaller available δ' carries the identical signature:
        the realized bar would relabel as REPEAT_δ', so REPEAT_δ is masked
        (gold always emits the minimal δ, so this never touches gold)."""
        sig = self.sigs[self.bar_i - delta]
        return any(d < delta and 0 <= self.bar_i - d < len(self.sigs)
                   and self.sigs[self.bar_i - d] == sig
                   for d in self.spec.motif_deltas)

    def _repeat_remaining(self):
        return (self.repeat_target is not None
                and self.repeat_i < len(self.repeat_target))

    def _repeat_span_start_ok(self):
        """A span opened on the current target must leave a slot to close in
        before the NEXT target (the last target may carry across the bar)."""
        tgt, k = self.repeat_target, self.repeat_i
        return k + 1 >= len(tgt) or tgt[k + 1] - tgt[k] >= 2

    def _repeat_end_hi(self):
        """Highest legal SPAN_END index: strictly before the next forced
        target (or the bar's last slot once the targets are exhausted)."""
        if self._repeat_remaining():
            return self.repeat_target[self.repeat_i] - 1
        return self.denom - 1

    # -- masks -------------------------------------------------------------
    def _legal_positions(self):
        """Legal position indices for a plain (countable) skeleton position:
        strictly increasing, on-grid, leaving room for the DENS lower bound
        to stay reachable.  Under a committed REPEAT_δ the position is fully
        determined by the reference occupancy."""
        if self.denom is None:
            return range(0)
        if self.repeat_target is not None:
            if not self._repeat_remaining():
                return range(0)
            nxt = self.repeat_target[self.repeat_i]
            return range(nxt, nxt + 1)
        lo = self.last_pos + 1
        hi = self.denom - 1
        need_after = max(self.dens_lo - (self.n_count + 1), 0)
        return range(lo, min(hi, self.denom - 1 - need_after) + 1)

    def _skel_mask_repeat(self, mask):
        """Skeleton mask under a committed REPEAT_δ: countable positions are
        forced onto the reference occupancy, [REAL] stays masked until every
        target is placed, and the open-span carry keeps priority."""
        v = self.v
        remain = self._repeat_remaining()
        any_skel = False
        if self.open_span:
            if self.last_pos + 1 <= self._repeat_end_hi():
                mask[v.span_end] = True
                any_skel = True
        elif remain:
            mask[v.time0 + self.repeat_target[self.repeat_i]] = True
            any_skel = True
            if ((self.budget is None or self.budget > 0)
                    and self._repeat_span_start_ok()):
                mask[v.span_start] = True
        if not remain or not any_skel:
            # targets met -> [REAL] closes the skeleton; the `not any_skel`
            # arm is the same fail-open escape hatch as the soft path.
            mask[v.real] = True

    def _skel_mask(self, mask):
        v = self.v
        if self.repeat_target is not None:
            self._skel_mask_repeat(mask)
            return
        room = self.last_pos < self.denom - 1
        can_count = self.dens_hi is None or self.n_count < self.dens_hi
        any_skel = False
        if self.open_span:
            # carry: only SPAN_END may place content (no swallowed hits, no
            # nested spans); the span may also pass through ([REAL] below).
            if room:
                mask[v.span_end] = True
                any_skel = True
        else:
            if can_count:
                legal = self._legal_positions()
                if len(legal):
                    for i in legal:
                        mask[v.time0 + i] = True
                    any_skel = True
                    # SPAN_START is only offered when a countable position
                    # exists for it to land on: skel_pos reuses the same
                    # _legal_positions() range, so gating on non-empty legal
                    # keeps the never-dead-end guarantee (a room/budget-only
                    # gate could open SPAN_START into an empty position mask
                    # when the DENS room reservation exhausts the range,
                    # e.g. a carried span closing late in a high-L bar).
                    if self.budget is None or self.budget > 0:
                        mask[v.span_start] = True
        if self.n_count >= self.dens_lo or not any_skel:
            # fail-open escape hatch (docstring): [REAL] is always legal when
            # nothing else is, so generation can never dead-end. Gold
            # sequences satisfy [L, U] by construction and never need it.
            mask[v.real] = True

    def legal_mask(self):
        v = self.v
        mask = np.zeros(v.size, dtype=bool)
        if self.done:
            return mask
        ph = self.phase
        if ph == "start":
            mask[v.bar] = True
            if not self.open_span:
                mask[v.eos] = True
        elif ph == "denom":
            for t in v.bar_denom.values():
                mask[t] = True
        elif ph == "plan0":
            for t in v.bar_dens.values():
                mask[t] = True
        elif ph == "plan1":
            for t in v.span_budget.values():
                mask[t] = True
        elif ph == "plan2":
            for t in v.bar_sync.values():
                mask[t] = True
        elif ph == "plan3":
            mask[v.motif["new"]] = True
            for name in MOTIF_REF_NAMES[1:]:
                delta = int(name.rsplit("_", 1)[1])
                if self.bar_i - delta < 0:
                    continue
                if self.motif_hard and name.startswith("repeat_"):
                    # plan-stage prevention: never commit to a reference the
                    # skeleton stage could not reproduce, and never offer a
                    # δ that would relabel as a smaller one.
                    if (self._repeat_indices(delta) is None
                            or self._repeat_redundant(delta)):
                        continue
                mask[v.motif[name]] = True
        elif ph == "skel":
            self._skel_mask(mask)
        elif ph == "skel_pos":
            lo = self.last_pos + 1
            if self.pending == "span_end":
                # closing position: strictly increasing is the legality rule
                # (same-bar closes sit after the start; cross-bar closes are
                # later by construction); no DENS room reservation (an end
                # position is not countable).  Under a committed REPEAT_δ the
                # close must also land before the next forced target.
                hi = (self._repeat_end_hi() if self.repeat_target is not None
                      else self.denom - 1)
                for i in range(lo, hi + 1):
                    mask[v.time0 + i] = True
            else:
                for i in self._legal_positions():
                    mask[v.time0 + i] = True
        elif ph == "real":
            idx, role = self.skeleton[self.real_i]
            if role == "span_start":
                for cls in SPAN_CLASSES:
                    mask[v.note[cls]] = True
            elif role == "span_end":
                mask[v.note["end"]] = True
            else:
                for cls in HIT_CLASSES:
                    mask[v.note[cls]] = True
        elif ph == "real_dur":
            for t in v.span_dur.values():
                mask[t] = True
        return mask

    def state_summary(self):
        motif = self.motif_ref or "-"
        if self.repeat_target is not None:
            motif += f"{self.repeat_target}@{self.repeat_i}"
        return (f"bar={self.bar_i} phase={self.phase} denom={self.denom} "
                f"dens=[{self.dens_lo},{self.dens_hi}] count={self.n_count} "
                f"budget={self.budget} open_span={self.open_span} "
                f"last_pos={self.last_pos} real_i={self.real_i} "
                f"motif={motif}")

    # -- transition --------------------------------------------------------
    def step(self, tok):
        mask = self.legal_mask()
        if not (0 <= tok < len(mask)) or not mask[tok]:
            raise IllegalTokenError(
                f"token {tok} illegal at [{self.state_summary()}]")
        v = self.v
        ph = self.phase
        if ph == "start":
            if tok == v.eos:
                self.done = True
            else:
                self.bar_i += 1
                self.phase = "denom"
        elif ph == "denom":
            self.denom = self._denom_rev[tok]
            self.last_pos = -1
            self.n_count = 0
            self.skeleton = []
            self.real_i = 0
            self.pending = None
            self.motif_ref = None
            self.repeat_target = None
            self.repeat_i = 0
            self.phase = "plan0"
        elif ph == "plan0":
            bucket = next(d for d, t in v.bar_dens.items() if t == tok)
            self.dens_lo, self.dens_hi = self.spec.dens_bounds(bucket)
            self.phase = "plan1"
        elif ph == "plan1":
            b = next(s for s, t in v.span_budget.items() if t == tok)
            self.budget = None if b == N_SPAN_BUDGET - 1 else b
            self.phase = "plan2"
        elif ph == "plan2":
            self.phase = "plan3"
        elif ph == "plan3":
            name = self._motif_rev[tok]
            self.motif_ref = name
            if self.motif_hard and name.startswith("repeat_"):
                self.repeat_target = self._repeat_indices(
                    int(name.rsplit("_", 1)[1]))
            self.repeat_i = 0
            self.phase = "skel"
        elif ph == "skel":
            if tok == v.real:
                self._record_bar_sig()
                self.phase = "real" if self.skeleton else "start"
            elif tok == v.span_start:
                self.pending = "span_start"
                self.phase = "skel_pos"
            elif tok == v.span_end:
                self.pending = "span_end"
                self.phase = "skel_pos"
            else:  # plain position
                self._add_pos(tok - v.time0, "hit")
        elif ph == "skel_pos":
            role = self.pending
            self.pending = None
            self._add_pos(tok - v.time0, role)
            self.phase = "skel"
        elif ph == "real":
            _idx, role = self.skeleton[self.real_i]
            if role == "span_start":
                self.phase = "real_dur"
            else:
                self.real_i += 1
                if self.real_i == len(self.skeleton):
                    self.phase = "start"
        elif ph == "real_dur":
            self.real_i += 1
            self.phase = "real" if self.real_i < len(self.skeleton) else "start"

    def _add_pos(self, idx, role):
        self.last_pos = idx
        self.skeleton.append((idx, role))
        if role == "span_end":
            self.open_span = False
            return
        if role == "span_start":
            self.open_span = True
            if self.budget is not None:
                self.budget -= 1
        self.n_count += 1
        if self.repeat_target is not None:
            self.repeat_i += 1


def gold_masks(token_ids, spec, motif_hard=True):
    """Precompute the (L, V) bool legal-mask array for a gold sequence (the
    batch path the trainer consumes; the stepwise FSM is the correctness
    oracle it is verified against). Raises IllegalTokenError if the sequence
    leaves the grammar."""
    fsm = BarscriptFSM(spec, motif_hard=motif_hard)
    masks = np.zeros((len(token_ids), VOCAB.size), dtype=bool)
    for i, tok in enumerate(token_ids):
        masks[i] = fsm.legal_mask()
        fsm.step(tok)
    return masks
