"""Token vocabulary for chart sequences.

Sequence layout (decoder):
    [BOS] [COURSE_c] [LEVEL_l] [DENS_d] ([TIME_t] [NOTE_e])* [EOS]
Conditions may be UNK (condition dropout / unspecified at inference).
"""

import math
from dataclasses import dataclass
from numbers import Real

# --- audio / windowing constants (single source of truth) ---
SR = 22050
N_FFT = 2048
HOP = 256
N_MELS = 128
FPS = SR / HOP  # 86.1328125 frames/sec
WINDOW = 1728  # frames per training window (~20.06 s)
MAX_TGT = 960  # max decoder length (prefix + events + eos); 960 covers densest ura
# BarScript window (912 content tokens + prefix, train-split measured) with headroom.
# Frozen 2026-07-25 (spec ruling); was 768 through v1.8.

SLOTS = 96  # TJA lattice: slots per 4/4 measure (LCM of 16ths/triplets/32nds)
MEAS_MAX = WINDOW // SLOTS  # max whole measures per window in slot mode (18)
SLOTS_PER_BEAT = 24  # meter-general lattice (v3): 3/4 -> 72, 5/4 -> 120 slots
BEATS_MAX = WINDOW // SLOTS_PER_BEAT  # 72 beats of token space per window

COURSES = ["easy", "normal", "hard", "oni", "ura"]
N_LEVELS = 12  # levels clamped to 1..12
N_DENS = 16
DENS_BUCKET_NPS = 0.75  # bucket width in notes/sec

# canonical note classes; parsed TJA note_type strings are mapped onto these
NOTE_CLASSES = ["don", "ka", "don_big", "ka_big", "roll", "roll_big", "balloon", "end"]

# --- SoftChart 2.0 BarScript constants (SOFTCHART2_DESIGN §1.2/§8.1) ---
# Per-bar plan-token families. These are DISTINCT from the window-prefix
# condition families (pdens/sync): plan tokens live inside the bar stream at
# bar granularity, the older families are window-level prefix conditions.
N_DENS_D8 = 8       # per-bar onset-count density bucket (DENS_d8)
N_SPAN_BUDGET = 4   # per-bar span-start budget
N_BAR_SYNC = 6      # per-bar LHL syncopation band
MOTIF_DELTAS = (1, 2, 4, 8)  # lookback (bars) for REPEAT/VAR references
MOTIF_REF_NAMES = (
    ("new",)
    + tuple(f"repeat_{d}" for d in MOTIF_DELTAS)
    + tuple(f"var_{d}" for d in MOTIF_DELTAS)
)  # 9: NEW, REPEAT x {1,2,4,8}, VAR x {1,2,4,8}
N_MOTIF_REF = len(MOTIF_REF_NAMES)
# split-axis condition buckets (prefix conditions, per-axis CFG)
N_BIG_RATE = 8
N_SPAN_RATE = 8
N_STREAM = 6
# span duration buckets (log grid anchored on authored p50/p90/p99).
# TENTATIVE: the design doc fixes the grid family but not the bucket count;
# 16 is a placeholder pending the frozen label distribution report.
N_SPAN_DUR_BUCKETS = 16
# per-bar rational-lattice denominator support (BarScript skeleton positions
# are bar-local indices on a D_m grid, D_m = LCM of the bar's reduced event
# denominators; SLOTPLUS_DESIGN §2.1). 1..192 is a draft SUPERSET: it covers
# dyadic 64ths (denom 192) and the observed 5/7/9/11 tuplet families; the
# frozen support set comes from the S0 census. Bars needing D_m > 192 stay
# fail-closed (counted), never snapped.
MAX_BAR_DENOM = 192

# BarScript stage ids (stage embedding routing; SOFTCHART2_DESIGN §1.1)
STAGE_HEADER = 0
STAGE_PLAN = 1
STAGE_SKELETON = 2
STAGE_REALIZATION = 3
N_STAGES = 4

# slot+ TEMPO factorization heads (SLOTPLUS_DESIGN §2.2/§3.2/§3.3):
# beat period = p * 2^k with canonical p in [TEMPO_P_LO, TEMPO_P_HI) seconds
# (draft window; frozen from bpm_map before S1). The regression head predicts
# log p (no hard clamp); k is a classification label. TENTATIVE: the spec
# leaves k in Z; classification needs finite support, so k is clamped to
# [TEMPO_K_MIN, TEMPO_K_MIN + N_TEMPO_K).
TEMPO_P_LO = 1.0
TEMPO_P_HI = 2.0
TEMPO_K_MIN = -3
N_TEMPO_K = 7  # k in {-3..+3}

# mapping from strings observed in the dataset -> canonical class
# (observed in taiko-1000-parsed: Don, Ka, DonBig, KaBig, Roll, RollBig,
#  Balloon, BalloonAlt, EndOf)
NOTE_TYPE_MAP = {
    "Don": "don",
    "Ka": "ka",
    "DonBig": "don_big",
    "KaBig": "ka_big",
    "Roll": "roll",
    "RollBig": "roll_big",
    "Balloon": "balloon",
    "BalloonAlt": "balloon",  # kusudama, treated as balloon-class span
    "EndOf": "end",
}


@dataclass
class Vocab:
    pad: int = 0
    eos: int = 1
    bos: int = 2
    unk_cond: int = 3

    def __post_init__(self):
        base = 4
        self.course = {c: base + i for i, c in enumerate(COURSES)}
        base += len(COURSES)
        self.level = {l: base + l - 1 for l in range(1, N_LEVELS + 1)}
        base += N_LEVELS
        self.dens = {d: base + d for d in range(N_DENS)}
        base += N_DENS
        self.note = {n: base + i for i, n in enumerate(NOTE_CLASSES)}
        base += len(NOTE_CLASSES)
        self.time0 = base
        base += WINDOW
        # v2+ tokens appended at the end so older checkpoints stay loadable
        self.sep = base  # separates prev-window context from conditions
        self.sib = base + 1  # marks the sibling-chart (easier course) segment
        self.style = {s: base + 2 + s for s in range(8)}  # charting-intent codes
        self.sync = {s: base + 10 + s for s in range(6)}  # LHL syncopation bands
        # plan-realize tokens: [PLAN] marker + block density (8) + block flags (3)
        self.plan = base + 16
        self.pdens = {d: base + 17 + d for d in range(8)}
        self.pflag = {f: base + 25 + f for f in range(3)}  # 0 none / 1 gap / 2 climax
        self.mask = base + 28  # type-infill placeholder
        self.cplx = {c: base + 29 + c for c in range(6)}  # rhythmic-complexity band
        # dual-mode models: explicit output-semantics switch (v3). mode "slot"
        # = positions are lattice indices on a supplied grid; "time" = frames.
        self.mode = {"slot": base + 35, "time": base + 36}
        # --- SoftChart 2.0 BarScript tokens (append-only; older checkpoints
        # keep every pre-existing id). SOFTCHART2_DESIGN §1.2/§8.1. ---
        b2 = base + 37
        self.bar = b2  # [BAR] header-stage marker (TEMPO/MEASURE/k anchor)
        self.real = b2 + 1  # [REAL] realization-stage marker
        self.bar_dens = {d: b2 + 2 + d for d in range(N_DENS_D8)}
        self.span_budget = {s: b2 + 10 + s for s in range(N_SPAN_BUDGET)}
        self.bar_sync = {s: b2 + 14 + s for s in range(N_BAR_SYNC)}
        self.motif = {m: b2 + 20 + i for i, m in enumerate(MOTIF_REF_NAMES)}
        self.big_rate = {r: b2 + 29 + r for r in range(N_BIG_RATE)}
        self.span_rate = {r: b2 + 37 + r for r in range(N_SPAN_RATE)}
        self.stream = {s: b2 + 45 + s for s in range(N_STREAM)}
        self.span_dur = {b: b2 + 51 + b for b in range(N_SPAN_DUR_BUCKETS)}
        # --- BarScript skeleton markers + notation denominator (append-only,
        # 2026-07-24; SOFTCHART2_DESIGN §1.1 skeleton stage, §1.3 span object).
        # Skeleton POSITION tokens reuse the time0 family as BAR-LOCAL lattice
        # indices (v3 slot-mode precedent: "positions are lattice indices on a
        # supplied grid"), so hit token counts match the flat encoding exactly.
        b3 = b2 + 51 + N_SPAN_DUR_BUCKETS
        self.span_start = b3  # skeleton marker: next position opens a span
        self.span_end = b3 + 1  # skeleton marker: next position closes the span
        self.bar_denom = {d: b3 + 2 + (d - 1)
                          for d in range(1, MAX_BAR_DENOM + 1)}
        self.size = b3 + 2 + MAX_BAR_DENOM

        self.id2note = {v: k for k, v in self.note.items()}
        self.id2course = {v: k for k, v in self.course.items()}

    def time(self, frame: int) -> int:
        assert 0 <= frame < WINDOW
        return self.time0 + frame

    def is_time(self, tok: int) -> bool:
        return self.time0 <= tok < self.time0 + WINDOW

    def is_note(self, tok: int) -> bool:
        return tok in self.id2note

    def dens_bucket(self, nps: float) -> int:
        return min(N_DENS - 1, max(0, int(nps / DENS_BUCKET_NPS)))


VOCAB = Vocab()


def v2_token_ids(vocab=VOCAB):
    """Explicit name -> id map for every SoftChart 2.0 appended token."""
    ids = {"BAR": vocab.bar, "REAL": vocab.real}
    ids.update({f"DENS_D8_{d}": t for d, t in vocab.bar_dens.items()})
    ids.update({f"SPAN_BUDGET_{s}": t for s, t in vocab.span_budget.items()})
    ids.update({f"BAR_SYNC_{s}": t for s, t in vocab.bar_sync.items()})
    ids.update({f"MOTIF_{m.upper()}": t for m, t in vocab.motif.items()})
    ids.update({f"BIG_RATE_{r}": t for r, t in vocab.big_rate.items()})
    ids.update({f"SPAN_RATE_{r}": t for r, t in vocab.span_rate.items()})
    ids.update({f"STREAM_{s}": t for s, t in vocab.stream.items()})
    ids.update({f"SPAN_DUR_{b}": t for b, t in vocab.span_dur.items()})
    ids["SPAN_START"] = vocab.span_start
    ids["SPAN_END"] = vocab.span_end
    ids.update({f"BAR_DENOM_{d}": t for d, t in vocab.bar_denom.items()})
    return ids


V2_TOKEN_IDS = v2_token_ids()


SIB_EVENTS = 12  # fixed number of sibling-chart events in the prefix (24 tokens)


PLAN_SLOTS = 5  # fixed number of plan blocks in the prefix (2 tokens each)


def complexity_band(frames):
    """IOI-class entropy of a window's hit sequence -> band 0-5 (rhythmic
    complexity independent of raw density)."""
    import math
    hits = sorted(f for f, c in frames if NOTE_CLASSES[c] in ("don","ka","don_big","ka_big"))
    if len(hits) < 4:
        return 0
    iois = [b - a for a, b in zip(hits, hits[1:]) if b > a]
    if not iois:
        return 0
    from collections import Counter
    cls = Counter(int(round(math.log2(max(i,1)) * 2)) for i in iois)
    tot = sum(cls.values())
    ent = -sum((n/tot) * math.log2(n/tot) for n in cls.values())
    return min(5, int(ent / 0.5))


def encode_window(vocab, course, level, notes, cond_drop=0.0, rng=None, ctx_types=None,
                  sib_pairs=None, style=None, sync_band=None, plan_slice=None,
                  complexity=None, mode=None, duration_seconds=None):
    """Build a token sequence for one window.

    notes: list of (frame_idx, note_class_id) sorted by frame, frame in [0, WINDOW).
    ctx_types: optional list of note-class ids from the previous window's tail
        (pattern continuity context, v2). Encoded as [BOS] ctx.. [SEP] conds..
    sib_pairs: optional list of (frame, class_id) events from an easier course of
        the same song (skeleton hint, easy⊂hard). Fixed SIB_EVENTS slots,
        missing slots filled with UNK. Encoded after [SIB].
    Returns (tokens, prefix_len) where loss should be applied after the prefix.
    """
    if duration_seconds is None:
        duration_seconds = WINDOW / FPS
    if (not isinstance(duration_seconds, Real)
            or isinstance(duration_seconds, bool)
            or not math.isfinite(duration_seconds)
            or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive and finite")
    n_hits = sum(1 for _, c in notes if NOTE_CLASSES[c] not in ("end",))
    nps = n_hits / float(duration_seconds)
    d = vocab.dens_bucket(nps)

    def maybe(tok):
        if cond_drop > 0 and rng is not None and rng.random() < cond_drop:
            return vocab.unk_cond
        return tok

    lvl = max(1, min(N_LEVELS, level if level and level > 0 else 1))
    seq = [vocab.bos]
    if mode is not None:  # dual-mode: explicit output-semantics token (never dropped)
        seq.append(vocab.mode[mode])
    if ctx_types is not None:  # fixed-length, left-padded with UNK (id < 0 = pad)
        seq += [vocab.note[NOTE_CLASSES[c]] if c >= 0 else vocab.unk_cond
                for c in ctx_types]
        seq.append(vocab.sep)
    if sib_pairs is not None:
        seq.append(vocab.sib)
        pairs = list(sib_pairs)[:SIB_EVENTS]
        for f, c in pairs:
            seq += [vocab.time(int(f)), vocab.note[NOTE_CLASSES[c]]]
        seq += [vocab.unk_cond] * (2 * (SIB_EVENTS - len(pairs)))
    seq += [
        maybe(vocab.course[course]),
        maybe(vocab.level[lvl]),
        maybe(vocab.dens[d]),
    ]
    if style is not None:  # charting-intent code (style >= 0; -1 = unknown)
        seq.append(maybe(vocab.style[style]) if style >= 0 else vocab.unk_cond)
    if sync_band is not None:  # LHL syncopation band (groove-intensity control)
        seq.append(maybe(vocab.sync[sync_band]) if sync_band >= 0 else vocab.unk_cond)
    if complexity is not None:  # rhythmic-complexity band (difficulty beyond density)
        seq.append(maybe(vocab.cplx[complexity]) if complexity >= 0 else vocab.unk_cond)
    if plan_slice is not None:  # song-level plan blocks overlapping this window
        seq.append(vocab.plan)
        blocks = list(plan_slice)[:PLAN_SLOTS]
        for d8, fl in blocks:
            seq += [maybe(vocab.pdens[min(7, max(0, d8))]),
                    maybe(vocab.pflag[min(2, max(0, fl))])]
        seq += [vocab.unk_cond] * (2 * (PLAN_SLOTS - len(blocks)))
    prefix_len = len(seq)
    for f, c in notes:
        seq.append(vocab.time(int(f)))
        seq.append(vocab.note[NOTE_CLASSES[c]])
    seq.append(vocab.eos)
    return seq, prefix_len


def decode_tokens(vocab, tokens):
    """Token ids -> list of (frame_idx, note_class_name). Ignores malformed pairs."""
    out = []
    cur_t = None
    for tok in tokens:
        if vocab.is_time(tok):
            cur_t = tok - vocab.time0
        elif tok in vocab.id2note and cur_t is not None:
            out.append((cur_t, vocab.id2note[tok]))
        elif tok == vocab.eos:
            break
    return out
