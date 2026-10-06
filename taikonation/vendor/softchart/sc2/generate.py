"""Inference: full-song chart generation with constrained autoregressive decoding.

Features:
- structural constraints via logit masking: TIME/NOTE alternation, strictly
  increasing times, minimum playable gap (4 frames ~= 46 ms);
- classifier-free guidance (cfg_w > 1): contrasts conditional vs
  condition-UNK logits to strengthen condition adherence;
- previous-window pattern context (v2 models): two-pass generation, pass 2
  conditions each window on the tail of the previous window's pass-1 output.
"""

import json
from fractions import Fraction

import numpy as np
import torch

from .checkpoint import (attach_generator_capabilities,
                         parse_generator_checkpoint)
from .meter import resolve_meter_layout
from .model import ChartModel
from .vocab import FPS, MAX_TGT, N_LEVELS, N_MELS, NOTE_CLASSES, VOCAB, WINDOW

HIT_CLASSES = ("don", "ka", "don_big", "ka_big")
SPAN_CLASSES = ("roll", "roll_big", "balloon")


def _dev_type(device):
    return device.split(":")[0] if isinstance(device, str) else device.type


def _autocast(device):
    """bf16 autocast on CUDA; no-op elsewhere (MPS/CPU run fp32)."""
    import contextlib

    if _dev_type(device) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
CTX_LEN = 12  # fixed-length previous-window context (note-type tokens)
MIN_GAP_FRAMES = 4  # ~46 ms minimum inter-note gap enforced during decoding
LATTICE_RATIOS = (1 / 3, 0.5, 2 / 3, 1.0, 4 / 3, 1.5, 2.0, 3.0)  # allowed IOI ratios


def load_model(ckpt_path, device="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state, _args, config, capabilities = parse_generator_checkpoint(checkpoint)
    model = ChartModel(**config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    attach_generator_capabilities(model, capabilities)
    return model


def load_hf(repo_or_dir, device="cuda"):
    """Load a Hugging Face SoftChartGenerator (safetensors) and attach the
    capability flags that generate_song() reads, so HF models are drop-in."""
    from .hf import SoftChartGenerator

    hf = SoftChartGenerator.from_pretrained(repo_or_dir).to(device).eval()
    m = hf.net
    caps = hf.capabilities
    if not isinstance(caps, dict):
        raise ValueError("HF generator capabilities must be an explicit mapping")
    attach_generator_capabilities(m, caps)
    if (m.section_encoder is not None) != m._hierarchical_ctx:
        raise ValueError("HF hierarchy capability disagrees with its architecture")
    if bool(m.gsum_proj is not None) != bool(caps.get("global_ctx", False)):
        # New V1.8 artifacts include global_ctx explicitly.  Requiring it here
        # prevents a semantically different flat-global model from loading as
        # a local-only generator.
        raise ValueError("HF global-context capability disagrees with architecture")
    return m


def build_prefix(course=None, level=None, density_bucket=None, ctx_types=None,
                 sib_pairs=None, style=None, sync_band=None,
                 plan_slice=None, mode=None):
    v = VOCAB
    seq = [v.bos]
    if mode is not None:  # dual-mode output-semantics switch
        seq.append(v.mode[mode])
    if ctx_types is not None:
        ctx = list(ctx_types)[-CTX_LEN:]
        ctx = [v.unk_cond] * (CTX_LEN - len(ctx)) + [v.note[NOTE_CLASSES[c]] if isinstance(c, int) else v.note[c] for c in ctx]
        seq += ctx + [v.sep]
    if sib_pairs is not None:  # fixed 12-event sibling segment (unk-padded)
        from .vocab import SIB_EVENTS

        seq.append(v.sib)
        pairs = list(sib_pairs)[:SIB_EVENTS]
        for f, c in pairs:
            cls = c if isinstance(c, int) else NOTE_CLASSES.index(c)
            seq += [v.time(int(f)), v.note[NOTE_CLASSES[cls]]]
        seq += [v.unk_cond] * (2 * (SIB_EVENTS - len(pairs)))
    seq += [
        v.course[course] if course else v.unk_cond,
        v.level[max(1, min(N_LEVELS, level))] if level else v.unk_cond,
        v.dens[density_bucket] if density_bucket is not None else v.unk_cond,
    ]
    if style is not None:  # -1 = style-capable model, no specific style requested
        seq.append(v.style[style] if style >= 0 else v.unk_cond)
    if sync_band is not None:
        seq.append(v.sync[sync_band] if sync_band >= 0 else v.unk_cond)
    if plan_slice is not None:  # fixed PLAN_SLOTS blocks, unk-padded
        from .vocab import PLAN_SLOTS

        seq.append(v.plan)
        blocks = list(plan_slice)[:PLAN_SLOTS]
        for d8, fl in blocks:
            seq += [v.pdens[min(7, max(0, int(d8)))], v.pflag[min(2, max(0, int(fl)))]]
        seq += [v.unk_cond] * (2 * (PLAN_SLOTS - len(blocks)))
    return seq


def _conditioning_token_ids():
    """Token ids whose *values* CFG is allowed to remove.

    Structural markers (BOS/SEP/SIB/PLAN/mode), prior-window note context, and
    sibling-chart events deliberately stay intact.  Positional masking such as
    ``prefix[:, -n:]`` is wrong as soon as a variable-size plan or another
    field follows the basic course/level/density tuple.
    """
    fields = (
        VOCAB.course, VOCAB.level, VOCAB.dens, VOCAB.style, VOCAB.sync,
        VOCAB.cplx, VOCAB.pdens, VOCAB.pflag,
        # SoftChart 2.0 split-axis knobs (never emitted by v1.8 prefixes)
        VOCAB.big_rate, VOCAB.span_rate, VOCAB.stream,
    )
    return frozenset(tok for field in fields for tok in field.values())


_CONDITIONING_IDS = _conditioning_token_ids()


def _make_unconditional_prefixes(prefixes):
    """Return CFG prefixes with named condition values replaced by UNK."""
    unc = prefixes.clone()
    is_condition = torch.zeros_like(unc, dtype=torch.bool)
    for tok in _CONDITIONING_IDS:
        is_condition |= unc == tok
    return unc.masked_fill(is_condition, VOCAB.unk_cond)


@torch.no_grad()
def decode_windows(model, mels, prefixes, device="cuda", temperature=1.0, top_p=0.95,
                   greedy=False, seed=0, cfg_w=0.0, gsum=None, pos_buckets=None,
                   lattice=False, min_gap=MIN_GAP_FRAMES,
                   position_limits=None):
    """mels: (B, n_mels, WINDOW); prefixes: list of B equal-length token lists.
    Returns list of lists of (frame, note_class)."""
    v = VOCAB
    B = mels.shape[0]
    plen = len(prefixes[0])
    assert all(len(p) == plen for p in prefixes)
    use_cfg = cfg_w and cfg_w > 0 and not greedy

    gen_device = "cpu" if _dev_type(device) == "mps" else device
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(seed)
    mels_dev = mels.to(device)
    with _autocast(device):
        memory = model.encode(
            mels_dev,
            gsum=gsum.to(device) if gsum is not None else None,
            pos_bucket=pos_buckets.to(device) if pos_buckets is not None else None,
        )
        # clean_phase (v1.6): inject the grid-phase channels into the memory the
        # DECODER reads. No-op for legacy models (in_ch already carried phase).
        memory = model.phase_mem(memory, mels_dev)

    seqs = torch.tensor(prefixes, dtype=torch.long, device=device)
    if use_cfg:  # rows B..2B: same ctx, conditions replaced by UNK
        unc = _make_unconditional_prefixes(seqs)
        seqs = torch.cat([seqs, unc], dim=0)
        memory = torch.cat([memory, memory], dim=0)

    if position_limits is None:
        position_limits = torch.full((B,), WINDOW, dtype=torch.long, device=device)
    else:
        position_limits = torch.as_tensor(position_limits, dtype=torch.long,
                                          device=device)
        if position_limits.shape != (B,):
            raise ValueError(f"position_limits must have shape ({B},), got "
                             f"{tuple(position_limits.shape)}")
        if ((position_limits < 1) | (position_limits > WINDOW)).any():
            raise ValueError(f"position_limits must be within [1, {WINDOW}]")

    done = torch.zeros(B, dtype=torch.bool, device=device)
    last_time = torch.full((B,), -min_gap, dtype=torch.long, device=device)
    expect_note = torch.zeros(B, dtype=torch.bool, device=device)
    note_ids = torch.tensor(sorted(v.note.values()), device=device)
    # lattice state: previous inter-onset interval (frames); -1 = unknown
    prev_ioi = torch.full((B,), -1, dtype=torch.long, device=device)
    ratios = torch.tensor(LATTICE_RATIOS, device=device)

    for _ in range(MAX_TGT - plen):
        with _autocast(device):
            logits_all = model.decode(seqs, memory)[:, -1].float()
        if use_cfg:
            logits = logits_all[B:] + cfg_w * (logits_all[:B] - logits_all[B:])
        else:
            logits = logits_all

        mask = torch.full_like(logits, float("-inf"))
        note_row = torch.full((logits.shape[1],), float("-inf"), device=device)
        note_row[note_ids] = 0.0
        mask[expect_note] = note_row
        idx = (~expect_note).nonzero(as_tuple=True)[0]
        if len(idx):
            time_pos = torch.arange(WINDOW, device=device).unsqueeze(0)
            sub = torch.full((len(idx), logits.shape[1]), float("-inf"), device=device)
            sub[:, v.eos] = 0.0
            ok = ((time_pos >= (last_time[idx] + min_gap).unsqueeze(1))
                  & (time_pos < position_limits[idx].unsqueeze(1)))
            tmask = torch.where(
                ok, torch.zeros_like(sub[:, :WINDOW]),
                torch.full_like(sub[:, :WINDOW], float("-inf")),
            )
            if lattice:
                # soft rhythmic-lattice constraint: once an IOI is established,
                # penalize next onsets whose IOI ratio is not a musical fraction
                # (kills "between two subdivisions" notes at the source)
                pi = prev_ioi[idx]
                active = (pi >= min_gap) & (pi <= 86)  # sub-second IOIs only
                if active.any():
                    delta = (time_pos - last_time[idx].unsqueeze(1)).float()  # (n, W)
                    rel = delta / pi.unsqueeze(1).clamp(min=1).float()
                    err = (rel.unsqueeze(-1) / ratios - 1.0).abs().min(-1).values
                    bad = (err > 0.13) & (rel < 4.0) & (delta > 0)
                    pen = torch.where(bad & active.unsqueeze(1),
                                      torch.full_like(delta, -6.0),
                                      torch.zeros_like(delta))
                    tmask = tmask + pen
            sub[:, v.time0 : v.time0 + WINDOW] = tmask
            mask[idx] = sub
        logits = logits + mask

        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp < top_p
            keep[:, 0] = True
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            if _dev_type(device) == "mps":
                pick = torch.multinomial(sp.cpu(), 1, generator=gen).squeeze(1).to(device)
            else:
                pick = torch.multinomial(sp, 1, generator=gen).squeeze(1)
            nxt = si[torch.arange(B, device=device), pick]

        nxt = torch.where(done, torch.full_like(nxt, v.pad), nxt)
        step_tok = torch.cat([nxt, nxt], dim=0) if use_cfg else nxt
        seqs = torch.cat([seqs, step_tok.unsqueeze(1)], dim=1)
        is_time = (nxt >= v.time0) & (nxt < v.time0 + WINDOW)
        new_time = nxt - v.time0
        upd = is_time & (last_time >= 0)
        prev_ioi = torch.where(upd, new_time - last_time, prev_ioi)
        last_time = torch.where(is_time, new_time, last_time)
        expect_note = is_time
        done = done | (nxt == v.eos)
        if done.all():
            break

    out = []
    for b in range(B):
        toks = seqs[b, plen:].tolist()
        events = []
        cur_t = None
        for t in toks:
            if t == v.eos or t == v.pad:
                break
            if v.time0 <= t < v.time0 + WINDOW:
                cur_t = t - v.time0
            elif t in v.id2note and cur_t is not None:
                events.append((cur_t, v.id2note[t]))
        out.append(events)
    return out


def _decode_pass(model, wins, starts, course, level, density_bucket, ctx_lists,
                 device, greedy, temperature, top_p, seed, cfg_w, batch_windows,
                 gsum=None, T=None, lattice=False, sib_default=False, style=None,
                 sync_band=None, plan_blocks=None, plan_default=False, mode=None,
                 on_chunk=None):
    all_events = []
    per_window = []
    for i in range(0, len(wins), batch_windows):
        chunk = torch.stack(wins[i : i + batch_windows])
        prefixes = [
            build_prefix(course, level, density_bucket, mode=mode,
                         ctx_types=(ctx_lists[i + j] if ctx_lists is not None else None),
                         sib_pairs=([] if sib_default else None), style=style,
                         sync_band=sync_band,
                         plan_slice=(
                             [(b[2], b[3]) for b in plan_blocks
                              if b[1] > starts[i + j] / FPS
                              and b[0] < (starts[i + j] + WINDOW) / FPS]
                             if plan_blocks is not None
                             else ([] if plan_default else None)))
            for j in range(chunk.shape[0])
        ]
        g = pb = None
        if gsum is not None:
            g = gsum.unsqueeze(0).expand(chunk.shape[0], -1, -1)
            n_pos = (100 if getattr(model, "_hierarchical_ctx", False)
                     else getattr(getattr(model, "pos_emb", None),
                                  "num_embeddings", 16))
            pb = torch.tensor(
                [min(n_pos - 1, int(n_pos * starts[i + j] / max(T, 1)))
                 for j in range(chunk.shape[0])],
                dtype=torch.long)
        limits = [min(WINDOW, max(1, int(T) - starts[i + j]))
                  for j in range(chunk.shape[0])]
        evs = decode_windows(model, chunk, prefixes, device=device, greedy=greedy,
                             temperature=temperature, top_p=top_p, seed=seed + i,
                             cfg_w=cfg_w, gsum=g, pos_buckets=pb, lattice=lattice,
                             min_gap=MIN_GAP_FRAMES, position_limits=limits)
        for j, events in enumerate(evs):
            per_window.append(events)
            t_off = starts[i + j] / FPS
            for f, cls in events:
                all_events.append((t_off + f / FPS, cls))
        if on_chunk:
            on_chunk(min(i + batch_windows, len(wins)), len(wins))
    return all_events, per_window


@torch.no_grad()
def generate_song(model, mel, course, level=None, density_bucket=None, device="cuda",
                  greedy=False, temperature=1.0, top_p=0.95, seed=0, batch_windows=8,
                  cfg_w=0.0, use_ctx=None, lattice=False, style=None, sync_band=None,
                  plan=None, on_progress=None):
    """mel: (n_mels, T). Returns dict with 'hits' and 'spans' in seconds."""
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel.astype(np.float32))
    T = mel.shape[1]
    starts = list(range(0, max(T, 1), WINDOW))
    dual = getattr(model, "_dual", False)
    wins = []
    for s in starts:
        w = mel[:, s : s + WINDOW]
        if w.shape[1] < WINDOW:
            w = torch.nn.functional.pad(w, (0, WINDOW - w.shape[1]), value=float(np.log(1e-5)))
        if dual:  # gridless time mode: phase channels = -1
            w = torch.cat([w, torch.full((2, WINDOW), -1.0)], dim=0)
        wins.append(w)

    if use_ctx is None:
        use_ctx = getattr(model, "_has_ctx", False)
    sib_default = getattr(model, "_has_sib", False)
    style_val = None
    if getattr(model, "_has_style", False):
        style_val = style if style is not None else -1
    sync_val = None
    if getattr(model, "_has_sync", False):
        sync_val = sync_band if sync_band is not None else -1
    plan_blocks = plan if getattr(model, "_has_plan", False) else None
    plan_default = getattr(model, "_has_plan", False) and plan is None
    gsum = None
    if getattr(model, "_has_gctx", False):
        from .data import song_summary

        gsum = torch.from_numpy(song_summary(mel.detach().cpu().numpy()))

    all_events, per_window = _decode_pass(
        model, wins, starts, course, level, density_bucket,
        None if not use_ctx else [[] for _ in wins],  # pass 1: empty ctx
        device, greedy, temperature, top_p, seed, cfg_w, batch_windows,
        gsum=gsum, T=T, lattice=lattice, sib_default=sib_default, style=style_val,
        sync_band=sync_val, plan_blocks=plan_blocks, plan_default=plan_default,
        mode=("time" if dual else None), on_chunk=on_progress)

    if use_ctx and len(wins) > 1:
        # pass 2: condition each window on the tail of the previous window's pass-1 output
        ctx_lists = [[]]
        for w_ev in per_window[:-1]:
            tail = [NOTE_CLASSES.index(c) for _, c in w_ev if c in HIT_CLASSES][-CTX_LEN:]
            ctx_lists.append(tail)
        all_events, per_window = _decode_pass(
            model, wins, starts, course, level, density_bucket, ctx_lists,
            device, greedy, temperature, top_p, seed, cfg_w, batch_windows,
            gsum=gsum, T=T, lattice=lattice, sib_default=sib_default, style=style_val,
            sync_band=sync_val, plan_blocks=plan_blocks, plan_default=plan_default,
            mode=("time" if dual else None))

    all_events.sort(key=lambda e: e[0])
    hits, spans = [], []
    open_span = None
    last_hit_t = -1.0
    for t, cls in all_events:
        if cls in HIT_CLASSES:
            if open_span is not None:
                # Missing END is malformed decoder output.  Drop the orphaned
                # span and preserve the authored hit; inventing an endpoint or
                # swallowing hits would silently rewrite the sequence.
                open_span = None
            if t - last_hit_t < 0.025:
                continue
            hits.append({"t": round(t, 4), "type": cls})
            last_hit_t = t
        elif cls in SPAN_CLASSES:
            if open_span is None:
                open_span = (t, cls)
        elif cls == "end":
            if open_span is not None:
                if t - open_span[0] > 0.05:
                    spans.append({"t0": round(open_span[0], 4), "t1": round(t, 4),
                                  "type": open_span[1]})
                # END always consumes the open span.  A too-short span is
                # malformed and dropped; keeping it open would let a later END
                # close the wrong start and swallow subsequent span starts.
                open_span = None
    return {"hits": hits, "spans": spans, "course": course, "level": level,
            "density_bucket": density_bucket}


def _grid_edges(grid, downbeats):
    """Return explicitly supplied measure edges without inventing a tail."""
    if len(downbeats) == 0:
        raise ValueError("slot generation requires at least one downbeat")
    if not np.all(np.isfinite(downbeats)) or (np.diff(downbeats) <= 0).any():
        raise ValueError("grid downbeats must be finite and strictly increasing")
    if "measure_edges" not in grid:
        raise ValueError(
            "slot generation requires explicit measure_edges (starts plus a "
            "trusted terminal edge); terminal duration is never extrapolated"
        )
    edges = np.asarray(grid["measure_edges"], dtype=np.float64)
    if (len(edges) != len(downbeats) + 1
            or not np.allclose(edges[:-1], downbeats, rtol=0, atol=1e-6)
            or not np.all(np.isfinite(edges))
            or (np.diff(edges) <= 0).any()):
        raise ValueError("grid measure_edges must be valid starts plus one end")
    return edges


@torch.no_grad()
def generate_song_slot(model, mel, grid, course, level=None, density_bucket=None,
                       device="cuda", greedy=False, temperature=1.0, top_p=0.95,
                       seed=0, batch_windows=8, plan=None, on_progress=None):
    """Meter-aware generation on the exact 96-slots-per-4/4 TJA lattice.

    Decoder positions are slot-linear across complete measures. Rational
    meters such as 1/16, 5/8, 7/8, and 8/4 are represented by
    ``96 * numerator / denominator`` slots. No quantization step exists:
    ``hits_slots`` are exact per-measure lattice positions.

    mel: (n_mels, T); grid: trusted rational layout with explicit
    ``measure_edges`` including its terminal edge.
    Returns {hits, spans (seconds, for rendering/metrics),
             hits_slots [(measure, slot, cls)], spans_slots, n_measures}.
    """
    from .vocab import SLOTS

    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel.astype(np.float32))
    T = mel.shape[1]
    db = np.asarray(grid["downbeats"], np.float64)
    measure_slots, measure_num, measure_den, quarter_beats = \
        resolve_meter_layout(grid, len(db))
    if "measure_slot_safe" in grid:
        slot_safe = np.asarray(grid["measure_slot_safe"])
        if (slot_safe.dtype != np.bool_ or slot_safe.ndim != 1
                or len(slot_safe) != len(db)):
            raise ValueError(
                "measure_slot_safe must be one boolean per measure")
        unsafe = np.flatnonzero(~slot_safe)
        if len(unsafe):
            raise ValueError(
                "slot generation refuses unsafe BPM/DELAY/off-lattice "
                f"measures: {unsafe.astype(int).tolist()}")
    edges = _grid_edges(grid, db)

    # meta = (first_measure_idx, measure_count, cumulative local slot offsets)
    wins, metas = [], []
    j = 0
    while j < len(db):
        K = 0
        nslots = 0
        while (j + K < len(db)
               and nslots + int(measure_slots[j + K]) <= WINDOW
               and (edges[j + K + 1] - edges[j]) * FPS <= WINDOW):
            nslots += int(measure_slots[j + K])
            K += 1
        if K == 0:
            raise ValueError(
                f"measure {j} does not fit the {WINDOW}-frame/{WINDOW}-slot "
                "slot window"
            )
        t0 = edges[j]
        f0 = int(round(t0 * FPS))
        src = max(1, int(round((edges[j + K] - t0) * FPS)))
        # Negative chart offsets are real; pad the pre-audio region instead of
        # letting Python's negative slice wrap around to the song tail.
        x = torch.full((mel.shape[0], src), float(np.log(1e-5)),
                       dtype=mel.dtype, device=mel.device)
        src0, src1 = max(0, f0), min(T, f0 + src)
        if src1 > src0:
            dst0 = src0 - f0
            x[:, dst0:dst0 + (src1 - src0)] = mel[:, src0:src1]
        if x.shape[1] < WINDOW:
            x = torch.nn.functional.pad(x, (0, WINDOW - x.shape[1]), value=float(np.log(1e-5)))
        ph = torch.full((2, WINDOW), -1.0)
        ef = (edges[j : j + K + 1] - t0) * FPS
        for m in range(K):
            a, b = ef[m], ef[m + 1]
            i0, i1 = int(np.ceil(a - 1e-6)), min(int(np.ceil(b - 1e-6)), WINDOW)
            if i1 <= i0:
                continue
            frac = (torch.arange(i0, i1, dtype=torch.float32) - a) / max(b - a, 1e-6)
            ph[0, i0:i1] = frac
            ph[1, i0:i1] = (frac * float(quarter_beats[j + m])) % 1.0
        wins.append(torch.cat([x, ph], dim=0))
        local_offsets = np.concatenate([
            np.array([0], dtype=np.int64),
            np.cumsum(measure_slots[j:j + K], dtype=np.int64),
        ])
        metas.append((j, K, local_offsets))
        j += K

    sib_default = getattr(model, "_has_sib", False)
    style_val = -1 if getattr(model, "_has_style", False) else None
    plan_blocks = plan if getattr(model, "_has_plan", False) else None
    plan_default = getattr(model, "_has_plan", False) and plan is None
    gsum = None
    if getattr(model, "_has_gctx", False):
        from .data import song_summary

        gsum = torch.from_numpy(song_summary(mel.detach().cpu().numpy()))

    events = []  # (measure_global, slot, cls) in decode order
    for i0 in range(0, len(wins), batch_windows):
        chunk = torch.stack(wins[i0 : i0 + batch_windows])
        prefixes = []
        for b_ in range(chunk.shape[0]):
            jj, KK, _ = metas[i0 + b_]
            t0, t1 = edges[jj], edges[jj + KK]
            prefixes.append(build_prefix(
                course, level, density_bucket,
                mode=("slot" if getattr(model, "_dual", False) else None),
                sib_pairs=([] if sib_default else None), style=style_val,
                plan_slice=([(b[2], b[3]) for b in plan_blocks
                             if b[1] > t0 and b[0] < t1] if plan_blocks is not None
                            else ([] if plan_default else None))))
        limits = [int(metas[i0 + b_][2][-1])
                  for b_ in range(chunk.shape[0])]
        g = pb = None
        if gsum is not None:
            g = gsum.unsqueeze(0).expand(chunk.shape[0], -1, -1)
            n_pos = (100 if getattr(model, "_hierarchical_ctx", False)
                     else getattr(getattr(model, "pos_emb", None),
                                  "num_embeddings", 16))
            pb = torch.tensor([
                min(n_pos - 1, max(0, int(
                    n_pos * edges[metas[i0 + b_][0]] * FPS / max(T, 1))))
                for b_ in range(chunk.shape[0])
            ], dtype=torch.long)
        evs = decode_windows(model, chunk, prefixes, device=device, greedy=greedy,
                             temperature=temperature, top_p=top_p, seed=seed + i0,
                             min_gap=2, position_limits=limits,
                             gsum=g, pos_buckets=pb)
        for b_, w_ev in enumerate(evs):
            jj, KK, offsets = metas[i0 + b_]
            for g, cls in w_ev:
                if g < offsets[-1]:
                    local_m = int(np.searchsorted(offsets[1:], g, side="right"))
                    events.append((jj + local_m, int(g - offsets[local_m]), cls))
        if on_progress:
            on_progress(min(i0 + batch_windows, len(wins)), len(wins))

    events.sort(key=lambda e: (e[0], e[1]))
    def _t(me, sl):
        slots_in_measure = int(measure_slots[me])
        return float(edges[me] + sl / slots_in_measure * (edges[me + 1] - edges[me])) \
            if me < len(edges) - 1 else float(edges[-1])

    hits, spans, hits_slots, spans_slots = [], [], [], []
    open_span = None
    last_key = None
    for me, sl, cls in events:
        t = _t(me, sl)
        if cls in HIT_CLASSES:
            # A hit is never a valid implicit span terminator.  Keeping the
            # authored hit and dropping the malformed, unclosed span avoids
            # inventing an end position that the decoder did not emit.
            if open_span is not None:
                open_span = None
            if (me, sl) == last_key:
                continue
            hits.append({"t": round(t, 4), "type": cls})
            hits_slots.append((me, sl, cls))
            last_key = (me, sl)
        elif cls in SPAN_CLASSES:
            if open_span is None:
                open_span = (me, sl, t, cls)
        elif cls == "end":
            if open_span is not None:
                if t - open_span[2] > 0.05:
                    # The explicit END slot is authoritative.  Derive both
                    # preview seconds and TJA coordinates from that same slot;
                    # duration caps must not move either representation.
                    spans.append({"t0": round(open_span[2], 4), "t1": round(t, 4),
                                  "type": open_span[3]})
                    spans_slots.append(
                        (open_span[0], open_span[1], me, sl, open_span[3]))
                open_span = None
    n_meas = (metas[-1][0] + metas[-1][1]) if metas else 0
    return {"hits": hits, "spans": spans, "hits_slots": hits_slots,
            "spans_slots": spans_slots, "n_measures": n_meas,
            "measure_slots": measure_slots[:n_meas].tolist(),
            "measure_num": measure_num[:n_meas].tolist(),
            "measure_den": measure_den[:n_meas].tolist(),
            "slots_per_4_4": SLOTS,
            "course": course, "level": level, "density_bucket": density_bucket}


# ---------------------------------------------------------------------------
# SoftChart 2.0 BarScript serving (SOFTCHART2_DESIGN §5, supplied-grid mode)
#
# Opt-in path: every v1.8 function above is untouched and remains the
# default. generate_song_barscript() teacher-forces the bar structure
# ([BAR] + BAR_DENOM per bar) from a trusted supplied grid, lets the model
# generate plan/skeleton/realization autoregressively under the BarscriptFSM
# legal mask at EVERY step (DENS [L, U] interval, SPAN_BUDGET consumption
# and the cross-bar open-span carry are hard constraints), samples span
# durations from the dedicated span_dur head under the §1.3 fail-closed
# clamp, and re-zips the token stream through barscript_to_events so the
# existing slot TJA writer exports the chart (plan/stage tokens never reach
# the chart).

SPAN_MAX = {"roll": 3.5, "roll_big": 4.0, "balloon": 6.5}  # §1.3 clamp (s)

# Precedence for the single ``spans[].clamp_reason`` string: reasons that
# determined where the span ENDED outrank the plan-stage duration-bucket mask.
CLAMP_PRIORITY = ("terminal_forced_close", "span_floor_carry",
                  "span_bucket_carry", "dur_bucket_floor")

# --- §1.3 fail-closed FLOORS, the symmetric counterparts of SPAN_MAX --------
#
# Every number below is the AUTHORED POPULATION MINIMUM measured on the frozen
# event contract (experiments/sc2_eval/visual_review/data/*/events, 36 songs,
# 4 courses, authored side deduplicated across arms).  They are corpus-empirical
# bounds -- "the corpus never goes below this" -- and deliberately NOT
# perceptual claims and NOT game judgement windows.
#
# SPAN_MIN_SEC / SPAN_MIN_BEATS: shortest closed authored span, 762 spans.
#   seconds  min 0.0833   p1 0.1769   p5 0.2727   p50 0.9636
#   beats    min 0.3333   p1 0.4167   p5 0.7500   p50 2.7500
# Both axes are load-bearing because each covers a tempo regime the other
# misses: at 300 BPM 1/3 beat is 67 ms (under the absolute floor), and at
# 60 BPM 83 ms is 0.083 beat (far under the rational-lattice floor).  The
# disjunction intercepts 0.00 % of authored spans and 8.9 % / 8.0 % / 4.0 % /
# 13.9 % of c1_best / c1_s4321 / b0_last / c11_on.
#
# NOTE (correction to the visual review's §2.2 proposal): the frozen spec's
# ``span_dur_edges[0]`` = 0.9167 beats is the top edge of duration bucket 0,
# NOT an authored lower bound -- 14.17 % of authored spans live inside bucket 0.
# Clamping to it would delete one authored span in seven.
SPAN_MIN_SEC = 0.0833
SPAN_MIN_BEATS = 1.0 / 3.0

# --- SHORT-span length profile (the open-bottomed duration bucket 0) --------
#
# MICROSPAN_CAUSE.md §6: ``span_dur_edges[0]`` is duration bucket 0's TOP
# edge, so bucket 0 is open at the bottom and its realized length collapses
# onto the first legal lattice position -- 80-92 % of every generated arm's
# bucket-0 spans sit there, against an authored population that is spread over
# the whole bucket.
#
# WHY THE FIX IS A CLOSING-STEP FLOOR AND NOT A DURATION-HEAD CONSTRAINT.  The
# BarScript grammar emits a bar as plan -> SKELETON -> realization, and the
# duration bucket is a realization token.  A span that opens and closes inside
# one bar therefore has its SPAN_END position fixed BEFORE the head names a
# bucket, so for that span the bucket describes the close rather than plans it
# -- and 80-92 % of every arm's bucket-0 spans (and 83-93 % of its sub-0.25 s
# spans) are exactly those same-bar spans.  Steering the close by the planned
# bucket can only reach the cross-bar remainder.  The lower bound is therefore
# drawn at the SPAN_START step, where the only span attributes decided are the
# course and the start position; the span CLASS is also a realization token,
# so it refines the draw only for spans that survive into a later bar.
#
# The distribution is corpus-derived, never hard-coded here: build it with
# scripts/build_span_bucket0_profile.py (train split, raw TJA) and pass the
# artifact to generate_song_barscript(span_bucket0_profile=...).  Absent it the
# serving path is bit-identical to the shipped one.
#
# HONEST LABEL: mechanically this is a per-span RANDOMIZED FLOOR, not a
# two-sided distribution match.  It only ever lengthens a span, so on a
# population whose short-span placement is already authored-like it biases
# upward; on the model's population, whose placement is degenerate at the
# lattice floor, the two nearly coincide.  What it buys over the fixed
# 0.75-beat floor MICROSPAN_CAUSE.md §8 priced is that the authored short tail
# survives in proportion instead of being deleted:
# scripts/build_span_bucket0_profile.py --report prints the authored-side cost.
class SpanBucket0Profile:
    """Authored within-bucket-0 span lengths (quarter-note beats) per cell.

    ``cells`` maps ``"{course}/{class}"`` to ``[[num, den, count], ...]``;
    the class ``"*"`` cell is the course marginal, used at the SPAN_START step
    where the realization has not yet named the class.  ``quantile`` is an
    inverse-CDF lookup, so one uniform drawn when the span opens keeps its
    meaning if the cell is refined later.
    """

    ANY_CLASS = "*"

    def __init__(self, cells, provenance=None):
        self.provenance = dict(provenance or {})
        self.cells = {}
        for key, values in cells.items():
            total = sum(int(n) for _, _, n in values)
            if total <= 0:
                raise ValueError(f"span bucket-0 profile cell {key} is empty")
            cum, acc = [], 0.0
            for num, den, n in values:
                if int(den) <= 0 or float(num) / float(den) <= 0.0:
                    raise ValueError(
                        f"span bucket-0 profile cell {key} has a "
                        "non-positive length")
                acc += int(n) / total
                cum.append((acc, float(num) / float(den)))
            self.cells[key] = tuple(cum)

    @classmethod
    def from_dict(cls, d):
        return cls({k: v["values"] for k, v in d["cells"].items()},
                   d.get("provenance"))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def quantile(self, course, span_class, u):
        """Inverse CDF of the (course, class) cell, in quarter-note beats.

        Falls back to the course marginal when the class is unknown or has no
        cell, and returns ``None`` when neither exists -- which leaves the
        shipped behaviour untouched for that span.
        """
        cum = self.cells.get(f"{course}/{span_class}") \
            or self.cells.get(f"{course}/{self.ANY_CLASS}")
        if not cum:
            return None
        for acc, beats in cum:
            if u <= acc:
                return beats
        return cum[-1][1]

    def target_beats(self, course, span_class, gen=None):
        """One draw from the (course, class) cell, in quarter-note beats."""
        return self.quantile(course, span_class,
                             float(torch.rand(1, generator=gen).item()))


# DEGENERATE_BAR_SEC: TJA authors insert sub-beat measures (#MEASURE 1/32 and
# friends) as scroll/BPM scaffolding.  Across 3 010 bars the authored side
# places ZERO HITS in any bar shorter than 0.375 s (28 such bars, 2 songs); the
# shortest authored hit-bearing bar is 0.375 s.  0.25 s is the largest round
# threshold with zero authored counterexamples AND a 1.5x margin to that bar,
# and it is the threshold the proposed ``degenerate_bar_empty`` gate uses, so
# fix and gate cannot disagree.
#
# Authors DO route span geometry through these bars: test_00004's 0.05 s bar 33
# carries a SPAN_START in all four courses and its bar 53 carries the matching
# SPAN_END.  The floor therefore blocks HITS only and leaves SPAN_START /
# SPAN_END / pass-through untouched.
DEGENERATE_BAR_SEC = 0.25

# MIN_ONSET_GAP_SEC -- a DEGENERACY guard, deliberately far below anything the
# domain writes.  It is NOT a style bound, and it must never be tuned to a corpus
# percentile.  Two revisions taught this:
#
#   0.0395 s  derived from a 43 387-pair SUBSET ("zero authored counterexamples").
#   0.0300 s  the 1 200-chart / 427 016-pair corpus minimum.
#
# Both were wrong, because our 1 000-song corpus is not the domain.  The series
# record for two adjacent notes is 29.4118 ms (TAIKO-TONGUE-TWISTER oni, BPM 170,
# 48th notes = 34 hits/s); arcade-only 29.76 ms; >=3 consecutive 33.48 ms;
# >=10 consecutive 34.71 ms.  Interval = 240000 / (BPM * denominator).  A 30 ms
# floor blocks the record itself, and every faster combination a future chart may
# use (BPM 175 /48 = 28.57 ms, BPM 200 /48 = 25.00 ms, BPM 170 /64 = 22.06 ms).
#
# The floor's only job is blocking output that is degenerate rather than fast --
# the visual review found six ka notes 16.7 ms apart rendering as a single blob.
# 0.020 s clears every documented chart with headroom and still blocks that.
#
# The model's real defect here is a RATE, not a value: sub-40 ms gaps are 0.045 %
# of authored pairs and were ~0.35 % of generated ones (~8x).  A mask cannot fix
# a distribution; that belongs to the gate and to training.
#
# NOTE (correction to the visual review's §2.4 mechanism): those bursts do NOT
# come from degenerate bars.  Zero of c1_best's 155 sub-40 ms gaps touch a bar
# under 0.25 s.  They come from ordinary 1.3-1.8 s bars whose authored
# BAR_DENOM is /48: the model fills adjacent /48 slots, which are 37 ms apart.
# DEGENERATE_BAR_SEC and MIN_ONSET_GAP_SEC address two independent defects.
MIN_ONSET_GAP_SEC = 0.0200


class SlotExportError(ValueError):
    """A generated bar-local position cannot be written on the 96-slot TJA
    export lattice.

    This is a RECORD-LEVEL failure, never a generation failure: the decoded
    chart (events, plans, counters, seconds preview) stays intact and only the
    slot export is unavailable, so a harness reports ``gen_tja = null`` instead
    of losing the whole record.  ``generate_song_barscript`` therefore records
    the condition in ``slot_export_error`` rather than raising; call
    ``require_slot_export`` at the export site to turn it into this exception.
    """


def require_slot_export(gen):
    """Raise :class:`SlotExportError` when ``gen`` has no valid slot export.

    The catchable surface a TJA writer should sit behind; returns ``gen``
    unchanged when the export lattice is clean.
    """
    err = gen.get("slot_export_error")
    if err:
        raise SlotExportError(err)
    return gen


# SoftChart 2.0 architecture flags a v2 checkpoint serializes in `args`
# alongside checkpoint.REQUIRED_GENERATOR_ARGS (which predates them; see
# load_model_v2), plus the split-axis knob-conditioning capability flag.
BARSCRIPT_MODEL_FLAGS = ("stage_emb", "span_duration_head", "tempo_head",
                         "rich_section_stats")
BARSCRIPT_KNOB_ARG = "axis_knobs"  # big_rate/span_rate/stream prefix tokens
V2_GENERATOR_ARGS = frozenset(BARSCRIPT_MODEL_FLAGS) | {BARSCRIPT_KNOB_ARG}

_PLAN_PHASES = ("plan0", "plan1", "plan2", "plan3")
_PLAN_FIELDS = ("dens", "span_budget", "sync", "motif")


def load_model_v2(ckpt_path, device="cuda"):
    """Load a checkpoint that may carry the SoftChart 2.0 model flags.

    checkpoint.py's REQUIRED_GENERATOR_ARGS predates the 2.0 flags, so this
    wrapper validates and applies them here: the four architecture flags
    (BARSCRIPT_MODEL_FLAGS) extend the model config with args/state presence
    cross-checks, and ``axis_knobs`` becomes the model's split-axis
    knob-conditioning capability. Absent v2 args mean False, so plain v1.8
    checkpoints load unchanged through the same strict path.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state, args, config, capabilities = parse_generator_checkpoint(checkpoint)
    v2 = {}
    for name in BARSCRIPT_MODEL_FLAGS + (BARSCRIPT_KNOB_ARG,):
        val = args.get(name, False)
        if not isinstance(val, bool):
            raise ValueError(f"checkpoint arg {name} must be boolean")
        v2[name] = val
    checks = (("stage_emb", "stage_emb"),
              ("span_duration_head", "span_dur."),
              ("tempo_head", "tempo_logp."),
              ("tempo_head", "tempo_k."),
              ("rich_section_stats", "section_encoder.inp_rich."))
    for name, prefix in checks:
        present = any(k == prefix or k.startswith(prefix) for k in state)
        if present != v2[name]:
            raise ValueError(
                f"checkpoint args/state disagree for {name}: "
                f"metadata={v2[name]}, state={present}")
    if v2["rich_section_stats"] and not config["hierarchical_ctx"]:
        raise ValueError("rich_section_stats requires hierarchical_ctx")
    config = dict(config, **{n: v2[n] for n in BARSCRIPT_MODEL_FLAGS})
    model = ChartModel(**config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    attach_generator_capabilities(model, capabilities)
    model._has_axis_knobs = v2[BARSCRIPT_KNOB_ARG]
    return model


def barscript_grid_from_bpm(bpm, offset, duration, meter=(4, 4), denom=16):
    """Uniform supplied grid from user BPM/offset (SOFTCHART2_DESIGN §5).

    ``offset`` is the audio time (seconds) of the first downbeat; bars tile
    forward until ``duration`` (audio length, seconds) is covered. ``denom``
    is the bar-local lattice denominator teacher-forced into every [BAR]
    header (16 = sixteenth-note grid in 4/4).
    """
    from .vocab import MAX_BAR_DENOM

    bpm = float(bpm)
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be positive and finite")
    num, den = int(meter[0]), int(meter[1])
    if num <= 0 or den <= 0:
        raise ValueError("meter terms must be positive")
    if (isinstance(denom, bool) or not isinstance(denom, (int, np.integer))
            or not 1 <= int(denom) <= MAX_BAR_DENOM):
        raise ValueError(f"denom must be an int in [1, {MAX_BAR_DENOM}]")
    offset, duration = float(offset), float(duration)
    if not np.isfinite(offset) or not np.isfinite(duration) \
            or duration <= offset:
        raise ValueError("duration must be finite and greater than offset")
    bar_sec = (240.0 / bpm) * num / den
    n = max(1, int(np.ceil((duration - offset) / bar_sec - 1e-9)))
    edges = offset + bar_sec * np.arange(n + 1, dtype=np.float64)
    return {"bpm": bpm, "downbeats": edges[:-1], "measure_edges": edges,
            "measure_num": [num] * n, "measure_den": [den] * n,
            "bar_denoms": [int(denom)] * n}


def barscript_grid_from_trace(trace, denoms=None):
    """Authored supplied grid: exact bar edges and meters from a parsed
    trace, bar denominators from the authored notes (the gold BAR_DENOM
    headers) unless overridden."""
    from .barscript import _bar_notes, bar_lattice_denom
    from .trace import bar_edge_times

    edges = np.asarray(bar_edge_times(trace), dtype=np.float64)
    if denoms is None:
        denoms = [bar_lattice_denom(notes) for notes in _bar_notes(trace)]
    return {"bpm": float(trace.bpm), "downbeats": edges[:-1],
            "measure_edges": edges,
            "measure_num": [b.num for b in trace.bars],
            "measure_den": [b.den for b in trace.bars],
            "bar_denoms": [int(d) for d in denoms]}


def _barscript_grid(grid):
    """Validate a BarScript supplied grid.

    Returns ``(edges, slots, num, den, quarter_beats, denoms)``; the first
    five reuse the exact slot-mode validators (§5: supplied-grid is an S2
    superset), ``denoms`` are the per-bar lattice denominators the [BAR]
    headers teacher-force.
    """
    from .vocab import MAX_BAR_DENOM

    db = np.asarray(grid["downbeats"], np.float64)
    edges = _grid_edges(grid, db)
    n = len(db)
    slots, num, den, quarter = resolve_meter_layout(grid, n)
    denoms = grid.get("bar_denoms")
    if denoms is None:
        raise ValueError(
            "BarScript generation requires bar_denoms: the per-bar lattice "
            "denominators teacher-forced in the [BAR] headers")
    denoms = np.asarray(denoms).reshape(-1)
    if len(denoms) != n:
        raise ValueError(f"bar_denoms must have one entry per bar ({n})")
    out = []
    for d in denoms.tolist():
        if isinstance(d, float):
            if not d.is_integer():
                raise ValueError("bar_denoms must be integers")
            d = int(d)
        if isinstance(d, bool) or not isinstance(d, int) \
                or not 1 <= d <= MAX_BAR_DENOM:
            raise ValueError(
                f"bar_denoms entries must be ints in [1, {MAX_BAR_DENOM}]")
        out.append(d)
    return edges, slots, num, den, quarter, out


def bar_denom_allowed(slots, mask="lattice"):
    """Denominators a bar may carry under ``mask`` (sorted ints).

    ``'lattice'`` keeps the divisors of the bar's OWN TJA export lattice --
    96 in 4/4, but 72 in 3/4, 84 in 7/8, 108 in 9/8 -- which is exactly the
    set whose bar-local positions ``write_tja_slots`` can express.  ``'div96'``
    is the literal DEPLOY_GRID_FIX §7 recommendation and coincides with
    ``'lattice'`` on 4/4 bars only.  ``None`` / ``'none'`` imposes nothing.

    Corpus check (396 763 authored bars, 4 657 charts of the barscript trace
    cache): 99.935 % of bars already satisfy ``'lattice'`` and only 97.434 %
    satisfy ``'div96'`` -- the 2.5 pp gap is authored bars in non-4/4 meters
    (13.6 % of the corpus), whose correct lattice ``'div96'`` would forbid.
    """
    from .vocab import MAX_BAR_DENOM

    if mask in (None, "none"):
        return list(range(1, MAX_BAR_DENOM + 1))
    if mask == "div96":
        from .vocab import SLOTS

        n = SLOTS
    elif mask == "lattice":
        n = int(slots)
    else:
        raise ValueError(f"unknown bar_denom_mask {mask!r}")
    return [d for d in range(1, min(n, MAX_BAR_DENOM) + 1) if n % d == 0]


def _knob_token(family, value, n, name):
    if value is None:
        return VOCAB.unk_cond
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) < n):
        raise ValueError(f"{name} bucket must be an int in [0, {n})")
    return family[int(value)]


def build_prefix_barscript(model, course, level=None, density_bucket=None,
                           style=None, sync_band=None, big_rate=None,
                           span_rate=None, stream=None):
    """[BOS] + base conditions (+ split-axis knobs) for the BarScript path.

    The §2.4 split-axis knobs sit AFTER the base condition block in the
    fixed order BIG_RATE, SPAN_RATE, STREAM (UNK when unspecified) and are
    emitted only for knob-capable models (``model._has_axis_knobs``). Knob
    values are raw bucket ints for now; the frozen spec's value->bucket
    mapping plugs in upstream later. The train-side prefix builder must
    match this ordering exactly.
    """
    from .vocab import N_BIG_RATE, N_SPAN_RATE, N_STREAM

    style_val = None
    if getattr(model, "_has_style", False):
        style_val = style if style is not None else -1
    sync_val = None
    if getattr(model, "_has_sync", False):
        sync_val = sync_band if sync_band is not None else -1
    seq = build_prefix(
        course, level, density_bucket,
        sib_pairs=([] if getattr(model, "_has_sib", False) else None),
        style=style_val, sync_band=sync_val)
    if getattr(model, "_has_axis_knobs", False):
        v = VOCAB
        seq += [_knob_token(v.big_rate, big_rate, N_BIG_RATE, "big_rate"),
                _knob_token(v.span_rate, span_rate, N_SPAN_RATE, "span_rate"),
                _knob_token(v.stream, stream, N_STREAM, "stream")]
    elif any(k is not None for k in (big_rate, span_rate, stream)):
        raise ValueError(
            "model lacks split-axis knob conditioning "
            "(big_rate/span_rate/stream)")
    return seq


def _normalize_supplied_plan(supplied_plan, n_bars):
    """Normalize the supplied-plan control surface (§5) to
    ``{global_bar: (dens|None, span_budget|None, sync|None, motif|None)}``.

    Accepts a mapping ``bar -> entry`` or a per-bar sequence (None = no
    override). An entry is a mapping over ``_PLAN_FIELDS`` or a sequence of
    up to 4 values in that order; None fields stay model-generated. Motif
    accepts a MOTIF_REF name or its index. Fail-closed validation.
    """
    from collections.abc import Mapping as _Mapping

    from .vocab import (MOTIF_REF_NAMES, N_BAR_SYNC, N_DENS_D8,
                        N_SPAN_BUDGET)

    if supplied_plan is None:
        return {}
    if isinstance(supplied_plan, _Mapping):
        items = dict(supplied_plan)
    else:
        items = {i: e for i, e in enumerate(supplied_plan) if e is not None}
    limits = (N_DENS_D8, N_SPAN_BUDGET, N_BAR_SYNC, None)
    out = {}
    for bar, entry in items.items():
        bar = int(bar)
        if not 0 <= bar < n_bars:
            raise ValueError(
                f"supplied plan bar {bar} outside 0..{n_bars - 1}")
        if isinstance(entry, _Mapping):
            unknown = set(entry) - set(_PLAN_FIELDS)
            if unknown:
                raise ValueError(
                    f"bar {bar}: unknown plan fields {sorted(unknown)}")
            vals = [entry.get(f) for f in _PLAN_FIELDS]
        else:
            vals = list(entry)
            if len(vals) > len(_PLAN_FIELDS):
                raise ValueError(f"bar {bar}: plan entry has more than "
                                 f"{len(_PLAN_FIELDS)} fields")
            vals += [None] * (len(_PLAN_FIELDS) - len(vals))
        norm = []
        for field, val, lim in zip(_PLAN_FIELDS, vals, limits):
            if val is None:
                norm.append(None)
            elif field == "motif":
                if (isinstance(val, (int, np.integer))
                        and not isinstance(val, bool)):
                    if not 0 <= int(val) < len(MOTIF_REF_NAMES):
                        raise ValueError(f"bar {bar}: motif index {val} "
                                         "out of range")
                    val = MOTIF_REF_NAMES[int(val)]
                if val not in MOTIF_REF_NAMES:
                    raise ValueError(f"bar {bar}: unknown motif {val!r}")
                norm.append(val)
            else:
                if (isinstance(val, bool)
                        or not isinstance(val, (int, np.integer))
                        or not 0 <= int(val) < lim):
                    raise ValueError(
                        f"bar {bar}: {field} must be an int in [0, {lim})")
                norm.append(int(val))
        out[bar] = tuple(norm)
    return out


def _stage_of_token(tok):
    """BarScript stage id of a generated token, exactly matching the gold
    assignment in barscript.trace_to_barscript."""
    from .barscript import _tok_family
    from .vocab import (STAGE_HEADER, STAGE_PLAN, STAGE_REALIZATION,
                        STAGE_SKELETON)

    fam = _tok_family(tok)
    if fam in ("bar", "bar_denom", "eos"):
        return STAGE_HEADER
    if fam in ("bar_dens", "span_budget", "bar_sync", "motif"):
        return STAGE_PLAN
    if fam in ("span_start", "span_end", "pos"):
        return STAGE_SKELETON
    return STAGE_REALIZATION  # [REAL] / type tokens / SPAN_DUR


def _sample_masked(logits, legal, temperature, top_p, greedy, gen):
    """One token id from a single-row logits vector under a boolean legal
    mask (grammar-constrained nucleus sampling)."""
    legal = legal.to(logits.device)
    if not bool(legal.any()):
        raise RuntimeError("empty legal mask (FSM invariant violated)")
    masked = logits.masked_fill(~legal, float("-inf"))
    if greedy:
        return int(masked.argmax().item())
    probs = torch.softmax(masked / max(float(temperature), 1e-6), dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = cum - sp < top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    pick = int(torch.multinomial(sp.cpu(), 1, generator=gen).item())
    return int(si[pick].item())


def _span_bucket_bounds(spec, bucket):
    """[lo, hi) beat bounds of a span-duration bucket (the top bucket
    extends the log grid by one step)."""
    e = spec.span_dur_edges
    lo = 0.0 if bucket == 0 else float(e[bucket - 1])
    hi = (float(e[bucket]) if bucket < len(e)
          else float(e[-1]) * float(e[-1]) / float(e[-2]))
    return lo, hi


def span_floor_sec(sec_per_beat, floor_sec=SPAN_MIN_SEC,
                   floor_beats=SPAN_MIN_BEATS):
    """Shortest span this bar's tempo admits, in seconds.

    The two authored minima are combined by ``max``: a span must clear the
    absolute floor AND the rational-lattice floor (see SPAN_MIN_SEC).
    """
    return max(float(floor_sec), float(floor_beats) * float(sec_per_beat))


def _legal_dur_buckets(spec, cls, sec_per_beat, remaining_sec=None,
                       min_sec=None):
    """§1.3 fail-closed clamp on the span-duration head: buckets whose
    lower bound (seconds at this bar's tempo) exceeds the class SPAN_MAX are
    masked; bucket 0 always stays legal.

    ``remaining_sec`` applies the same mask against the chart's own remaining
    length (seconds from this span's start to the last position at which a
    SPAN_END can still be placed).  Without it the head is free to plan a
    duration the grid cannot hold, which is what produced the terminal
    ``orphan_final_span`` spans: long buckets committed in the last bars.

    ``min_sec`` is the symmetric FLOOR: a bucket whose UPPER bound is already
    under the authored minimum span length cannot describe a legal span, so it
    is masked.  Bucket 0's ceiling is ``span_dur_edges[0]`` beats, so under the
    frozen grid the floor only reaches bucket 0 above ~660 BPM -- the floor's
    real work is done by the closing-position mask, and this arm exists so the
    duration head and the realized close cannot disagree about what is legal.
    The mask never empties: if the ceiling clamp and the floor clamp exclude
    each other, the ceiling wins (fail-closed on playability limits) and
    bucket 0 stays legal.
    """
    from .vocab import N_SPAN_DUR_BUCKETS

    limit = SPAN_MAX.get(cls, max(SPAN_MAX.values()))
    if remaining_sec is not None:
        limit = min(limit, max(float(remaining_sec), 0.0))
    legal = np.array(
        [_span_bucket_bounds(spec, b)[0] * float(sec_per_beat) <= limit
         for b in range(N_SPAN_DUR_BUCKETS)], dtype=bool)
    legal[0] = True
    if min_sec is not None:
        floored = legal & np.array(
            [_span_bucket_bounds(spec, b)[1] * float(sec_per_beat)
             > float(min_sec) - 1e-9 for b in range(N_SPAN_DUR_BUCKETS)],
            dtype=bool)
        if floored.any():  # never hand back an empty mask
            legal = floored
    return legal


HAND_OF = {"don": "don", "don_big": "don", "ka": "ka", "ka_big": "ka"}

# Two courses of one song do NOT share a bar lattice: TJA ``#MEASURE`` is
# per-course, so (bar, position) keys are not comparable across the ladder and
# the family reference has to live in SECONDS.  The tolerance is
# gates/family.py's MATCH_TOL_SEC, deliberately the same number the ladder gate
# scores with (and the same evaluate.match_onsets default the matched-eval path
# uses), so the decode-time constraint and the gate that measures it cannot
# disagree about what "the same musical instant" means.  Its defensibility is
# Rasch (1979), Acustica 43:121-131 -- performed asynchronies of 30-50 ms are
# still heard as simultaneous.  It is not a game judgement window.
FAMILY_MATCH_TOL_SEC = 0.050


class FamilyRef:
    """Onsets of an already generated HARDER sibling, in audio seconds.

    ``times`` is sorted; ``classes[i]`` is the note/span class at ``times[i]``.
    Span ENDS are excluded: they are not onsets, and the ladder convention is
    about where onsets fall.
    """

    __slots__ = ("times", "classes", "tol")

    def __init__(self, times, classes, tol=FAMILY_MATCH_TOL_SEC):
        order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
        self.times = np.asarray(times, dtype=np.float64)[order]
        self.classes = [classes[i] for i in order]
        self.tol = float(tol)

    def __len__(self):
        return int(self.times.size)

    def match(self, t):
        """Class of the parent onset within ``tol`` of ``t``, else None."""
        if not self.times.size:
            return None
        i = int(np.searchsorted(self.times, t))
        best, bd = None, self.tol + 1e-12
        for j in (i - 1, i):
            if 0 <= j < self.times.size:
                d = abs(float(self.times[j]) - t)
                if d < bd:
                    bd, best = d, self.classes[j]
        return best

    def covers(self, t):
        return self.match(t) is not None


def family_ref_from_generation(gen, tol=FAMILY_MATCH_TOL_SEC):
    """Build a :class:`FamilyRef` from a generate_song_barscript result.

    Reads the seconds preview the decoder already produced (``hits`` and span
    ``t0``), so it works whether or not the sibling shared this course's grid.
    """
    times = [float(h["t"]) for h in gen["hits"]]
    classes = [h["type"] for h in gen["hits"]]
    times += [float(s["t0"]) for s in gen["spans"]]
    classes += [s["type"] for s in gen["spans"]]
    return FamilyRef(times, classes, tol=tol)


def _apply_family_bias(row, ph, fsm, cur, legal, family_ref, family_bias,
                       family_hand_bias, counters):
    """Nested-difficulty-family steering: an additive logit bias toward the
    already generated HARDER sibling's onset positions and drum choices.

    A SOFT bias, not a subset constraint, because the authored convention is
    itself soft: on the 13-song ladder population the easier chart's onsets sit
    on an exact rational position of the harder chart 90.5-96.7 % of the time
    (per-song min 0.763, p10 0.887), so a strict subset would over-constrain
    at least one song in ten.  Charters do add notes an easier course does not
    inherit.

    Two separate strengths, because the two conventions have different
    tightness.  ``family_bias`` steers POSITION (authored 0.905-0.967).
    ``family_hand_bias`` steers only the don/ka HAND at an inherited position
    (authored 0.746-0.825) and deliberately leaves the big/small axis free:
    big-note share is a difficulty-dependent axis in its own right (authored
    easy 18.8 % vs oni 4.9 %), so forcing the parent's exact class would fight
    the difficulty ladder it is meant to support.
    """
    if (family_ref is None or not len(family_ref) or cur is None
            or not (family_bias or family_hand_bias)):
        return row
    v = VOCAB

    def _t(i):
        return cur[2] + i / cur[1] * (cur[3] - cur[2])

    if ph == "skel" or (ph == "skel_pos" and fsm.pending == "span_start"):
        if not family_bias:
            return row
        block = slice(v.time0, v.time0 + cur[1])
        keep = torch.from_numpy(np.asarray(legal[block], dtype=bool))
        if not bool(keep.any()):
            return row
        row = row.clone()
        signs = torch.tensor(
            [1.0 if family_ref.covers(_t(i)) else -1.0
             for i in range(cur[1])], device=row.device)
        sub = row[block]
        # Renormalize so the position block's log-sum-exp is unchanged over
        # the LEGAL positions.  Without this the bias also raises the total
        # mass on "place another onset" against [REAL]/[SPAN_START] at the
        # same step, i.e. a POSITION preference would silently become a
        # DENSITY preference and the easier course would gain notes.  The
        # family constraint must move onsets, never add them.
        before = torch.logsumexp(sub[keep], dim=0)
        biased = sub + float(family_bias) * signs
        after = torch.logsumexp(biased[keep], dim=0)
        row[block] = biased - (after - before)
        counters["family_bias_steps"] += 1
        return row
    if ph == "real" and family_hand_bias and fsm.real_i < len(fsm.skeleton):
        pos_i, role = fsm.skeleton[fsm.real_i]
        ref_cls = family_ref.match(_t(pos_i))
        if ref_cls is None:
            return row
        row = row.clone()
        if role == "hit" and ref_cls in HAND_OF:
            for cls in HIT_CLASSES:
                if HAND_OF[cls] == HAND_OF[ref_cls]:
                    row[v.note[cls]] += float(family_hand_bias)
            counters["family_hand_bias_steps"] += 1
        elif role == "span_start" and ref_cls in SPAN_CLASSES:
            row[v.note[ref_cls]] += float(family_hand_bias)
            counters["family_hand_bias_steps"] += 1
    return row


def _decode_barscript_window(model, memory, prefix, bars_meta, spec, carry,
                             plan_overrides, device, greedy, temperature,
                             plan_temperature, top_p, gen, counters,
                             span_buckets, last_close_t=None, final_bar=None,
                             clamp_reasons=None, span_min_sec=SPAN_MIN_SEC,
                             span_min_beats=SPAN_MIN_BEATS,
                             degenerate_bar_sec=DEGENERATE_BAR_SEC,
                             min_onset_gap_sec=MIN_ONSET_GAP_SEC,
                             prev_onset_t=None, family_ref=None,
                             family_bias=0.0, family_hand_bias=0.0,
                             family_mode="bias", course=None,
                             span_bucket0_profile=None,
                             denom_masks=None):
    """One window of grammar-masked AR decoding.

    ``bars_meta`` rows are ``[global_bar, denom, t0, t1, sec_per_beat]``.
    ``denom_masks`` switches the header's BAR_DENOM step from teacher forcing
    to sampling: ``None`` (default) forces the supplied ``denom``, otherwise it
    maps a global bar index to a bool mask over the vocabulary and the model
    picks its own denominator inside it, which is then written back into that
    bar's ``bars_meta`` row (so every downstream guard -- position mask, DENS
    feasibility, min-IOI, family bias, export -- reads the COMMITTED value).
    ``carry`` is the open-span state from the previous window (or None).
    ``last_close_t`` is the audio time of the last position in the whole song
    at which a SPAN_END can still be placed and ``final_bar`` the global index
    of the song's last bar; together they drive the terminal-boundary
    treatment (prevention at SPAN_START, forced closure in the final bar).
    ``prev_onset_t`` is the audio time of the last onset placed anywhere before
    this window, threading the min-IOI floor across the window boundary.
    Returns ``(tokens after the prefix, eos_emitted, carry_out, plans,
    prev_onset_t)``.
    """
    from .barscript import BarscriptFSM
    from .vocab import N_SPAN_DUR_BUCKETS, STAGE_HEADER

    v = VOCAB
    fsm = BarscriptFSM(spec)
    open_cls = open_t0 = open_key = None
    # the open span's short-length quantile, drawn once at its SPAN_START;
    # None means the profile is off or has no cell for this span
    open_dur_u = None
    if carry is not None:
        fsm.open_span = True  # cross-window continuation of the carry state
        open_cls, open_t0 = carry["cls"], carry["t0"]
        open_key = carry.get("key")
        open_dur_u = carry.get("dur_u")
    last_span_cls = open_cls
    if clamp_reasons is None:
        clamp_reasons = []
    # Shortest span the frozen duration grid distinguishes from the degenerate
    # bucket 0 (spec.span_dur_edges[0] quarter-note beats): a SPAN_START with
    # less than this much chart left cannot become a legal span, so it is
    # masked instead of being opened and later orphaned.
    min_span_beats = float(spec.span_dur_edges[0])
    real_start_ts = []  # FIFO of span-start times awaiting their SPAN_DUR step
    if family_mode not in ("bias", "subset"):
        raise ValueError("family_mode must be 'bias' or 'subset'")
    family_ref = family_ref or {}

    def _pos_time(meta, idx):
        return meta[2] + idx / meta[1] * (meta[3] - meta[2])

    def _span_floor(meta):
        """Shortest legal span length in seconds at this bar's tempo."""
        if span_min_sec is None and span_min_beats is None:
            return 0.0
        return span_floor_sec(meta[4], span_min_sec or 0.0,
                              span_min_beats or 0.0)

    def _bucket_floor(meta, elapsed):
        """LOWEST-priority floor: the authored short-span length this span
        drew when it opened, in seconds at this bar's tempo.

        ``elapsed`` are the candidate closing offsets from the span start
        (seconds).  Returns 0.0 -- i.e. stands down -- unless the span can
        still close INSIDE duration bucket 0, whose top edge is the profile's
        whole domain.  A span the model has already taken past that edge is
        no longer a short span and is left exactly as the shipped path would
        leave it, so the floor can never pull a long span back down.
        """
        if open_dur_u is None or span_bucket0_profile is None:
            return 0.0
        spb = float(meta[4])
        hi = float(spec.span_dur_edges[0]) * spb
        if not np.any(np.asarray(elapsed, dtype=np.float64) < hi - 1e-9):
            return 0.0
        beats = span_bucket0_profile.quantile(course, open_cls, open_dur_u)
        return 0.0 if not beats else float(beats) * spb

    def _gap_ok(meta, idx):
        """Does bar-local ``idx`` clear the min-IOI floor against the last
        onset placed anywhere in the chart so far?"""
        if min_onset_gap_sec is None or prev_onset_t is None:
            return True
        return (_pos_time(meta, idx) - prev_onset_t
                >= float(min_onset_gap_sec) - 1e-9)


    def _span_start_fits(meta, idx):
        """Can a span opened at bar-local ``idx`` still be closed?

        True iff the song's last legal closing position sits at least one
        span-duration grid step after it (tempo of the opening bar).  At the
        song's very last position this is 0 s and the answer is False, which
        also covers the structural case "no closing position exists at all".
        """
        if last_close_t is None:
            return True
        return (last_close_t - _pos_time(meta, idx)
                >= min_span_beats * meta[4] - 1e-9)
    seq = list(prefix)
    # Prefix tokens ride the header stage (interface decision; zero-init
    # stage_emb makes this a free choice at init — train side must match).
    stages = [STAGE_HEADER] * len(prefix)
    use_stage = getattr(model, "stage_emb", None) is not None
    seq_t = torch.tensor([seq], dtype=torch.long, device=device)
    stage_t = (torch.tensor([stages], dtype=torch.long, device=device)
               if use_stage else None)
    dens_rev = {t: d for d, t in v.bar_dens.items()}
    budget_rev = {t: s for s, t in v.span_budget.items()}
    sync_rev = {t: s for s, t in v.bar_sync.items()}
    motif_rev = {t: m for m, t in v.motif.items()}
    plan_fams = (v.bar_dens, v.span_budget, v.bar_sync, v.motif)
    plans = []
    eos = False
    while True:
        if len(seq) >= MAX_TGT:
            raise ValueError(
                f"BarScript window exceeded MAX_TGT={MAX_TGT} tokens at bar "
                f"{bars_meta[max(fsm.bar_i, 0)][0]}; lower the DENS plan or "
                "the supplied bar_denoms")
        ph = fsm.phase
        cur = bars_meta[fsm.bar_i] if 0 <= fsm.bar_i < len(bars_meta) else None
        forced = None
        if ph == "start":
            if fsm.bar_i + 1 < len(bars_meta):
                forced = v.bar  # supplied-grid header teacher-force
            elif fsm.open_span:
                break  # open span carries into the next window; no EOS here
            else:
                forced = v.eos
        elif ph == "denom":
            if denom_masks is None:
                forced = v.bar_denom[cur[1]]
        elif ph in _PLAN_PHASES:  # supplied-plan control surface
            ov = plan_overrides.get(cur[0])
            if ov is not None:
                val = ov[_PLAN_PHASES.index(ph)]
                if val is not None:
                    if ph == "plan0" and spec.dens_bounds(val)[0] > cur[1]:
                        raise ValueError(
                            f"bar {cur[0]}: supplied DENS bucket {val} needs "
                            f">= {spec.dens_bounds(val)[0]} onsets on a "
                            f"/{cur[1]} lattice (infeasible)")
                    forced = plan_fams[_PLAN_PHASES.index(ph)][val]
        legal = fsm.legal_mask()
        if forced is not None:
            if not legal[forced]:
                raise ValueError(
                    f"supplied token illegal for bar "
                    f"{cur[0] if cur else '?'} at [{fsm.state_summary()}]")
            tok = forced
        else:
            legal = legal.copy()
            if ph == "denom" and denom_masks is not None:
                # model-emitted lattice, restricted to the denominators this
                # bar's meter can export (fail-open + counted, never empty)
                allowed = denom_masks[cur[0]]
                if (legal & allowed).any():
                    legal &= allowed
                else:
                    counters["bar_denom_mask_fail_open"] += 1
            # Serving-side feasibility guards. The FSM's skel mask offers
            # SPAN_START whenever room and budget remain, even when no legal
            # start position exists (its escape hatch misses that corner and
            # the next step would dead-end); keep sampled paths out of that
            # state and out of DENS plans that cannot fit the bar lattice.
            if ph == "plan0":
                for db_, dt_ in v.bar_dens.items():
                    if legal[dt_] and spec.dens_bounds(db_)[0] > cur[1]:
                        legal[dt_] = False
            # The FSM's own position offer, captured BEFORE the floors narrow
            # it.  SPAN_START's landing position is chosen one step later at
            # "skel_pos" from this same set, so the feasibility guards below
            # must ask the FSM what it offers, not what the hit-side floors
            # have already removed.
            base_pos = (np.flatnonzero(legal[v.time0:v.time0 + cur[1]])
                        if cur is not None else np.array([], dtype=np.int64))
            degenerate = (cur is not None and degenerate_bar_sec is not None
                          and cur[3] - cur[2] < float(degenerate_bar_sec))
            # Serving-side masks may only ever NARROW the FSM's mask: fsm.step
            # re-validates against its own mask, so re-legalizing a token the
            # FSM refused (e.g. forcing [REAL] open) raises IllegalTokenError.
            # Every arm below therefore either subsets the FSM's offer or
            # steers the PLAN so the FSM offers the right thing by itself.
            #
            # (i) DEGENERATE BARS.  Across 3 010 bars the authored side places
            #     zero HITS in any bar under 0.375 s, but test_00004's 0.05 s
            #     bar 33 carries a SPAN_START in all four courses and bar 53
            #     the matching SPAN_END.  So the rule is "no hits", not "no
            #     content", and it is imposed at the PLAN stage where the FSM
            #     will then derive a consistent skeleton mask:
            #       plan0 -> DENS capped at one onset (bucket upper bound <= 1)
            #       plan1 -> that one onset must be affordable as a SPAN_START
            #       plan3 -> MOTIF pinned to "new" (a scaffolding bar repeats
            #                nothing, and a REPEAT target would force a hit)
            #       skel  -> the surviving plain (= hit) positions are removed;
            #                SPAN_START is guaranteed legal in exchange
            if degenerate:
                if ph == "plan0":
                    # at most one onset may be REQUIRED, and the hit block
                    # below leaves SPAN_START as the only way to supply it
                    for db_, dt_ in v.bar_dens.items():
                        if legal[dt_] and spec.dens_bounds(db_)[0] > 1:
                            legal[dt_] = False
                    counters["degenerate_bar_dens_capped"] += 1
                elif ph == "plan1" and fsm.dens_lo >= 1:
                    # the single planned onset can only be a span start, so
                    # the budget must be able to pay for it
                    for sb_, st_ in v.span_budget.items():
                        if legal[st_] and sb_ < 1:
                            legal[st_] = False
                elif ph == "plan3":
                    for mn_, mt_ in v.motif.items():
                        if mn_ != "new":
                            legal[mt_] = False
                elif ph == "skel":
                    plain = slice(v.time0, v.time0 + cur[1])
                    if legal[plain].any():
                        if legal[v.span_start] or legal[v.real]:
                            legal[plain] = False
                            counters["degenerate_bar_hits_blocked"] += 1
                        else:
                            # a SUPPLIED plan can force a DENS the cap above
                            # would have refused; the explicit override wins
                            # and the deviation is recorded, never a crash
                            counters["degenerate_bar_fail_open"] += 1
            # (ii) MIN-IOI FLOOR.  Every onset, hit ("skel") or span start
            #      ("skel_pos"), must clear the authored minimum gap.  Pure
            #      narrowing: when no position survives, the whole filter is
            #      abandoned rather than widening the mask, so a DENS plan
            #      denser than the floor allows still gets its onsets (counted
            #      as min_gap_fail_open) instead of dead-ending.
            if (min_onset_gap_sec is not None and cur is not None
                    and (ph == "skel"
                         or (ph == "skel_pos"
                             and fsm.pending == "span_start"))):
                plain = slice(v.time0, v.time0 + cur[1])
                idxs = np.flatnonzero(legal[plain])
                keep = np.array([i for i in idxs if _gap_ok(cur, int(i))],
                                dtype=np.int64)
                if len(keep) and len(keep) < len(idxs):
                    counters["min_gap_blocked"] += int(len(idxs) - len(keep))
                    legal[plain] = False
                    legal[v.time0 + keep] = True
                elif not len(keep) and len(idxs):
                    counters["min_gap_fail_open"] += 1
            # (iii) NESTED FAMILY, hard variant.  ``family_mode='subset'``
            #      restricts onsets to the harder sibling's positions.  Kept
            #      for ablation only: the authored convention is soft (see
            #      _apply_family_bias), and this arm fails open -- counted --
            #      whenever the restriction would leave no position, so it is
            #      never a strict subset in practice either.
            if (family_mode == "subset" and family_ref is not None
                    and len(family_ref) and cur is not None
                    and (ph == "skel"
                         or (ph == "skel_pos"
                             and fsm.pending == "span_start"))):
                plain = slice(v.time0, v.time0 + cur[1])
                idxs = np.flatnonzero(legal[plain])
                keep = np.array(
                    [i for i in idxs
                     if family_ref.covers(_pos_time(cur, int(i)))],
                    dtype=np.int64)
                if len(keep):
                    if len(keep) < len(idxs):
                        legal[plain] = False
                        legal[v.time0 + keep] = True
                        counters["family_subset_restricted"] += 1
                elif len(idxs):
                    counters["family_subset_failed_open"] += 1
            if ph == "skel" and legal[v.span_start]:
                # Two SPAN_START feasibility conditions, both read off the
                # FSM's own position mask (no FSM internals):
                #  (i) a legal landing position must exist at all — the skel
                #      mask offers SPAN_START on room+budget alone and its
                #      escape hatch misses that corner, so the next step would
                #      dead-end;
                #  (ii) the chart must still be long enough to close the span
                #      (terminal boundary; see _span_start_fits).
                idxs = base_pos
                if not len(idxs):
                    legal[v.span_start] = False
                elif not _span_start_fits(cur, int(idxs[0])):
                    legal[v.span_start] = False
                    counters["terminal_span_blocked"] += 1
            if ph == "skel_pos" and fsm.pending == "span_start":
                # keep the opening position itself inside the closable region
                idxs = np.flatnonzero(legal[v.time0:v.time0 + cur[1]])
                ok = [int(i) for i in idxs if _span_start_fits(cur, int(i))]
                if ok and len(ok) < len(idxs):
                    only = np.zeros_like(legal)
                    only[v.time0 + np.asarray(ok, dtype=np.int64)] = True
                    legal = only
                    counters["terminal_span_blocked"] += 1
            elif ph == "skel_pos" and fsm.pending == "span_end":
                # closing-position steering: (i) §1.3 fail-closed clamp on
                # the realized length, (ii) keep the DENS lower bound
                # reachable after the close; when both fail, close at the
                # bar's last slot so the escape hatch stays available
                # (over-clamp survivors are counted at assembly).
                # (iii) the §1.3 FLOOR, symmetric to the ceiling: a close that
                # would realize a span shorter than the authored minimum
                # (span_floor_sec) is steered away from.  The floor sits
                # BELOW the ceiling in the preference ladder, so a span that
                # can satisfy neither still closes inside SPAN_MAX -- the
                # ceiling is a playability limit, the floor a style floor.
                # (iv) the opt-in SHORT-SPAN floor (SpanBucket0Profile) sits
                # below all of those: it is interleaved rung by rung, so it is
                # the FIRST constraint dropped whenever it conflicts with any
                # of them, and while it is off the ladder is untouched.
                idxs = np.flatnonzero(legal[v.time0:v.time0 + cur[1]])
                tt = cur[2] + idxs / cur[1] * (cur[3] - cur[2])
                bf = idxs
                if open_t0 is not None:
                    limit = SPAN_MAX.get(last_span_cls,
                                         max(SPAN_MAX.values()))
                    cl = idxs[tt <= open_t0 + limit + 1e-9]
                    floor = _span_floor(cur)
                    fl = idxs[tt >= open_t0 + floor - 1e-9]
                    bfloor = _bucket_floor(cur, tt - open_t0)
                    if bfloor > floor:
                        bf = idxs[tt >= open_t0 + bfloor - 1e-9]
                else:
                    cl = fl = idxs
                need = max(fsm.dens_lo - fsm.n_count, 0)
                fe = idxs[idxs <= cur[1] - 1 - need] if need else idxs
                cand = idxs
                floored = bucketed = False
                fe_cl = np.intersect1d(fe, cl)
                base = ((np.intersect1d(fe_cl, fl), True),
                        (np.intersect1d(fe, fl), True),
                        (np.intersect1d(cl, fl), True), (fl, True),
                        (fe_cl, False), (fe, False),
                        (idxs[idxs == cur[1] - 1], False),
                        (cl, False), (idxs, False))
                ladder = []
                for choice, from_floor in base:
                    if len(bf) < len(idxs):
                        ladder.append((np.intersect1d(choice, bf),
                                       from_floor, True))
                    ladder.append((choice, from_floor, False))
                for choice, from_floor, from_bucket in ladder:
                    if len(choice):
                        cand, floored, bucketed = choice, from_floor, from_bucket
                        break
                # counted only when the floor BOTH removed candidates and the
                # surviving choice came from the floor-filtered set
                if floored and len(fl) < len(idxs):
                    counters["span_close_floor"] += 1
                if bucketed:
                    counters["span_bucket_floor"] += 1
                only = np.zeros_like(legal)
                only[v.time0 + cand] = True
                legal = only
            elif ph == "skel" and fsm.open_span and cur is not None:
                # Two reasons to take the pass-through option ([REAL]) away
                # from an open span: (i) §1.3 live clamp — the span has
                # reached its class limit; (ii) terminal boundary — this is
                # the song's last bar, so passing through would strand the
                # span with no bar left to close it (the orphan_final_span
                # mode).  Either way SPAN_END becomes the only legal token and
                # the model still chooses WHERE to close, via the
                # closing-position steering above.
                limit = SPAN_MAX.get(last_span_cls, max(SPAN_MAX.values()))
                over = (open_t0 is not None and cur[2] - open_t0 >= limit)
                terminal = final_bar is not None and cur[0] >= final_bar
                if (over or terminal) and legal[v.span_end]:
                    # pass-through was genuinely on offer only when [REAL] was
                    # legal; otherwise the FSM had already forced the closure
                    took_passthrough = bool(legal[v.real])
                    only = np.zeros_like(legal)
                    only[v.span_end] = True
                    legal = only
                    counters["forced_close"] += 1
                    if terminal and not over and took_passthrough:
                        counters["terminal_forced_close"] += 1
                        if open_key is not None:
                            clamp_reasons.append({
                                "reason": "terminal_forced_close",
                                "bar": int(open_key[0]),
                                "pos_num": int(open_key[1].numerator),
                                "pos_den": int(open_key[1].denominator),
                                "class": open_cls,
                                "detail": (
                                    "span still open in the song's last bar; "
                                    "SPAN_END forced at the last legal "
                                    "closing position"),
                            })
                elif (legal[v.span_end] and legal[v.real]
                      and open_t0 is not None):
                    # §1.3 FLOOR, pass-through side: if no position in this
                    # bar is far enough from the start to make a legal span,
                    # closing here can only produce a micro-span, so the span
                    # carries into the next bar instead.  Symmetric to the
                    # forced-close arm above and strictly weaker: it only
                    # fires when pass-through was already legal, and never
                    # when the ceiling or the terminal boundary applies.  The
                    # opt-in short-span floor rides the same arm (counted
                    # apart as span_bucket_carry) so a span that opens too
                    # late in a bar to reach its draw carries instead of
                    # closing under it.
                    # the closing positions skel_pos would offer one step later
                    end_idxs = np.arange(fsm.last_pos + 1, cur[1])
                    tt = cur[2] + end_idxs / cur[1] * (cur[3] - cur[2])
                    fl_sec = _span_floor(cur)
                    bf_sec = _bucket_floor(cur, tt - open_t0)
                    eff = max(fl_sec, bf_sec)
                    if not len(end_idxs) or not (
                            tt >= open_t0 + eff - 1e-9).any():
                        legal[v.span_end] = False
                        bucket_carry = eff > fl_sec and (
                            len(end_idxs) and (tt >= open_t0 + fl_sec
                                               - 1e-9).any())
                        if bucket_carry:
                            counters["span_bucket_carry"] += 1
                        else:
                            counters["span_floor_carry"] += 1
                        if open_key is not None:
                            clamp_reasons.append({
                                "reason": ("span_bucket_carry" if bucket_carry
                                           else "span_floor_carry"),
                                "bar": int(open_key[0]),
                                "pos_num": int(open_key[1].numerator),
                                "pos_den": int(open_key[1].denominator),
                                "class": open_cls,
                                "detail": (
                                    "no closing position in bar "
                                    f"{cur[0]} clears the "
                                    + ("span's drawn short-span bound"
                                       if bucket_carry else
                                       "authored span floor")
                                    + f" ({eff:.4f} s); SPAN_END withheld and "
                                    "the span carried to the next bar"),
                            })
            if not legal.any():
                raise ValueError(
                    f"bar {cur[0] if cur else '?'}: supplied plan is "
                    f"infeasible on the /{cur[1] if cur else '?'} lattice "
                    f"[{fsm.state_summary()}]")
            with _autocast(device):
                logits_all, hidden = model.decode(
                    seq_t, memory, stage_ids=stage_t, return_hidden=True)
            row = logits_all[0, -1].float()
            if ph == "real_dur":
                # span duration: dedicated head at the SPAN_START step
                # (§1.3), falling back to the LM head when the flag is off
                # the same clamp against the chart's REMAINING length, so a
                # span opened near the end cannot commit to a bucket the grid
                # can no longer hold (report §1.4 recommendation)
                t0_span = (real_start_ts.pop(0) if real_start_ts else open_t0)
                rem = (None if last_close_t is None or t0_span is None
                       else last_close_t - t0_span)
                free = _legal_dur_buckets(spec, last_span_cls, cur[4],
                                          remaining_sec=rem)
                dur_legal = _legal_dur_buckets(spec, last_span_cls, cur[4],
                                               remaining_sec=rem,
                                               min_sec=_span_floor(cur))
                if not np.array_equal(free, dur_legal):
                    counters["dur_bucket_floor"] += 1
                    if open_key is not None:
                        clamp_reasons.append({
                            "reason": "dur_bucket_floor",
                            "bar": int(open_key[0]),
                            "pos_num": int(open_key[1].numerator),
                            "pos_den": int(open_key[1].denominator),
                            "class": last_span_cls,
                            "detail": (
                                "span-duration buckets whose ceiling is under "
                                f"the authored floor ({_span_floor(cur):.4f} "
                                "s at this bar's tempo) masked"),
                        })
                if getattr(model, "span_dur", None) is not None:
                    blogits = model.span_dur(hidden[0, -1].float())
                else:
                    ids = [v.span_dur[b] for b in range(N_SPAN_DUR_BUCKETS)]
                    blogits = row[ids]
                b = _sample_masked(blogits, torch.from_numpy(dur_legal),
                                   temperature, top_p, greedy, gen)
                span_buckets.append({"bar": cur[0], "bucket": int(b),
                                     "cls": last_span_cls})
                tok = v.span_dur[b]
            else:
                temp = (plan_temperature if ph in _PLAN_PHASES
                        else temperature)  # plan-step diversity knob
                row = _apply_family_bias(
                    row, ph, fsm, cur, legal, family_ref, family_bias,
                    family_hand_bias, counters)
                tok = _sample_masked(
                    row, torch.from_numpy(legal).to(row.device),
                    temp, top_p, greedy, gen)
            if tok == v.real and ph == "skel" and fsm.n_count < fsm.dens_lo:
                counters["escape_hatch"] += 1
        # span timing/class bookkeeping (pre-step FSM state)
        closing = ph == "skel_pos" and fsm.pending == "span_end"
        # min-IOI bookkeeping: a plain "skel" position and a SPAN_START
        # position are both onsets; a SPAN_END position is not.
        if cur is not None and v.time0 <= tok < v.time0 + cur[1] and not (
                ph == "skel_pos" and fsm.pending == "span_end"):
            prev_onset_t = _pos_time(cur, int(tok - v.time0))
        if ph == "skel_pos" and fsm.pending == "span_start":
            open_t0 = cur[2] + (tok - v.time0) / cur[1] * (cur[3] - cur[2])
            open_key = (cur[0], Fraction(int(tok - v.time0), int(cur[1])))
            real_start_ts.append(open_t0)
            open_cls = last_span_cls = None
            # The short-length quantile is drawn HERE, at the span's start:
            # the SPAN_END position of a same-bar span is chosen in this same
            # skeleton, before the realization names a duration bucket or a
            # class, so nothing later in the bar can steer it.
            open_dur_u = (None if span_bucket0_profile is None else
                          float(torch.rand(1, generator=gen).item()))
        elif ph == "real" and fsm.skeleton[fsm.real_i][1] == "span_start":
            last_span_cls = v.id2note[tok]
        fsm.step(tok)
        if ph == "denom" and denom_masks is not None:
            # publish the COMMITTED denominator: cur is re-read from bars_meta
            # on every later step of this bar, so this one write is what makes
            # _pos_time, base_pos, the DENS feasibility bound, the min-IOI
            # floor, the family bias and the export agree with the FSM.
            cur[1] = fsm.denom
            counters["bar_denom_sampled"] += 1
        if closing:
            open_cls = open_t0 = open_key = None
            open_dur_u = None
        elif ph == "real" and fsm.open_span and last_span_cls is not None:
            open_cls = last_span_cls
        seq.append(tok)
        st = _stage_of_token(tok)
        stages.append(st)
        seq_t = torch.cat(
            [seq_t, torch.tensor([[tok]], dtype=torch.long, device=device)],
            dim=1)
        if use_stage:
            stage_t = torch.cat(
                [stage_t, torch.tensor([[st]], dtype=torch.long,
                                       device=device)], dim=1)
        if ph == "plan3":
            plans.append({"bar": cur[0],
                          "dens": dens_rev[seq[-4]],
                          "span_budget": budget_rev[seq[-3]],
                          "sync": sync_rev[seq[-2]],
                          "motif": motif_rev[seq[-1]]})
        if tok == v.eos:
            eos = True
            break
    carry_out = ({"cls": open_cls, "t0": open_t0, "key": open_key,
                  "dur_u": open_dur_u} if fsm.open_span else None)
    return seq[len(prefix):], eos, carry_out, plans, prev_onset_t


def _rich_song_summary(mel):
    """[mean, std, max, last-first] per section, concatenated to
    ``(100, 4*N_MELS)`` for rich_section_stats models (block order per
    model.py's HierarchicalSectionEncoder). data.py should become the
    canonical owner when the train-side lands; keep the two in lockstep."""
    n_mels, T = mel.shape
    sections = 100
    if T < sections:
        raise ValueError(f"song has only {T} mel frames; need >= {sections}")
    edges = np.linspace(0, T, sections + 1, dtype=np.int64)
    out = np.empty((sections, 4 * n_mels), dtype=np.float32)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        seg = mel[:, a:b].astype(np.float32)
        out[k] = np.concatenate([seg.mean(1), seg.std(1), seg.max(1),
                                 seg[:, -1] - seg[:, 0]])
    return out


@torch.no_grad()
def generate_song_barscript(model, mel, grid, course, level=None,
                            density_bucket=None, spec=None, device="cuda",
                            greedy=False, temperature=1.0, top_p=0.95,
                            plan_temperature=None, seed=0, style=None,
                            sync_band=None, big_rate=None, span_rate=None,
                            stream=None, supplied_plan=None,
                            span_min_sec=SPAN_MIN_SEC,
                            span_min_beats=SPAN_MIN_BEATS,
                            degenerate_bar_sec=DEGENERATE_BAR_SEC,
                            min_onset_gap_sec=MIN_ONSET_GAP_SEC,
                            family_ref=None, family_bias=0.0,
                            family_hand_bias=0.0, family_mode="bias",
                            span_bucket0_profile=None, on_progress=None,
                            bar_denoms="supplied", bar_denom_mask="lattice"):
    """SoftChart 2.0 supplied-grid BarScript generation (§5).

    mel: (n_mels, T) log-mel. grid: trusted bar structure with explicit
    ``measure_edges`` (terminal edge included), exact meters and per-bar
    ``bar_denoms`` (see barscript_grid_from_bpm / barscript_grid_from_trace).
    The [BAR]+BAR_DENOM headers are teacher-forced from the grid; the model
    generates plan/skeleton/realization under the BarscriptFSM legal mask at
    every step. ``supplied_plan`` optionally teacher-forces per-bar plan
    fields (dens/span_budget/sync/motif). ``plan_temperature`` is the
    independent diversity knob for the four plan steps (defaults to
    ``temperature``). big_rate/span_rate/stream/sync_band take raw bucket
    ints for now (spec value->bucket mapping lands upstream).

    Returns a write_tja_slots-compatible dict plus BarScript extras:
    ``events`` [(bar, pos Fraction, class)], ``bar_plans``,
    ``span_dur_buckets``, ``window_tokens``, ``windows`` [(first_bar, K)],
    ``clamp_reasons``, ``slot_export_error`` and ``counters``
    {escape_hatch, forced_close, span_over_clamp, orphan_final_span,
    terminal_span_blocked, terminal_forced_close, slot_export_off_lattice,
    bars}.

    Terminal boundary: a SPAN_START is refused where the remaining chart
    cannot hold the shortest span the duration grid distinguishes, the
    duration head is masked against the remaining length as well as SPAN_MAX,
    and a span still open in the song's last bar has SPAN_END forced (counted
    as ``terminal_forced_close``, recorded in ``clamp_reasons`` and echoed on
    the affected ``spans`` entry as ``clamp_reason``).

    Corpus-empirical floors (all authored population minima; see the module
    constants for the measurement, and pass ``None`` to disable one):
    ``span_min_sec`` / ``span_min_beats`` steer span closures and mask
    duration buckets away from lengths no author ever wrote;
    ``degenerate_bar_sec`` blocks HITS (never spans) in sub-beat scaffolding
    measures; ``min_onset_gap_sec`` blocks onsets closer together than the
    authored minimum inter-onset gap.  Each is fail-open at the mask level and
    counted (``span_close_floor``, ``span_floor_carry``, ``dur_bucket_floor``,
    ``degenerate_bar_hits_blocked``, ``min_gap_blocked``).

    Short-span length profile (OFF BY DEFAULT; the serving path is
    bit-identical to the shipped one without it).  Duration bucket 0 is open
    at the bottom, so nothing bounds how short a span may be beyond the
    corpus floor, and the realized length piles up on the lattice minimum
    (MICROSPAN_CAUSE.md §6).  ``span_bucket0_profile`` is a
    :class:`SpanBucket0Profile`; each span draws one lower bound from the
    authored within-bucket-0 length distribution when it OPENS (the only
    point at which a same-bar close can still be steered -- see the class
    comment).  It is the LOWEST-priority constraint: it is interleaved into
    the closing ladder so the DENS feasibility guard, the SPAN_MAX ceiling
    and the corpus floor each outrank it, and it stands down entirely once
    the span can no longer close inside bucket 0, so a long span is never
    pulled back down.  Counted as ``span_bucket_floor`` /
    ``span_bucket_carry``.

    Nested difficulty family: ``family_ref`` is a ``FamilyRef`` -- the onset
    times (SECONDS, not lattice positions: ``#MEASURE`` is per-course, so two
    courses of one song share no bar lattice) of an already generated HARDER
    sibling.  See ``family_ref_from_generation`` / ``generate_chart_family``.
    ``family_mode='bias'`` (default) adds ``+/-family_bias`` to position
    logits and ``+family_hand_bias`` to the parent's drum hand;
    ``family_mode='subset'`` hard-restricts positions and is for ablation.

    Bar lattice (``bar_denoms``).  ``"supplied"`` (default, unchanged
    behaviour) teacher-forces the grid's per-bar ``bar_denoms`` into every
    [BAR] header.  ``"model"`` lets the model emit its own, restricted to
    ``bar_denom_mask`` (``"lattice"`` = divisors of the bar's own TJA export
    lattice, the default and a superset of ``"div96"`` outside 4/4;
    ``"div96"``; ``None`` = unrestricted).  DEPLOY_GRID_FIX.md §4-§5: the
    supplied denominator is read by the model as a DENSITY ANNOUNCEMENT, so a
    uniform serving grid inflates easy by 69 % and erases every triplet, while
    the emitted denominators ladder by difficulty like the authored ones.  The
    cost is nesting (§4.4), so this is a deliberate two-sided choice and not a
    strict improvement; the training-side fix is ``--denom-augment``.  In
    ``"model"`` mode the grid's ``bar_denoms`` are still validated but only
    used for the terminal-boundary geometry, where the LARGEST allowed
    denominator of the final bar is assumed (a permissive bound: it can cost
    at most one span, counted as ``orphan_final_span``).  The realized
    denominators are returned as ``bar_denoms``.

    Export: a bar-local position off the 96-slot TJA lattice is a
    RECORD-level failure — ``slot_export_error`` carries the reason,
    ``hits_slots``/``spans_slots`` become ``None`` and everything else stays
    intact, so a harness records ``gen_tja = null`` instead of dropping the
    record.  ``require_slot_export`` turns it into a ``SlotExportError``.
    """
    from .barscript import BarscriptSpec, barscript_to_events
    from .vocab import SLOTS

    if spec is None:
        spec = BarscriptSpec()
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel.astype(np.float32))
    T = mel.shape[1]
    edges, slots, num, den, quarter, denoms = _barscript_grid(grid)
    n_bars = len(denoms)
    if bar_denoms not in ("supplied", "model"):
        raise ValueError("bar_denoms must be 'supplied' or 'model'")
    denom_masks = None
    if bar_denoms == "model":
        denom_masks = []
        for bi in range(n_bars):
            m = np.zeros(VOCAB.size, dtype=bool)
            allowed = bar_denom_allowed(int(slots[bi]), bar_denom_mask)
            if not allowed:
                raise ValueError(
                    f"bar {bi}: bar_denom_mask {bar_denom_mask!r} admits no "
                    f"denominator on a {int(slots[bi])}-slot bar")
            for d in allowed:
                m[VOCAB.bar_denom[d]] = True
            denom_masks.append(m)
    plan_overrides = _normalize_supplied_plan(supplied_plan, n_bars)
    if plan_temperature is None:
        plan_temperature = temperature
    use_phase = bool(getattr(model, "clean_phase", False)) \
        or getattr(model, "in_ch", N_MELS) > N_MELS

    # window packing: consecutive whole bars, WINDOW frames per window
    wins, metas = [], []
    j = 0
    while j < n_bars:
        K = 0
        while (j + K < n_bars
               and (edges[j + K + 1] - edges[j]) * FPS <= WINDOW):
            K += 1
        if K == 0:
            raise ValueError(f"bar {j} does not fit the {WINDOW}-frame window")
        t0 = edges[j]
        f0 = int(round(t0 * FPS))
        src = max(1, int(round((edges[j + K] - t0) * FPS)))
        x = torch.full((mel.shape[0], src), float(np.log(1e-5)),
                       dtype=mel.dtype, device=mel.device)
        src0, src1 = max(0, f0), min(T, f0 + src)
        if src1 > src0:
            dst0 = src0 - f0
            x[:, dst0:dst0 + (src1 - src0)] = mel[:, src0:src1]
        if x.shape[1] < WINDOW:
            x = torch.nn.functional.pad(
                x, (0, WINDOW - x.shape[1]), value=float(np.log(1e-5)))
        if use_phase:
            ph = torch.full((2, WINDOW), -1.0)
            ef = (edges[j:j + K + 1] - t0) * FPS
            for m in range(K):
                a, b = ef[m], ef[m + 1]
                i0 = int(np.ceil(a - 1e-6))
                i1 = min(int(np.ceil(b - 1e-6)), WINDOW)
                if i1 <= i0:
                    continue
                frac = (torch.arange(i0, i1, dtype=torch.float32) - a) \
                    / max(b - a, 1e-6)
                ph[0, i0:i1] = frac
                ph[1, i0:i1] = (frac * float(quarter[j + m])) % 1.0
            x = torch.cat([x, ph], dim=0)
        wins.append(x)
        metas.append((j, K))
        j += K

    gsum = None
    if getattr(model, "_has_gctx", False):
        mel_np = mel.detach().cpu().numpy()
        sec = getattr(model, "section_encoder", None)
        if sec is not None and getattr(sec, "rich_stats", False):
            gsum = torch.from_numpy(_rich_song_summary(mel_np))
        else:
            from .data import song_summary

            gsum = torch.from_numpy(song_summary(mel_np))

    prefix = build_prefix_barscript(
        model, course, level, density_bucket, style=style,
        sync_band=sync_band, big_rate=big_rate, span_rate=span_rate,
        stream=stream)

    counters = {"escape_hatch": 0, "forced_close": 0, "span_over_clamp": 0,
                "orphan_final_span": 0, "terminal_span_blocked": 0,
                "terminal_forced_close": 0, "slot_export_off_lattice": 0,
                # §1.3 floors (symmetric to the SPAN_MAX ceiling)
                "span_close_floor": 0, "span_floor_carry": 0,
                "dur_bucket_floor": 0,
                # opt-in duration-bucket closing steer
                "span_bucket_floor": 0, "span_bucket_carry": 0,
                # onset-placement floors
                "degenerate_bar_dens_capped": 0,
                "degenerate_bar_hits_blocked": 0,
                "degenerate_bar_fail_open": 0,
                "min_gap_blocked": 0, "min_gap_fail_open": 0,
                # nested-family steering
                "family_bias_steps": 0, "family_hand_bias_steps": 0,
                "family_subset_restricted": 0,
                "family_subset_failed_open": 0,
                # model-emitted bar lattice
                "bar_denom_sampled": 0, "bar_denom_mask_fail_open": 0,
                "bars": n_bars}
    span_buckets = []
    clamp_reasons = []
    # Terminal-boundary geometry: the last bar-local position in the whole
    # song at which a SPAN_END can still be placed, in audio seconds.
    final_bar = n_bars - 1
    # Under a model-emitted lattice the final bar's denominator is not known
    # until it is decoded, which is after every window that could open a span
    # into it; the largest ALLOWED denominator is the permissive bound (see
    # the docstring and DEPLOY_GRID_FIX §6.3's last loose end).
    final_denom = (max(bar_denom_allowed(int(slots[final_bar]), bar_denom_mask))
                   if denom_masks is not None else denoms[final_bar])
    last_close_t = float(edges[final_bar]) + (
        (final_denom - 1) / final_denom
        * float(edges[final_bar + 1] - edges[final_bar]))
    all_events, bar_plans, window_tokens, windows = [], [], [], []
    carry = None
    prev_onset_t = None  # threads the min-IOI floor across window boundaries
    for w, (j0, K) in enumerate(metas):
        x = wins[w].unsqueeze(0).to(device)
        enc_kw = {}
        if gsum is not None:
            n_pos = (100 if getattr(model, "_hierarchical_ctx", False)
                     else getattr(getattr(model, "pos_emb", None),
                                  "num_embeddings", 16))
            enc_kw = dict(
                gsum=gsum.unsqueeze(0).to(device),
                pos_bucket=torch.tensor(
                    [min(n_pos - 1,
                         max(0, int(n_pos * edges[j0] * FPS / max(T, 1))))],
                    dtype=torch.long, device=device))
        with _autocast(device):
            memory = model.encode(x, **enc_kw)
            memory = model.phase_mem(memory, x)
        bars_meta = [
            [j0 + m, denoms[j0 + m], float(edges[j0 + m]),
             float(edges[j0 + m + 1]),
             float((edges[j0 + m + 1] - edges[j0 + m]) / quarter[j0 + m])]
            for m in range(K)]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + w)
        toks, eos, carry, plans, prev_onset_t = _decode_barscript_window(
            model, memory, prefix, bars_meta, spec, carry, plan_overrides,
            device, greedy, temperature, plan_temperature, top_p, gen,
            counters, span_buckets, last_close_t=last_close_t,
            final_bar=final_bar, clamp_reasons=clamp_reasons,
            span_min_sec=span_min_sec, span_min_beats=span_min_beats,
            degenerate_bar_sec=degenerate_bar_sec,
            min_onset_gap_sec=min_onset_gap_sec, prev_onset_t=prev_onset_t,
            family_ref=family_ref, family_bias=family_bias,
            family_hand_bias=family_hand_bias, family_mode=family_mode,
            course=course, span_bucket0_profile=span_bucket0_profile,
            denom_masks=denom_masks)
        window_tokens.append(list(toks))
        windows.append((j0, K))
        for meta in bars_meta:  # realized lattice (== supplied unless "model")
            denoms[meta[0]] = int(meta[1])
        # A window that hands an open span to its successor stops before EOS
        # (the FSM masks EOS while a span is open); terminate the re-zip
        # copy only — the model never saw this EOS.
        ev = barscript_to_events(toks if eos else toks + [VOCAB.eos])
        all_events += [(j0 + b, pos, cls) for b, pos, cls in ev]
        bar_plans += plans
        if on_progress:
            on_progress(w + 1, len(metas))
    if carry is not None:
        # Span still open after the final bar even though the terminal forced
        # closure could not fire (no legal SPAN_END position was left): drop
        # it (counted), never invent an endpoint the model did not emit.
        counters["orphan_final_span"] += 1
        key = carry.get("key")
        clamp_reasons.append({
            "reason": "orphan_final_span_dropped",
            "bar": int(key[0]) if key else None,
            "pos_num": int(key[1].numerator) if key else None,
            "pos_den": int(key[1].denominator) if key else None,
            "class": carry.get("cls"),
            "detail": ("span open after the final bar with no legal closing "
                       "position remaining; start dropped from the export"),
        })

    # Every clamp keyed by the span START it applies to, so the exported span
    # can name what shaped it.  ``spans[].clamp_reason`` keeps its v2 shape (a
    # single string or None) and its v2 meaning: the reason that determined
    # WHERE THE SPAN ENDED.  CLAMP_PRIORITY therefore ranks the closure-
    # determining reasons above the plan-stage duration mask, so adding the
    # §1.3 floors cannot re-attribute a span the terminal treatment closed.
    # The complete per-span list stays in the top-level ``clamp_reasons``.
    by_start = {}
    for r in clamp_reasons:
        if r["bar"] is None or r["reason"] == "orphan_final_span_dropped":
            continue
        by_start.setdefault(
            (r["bar"], Fraction(r["pos_num"], r["pos_den"])), []
        ).append(r["reason"])
    hits, spans, hits_slots, spans_slots = [], [], [], []
    slot_error = None
    open_sp = None
    for bar, pos, cls in all_events:
        t = float(edges[bar]) + float(pos) * float(edges[bar + 1] - edges[bar])
        slot_f = pos * int(slots[bar])
        if slot_f.denominator != 1:
            # Record-level export failure, NOT a generation failure: the
            # decoded chart stays whole and only the slot lists are withheld,
            # so a harness writes gen_tja = null instead of losing the record
            # (non-random loss — unusual meters/lattices only). See
            # SlotExportError / require_slot_export.
            counters["slot_export_off_lattice"] += 1
            if slot_error is None:
                slot_error = (
                    f"bar {bar}: /{pos.denominator} lattice position does not "
                    f"sit on the {int(slots[bar])}-slot TJA export lattice; "
                    "choose 96-lattice-compatible bar_denoms")
            sl = None
        else:
            sl = int(slot_f)
        if cls in HIT_CLASSES:
            hits.append({"t": round(t, 4), "type": cls})
            if sl is not None:
                hits_slots.append((bar, sl, cls))
        elif cls in SPAN_CLASSES:
            open_sp = (bar, sl, t, cls, pos)
        elif cls == "end" and open_sp is not None:
            # The explicit END position is authoritative for both the TJA
            # and the seconds preview; over-clamp survivors (closure had no
            # legal in-limit position) are counted, not rewritten.
            if t - open_sp[2] > SPAN_MAX.get(open_sp[3], 6.5) + 1e-9:
                counters["span_over_clamp"] += 1
            why = sorted(by_start.get((open_sp[0], open_sp[4]), []),
                          key=lambda r: CLAMP_PRIORITY.index(r)
                          if r in CLAMP_PRIORITY else len(CLAMP_PRIORITY))
            spans.append({"t0": round(open_sp[2], 4), "t1": round(t, 4),
                          "type": open_sp[3],
                          "clamp_reason": (why[0] if why else None)})
            if sl is not None and open_sp[1] is not None:
                spans_slots.append(
                    (open_sp[0], open_sp[1], bar, sl, open_sp[3]))
            open_sp = None
    if slot_error is not None:
        # withhold a truncated slot export rather than let a caller write a
        # silently incomplete TJA
        hits_slots = spans_slots = None
    return {"hits": hits, "spans": spans, "hits_slots": hits_slots,
            "spans_slots": spans_slots, "n_measures": n_bars,
            "measure_slots": [int(s) for s in slots],
            "measure_num": [int(x) for x in num],
            "measure_den": [int(x) for x in den],
            "slots_per_4_4": SLOTS,
            "bar_denoms": [int(d) for d in denoms],
            "bar_denom_source": bar_denoms,
            "course": course, "level": level,
            "density_bucket": density_bucket,
            "events": all_events, "bar_plans": bar_plans,
            "span_dur_buckets": span_buckets,
            "window_tokens": window_tokens, "windows": windows,
            "clamp_reasons": clamp_reasons,
            "slot_export_error": slot_error,
            "counters": counters}


# Ordering of the Taiko difficulty ladder, hardest first.  A family is
# generated top-down: the hardest course is free, every easier course decodes
# against the one immediately above it.
COURSE_LADDER = ("ura", "oni", "hard", "normal", "easy")

# Default nested-family steering strengths.  PROVISIONAL: they are logit
# offsets on an untrained-at-this-setting decoder and no GPU run has yet
# measured the coverage they produce, so they are a documented starting point
# for the sweep in the GPU verification list, NOT a calibrated value.  The
# target the sweep aims at IS calibrated: authored adjacent-course coverage is
# 0.95-0.99 (exact-lattice subset 0.905-0.967) and authored hand agreement is
# 0.70-0.82, both measured on the 13-song ladder population.
FAMILY_BIAS = 2.0
FAMILY_HAND_BIAS = 1.5


@torch.no_grad()
def generate_chart_family(model, mel, grid, courses, levels=None,
                          density_buckets=None, family_bias=FAMILY_BIAS,
                          family_hand_bias=FAMILY_HAND_BIAS,
                          family_mode="bias", on_course=None, **kwargs):
    """Generate a whole nested difficulty family for one song.

    Authored Taiko charts are a FAMILY, not four independent charts: the same
    note sits at the same instant across courses (adjacent-pair coverage
    0.95-0.99 authored, against 0.77-0.82 when each (song, course) is generated
    independently).  This entry point restores the convention at decode time --
    no retraining -- by generating the hardest requested course freely and
    conditioning every easier course on the chart immediately above it.

    ``courses`` is any iterable of course names; they are reordered
    hardest-first by COURSE_LADDER regardless of the order given, because the
    constraint only makes sense in that direction.  ``grid`` is either one
    supplied grid for the whole family or a ``{course: grid}`` map -- authored
    grids differ per course (``#MEASURE`` is per-course; ``test_00000`` is 97
    bars on hard/oni and 60 on easy), which is exactly why the family reference
    is carried in seconds rather than in lattice positions.  ``levels`` and
    ``density_buckets`` are optional ``{course: value}`` maps.  Remaining
    keyword arguments go to generate_song_barscript unchanged, so seeds,
    knobs, spec and the floors all behave exactly as in single-course serving.

    Returns ``{course: generation dict}``; each dict carries the usual
    ``counters``, with ``family_parent`` naming the course it was conditioned
    on (None for the hardest).
    """
    levels = levels or {}
    density_buckets = density_buckets or {}
    unknown = [c for c in courses if c not in COURSE_LADDER]
    if unknown:
        raise ValueError(f"unknown courses {unknown}; expected "
                         f"a subset of {list(COURSE_LADDER)}")
    ordered = [c for c in COURSE_LADDER if c in set(courses)]
    if not ordered:
        raise ValueError("generate_chart_family needs at least one course")
    if family_mode not in ("bias", "subset"):
        raise ValueError("family_mode must be 'bias' or 'subset'")

    from collections.abc import Mapping as _Mapping

    per_course = (isinstance(grid, _Mapping)
                  and all(c in grid for c in ordered)
                  and "downbeats" not in grid)
    if per_course:
        missing = [c for c in ordered if c not in grid]
        if missing:
            raise ValueError(f"grid map is missing courses {missing}")

    out = {}
    ref = None
    parent = None
    for course in ordered:
        gen = generate_song_barscript(
            model, mel, grid[course] if per_course else grid, course,
            level=levels.get(course), density_bucket=density_buckets.get(course),
            family_ref=ref,
            family_bias=(0.0 if ref is None else family_bias),
            family_hand_bias=(0.0 if ref is None else family_hand_bias),
            family_mode=family_mode, **kwargs)
        gen["family_parent"] = parent
        out[course] = gen
        # The next (easier) course inherits THIS course's realized onsets, so
        # the constraint chains down the ladder the way authored families do
        # rather than every course pointing at the single hardest chart.
        ref = family_ref_from_generation(gen)
        parent = course
        if on_course:
            on_course(course, gen)
    return out
