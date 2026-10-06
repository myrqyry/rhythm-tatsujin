"""Inference: full-song chart generation with constrained autoregressive decoding.

Features:
- structural constraints via logit masking: TIME/NOTE alternation, strictly
  increasing times, minimum playable gap (4 frames ~= 46 ms);
- classifier-free guidance (cfg_w > 1): contrasts conditional vs
  condition-UNK logits to strengthen condition adherence;
- previous-window pattern context (v2 models): two-pass generation, pass 2
  conditions each window on the tail of the previous window's pass-1 output.
"""

import numpy as np
import torch

from .model import ChartModel
from .vocab import FPS, MAX_TGT, N_LEVELS, NOTE_CLASSES, VOCAB, WINDOW

HIT_CLASSES = ("don", "ka", "don_big", "ka_big")
SPAN_CLASSES = ("roll", "roll_big", "balloon")


def _dev_type(device):
    return device.split(":")[0] if isinstance(device, str) else device.type


def _autocast(device):
    """Use the best safe CUDA autocast dtype for the actual GPU."""
    import contextlib

    if _dev_type(device) != "cuda":
        return contextlib.nullcontext()
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)
CTX_LEN = 12  # fixed-length previous-window context (note-type tokens)
MIN_GAP_FRAMES = 4  # ~46 ms minimum inter-note gap enforced during decoding
LATTICE_RATIOS = (1 / 3, 0.5, 2 / 3, 1.0, 4 / 3, 1.5, 2.0, 3.0)  # allowed IOI ratios
# GT span-length p99 per class (train stats); generated spans beyond this are
# truncated — a visual audit found generated balloons up to ~10s vs GT p50 ~1s
SPAN_MAX = {"roll": 3.5, "roll_big": 4.0, "balloon": 6.5}


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"]
    vocab_size = sd["tok_emb.weight"].shape[0]
    aux = any(k.startswith("aux") for k in sd)
    gctx = any(k.startswith("gsum_proj") for k in sd)
    a = ck.get("args", {}) or {}
    front_ch = sd["frontend.0.weight"].shape[1]  # 128 (clean) or 130 (legacy slot)
    emb_w = sd["tok_emb.weight"].shape[1]
    d_model = a.get("d_model", 512)
    # clean_phase (v1.6): encoder frontend is 128ch and a phase_proj is present.
    clean_phase = bool(a.get("clean_phase", False)) or ("phase_proj.0.weight" in sd)
    # legacy slot models fed phase to the encoder (in_ch=130); clean models keep
    # in_ch=130 at the data level but only 128 reach the frontend.
    in_ch = 130 if (clean_phase or front_ch == 130) else front_ch
    # v1.7 arch flags: prefer saved args; fall back to state-dict keys so
    # args-less checkpoints of the new arch still load
    adapter_rank_ffn = int(a.get("adapter_rank_ffn", 0) or 0)
    if not adapter_rank_ffn and "decoder.ffn_adapters.0.down.weight" in sd:
        adapter_rank_ffn = sd["decoder.ffn_adapters.0.down.weight"].shape[0]
    depth_emb = bool(a.get("depth_emb", False)) or ("decoder.depth_emb" in sd)
    model = ChartModel(
        d_model=d_model, nhead=a.get("nhead", 8),
        enc_layers=a.get("enc_layers", 6),
        dec_layers=a.get("dec_layers", 6), ffn=a.get("ffn", 2048),
        vocab_size=vocab_size, aux=aux, global_ctx=gctx,
        func_time=a.get("func_time", False), in_ch=in_ch,
        emb_factor=emb_w if emb_w != d_model else None,
        clean_phase=clean_phase,
        enc_share=a.get("enc_share") or None,
        dec_share=a.get("dec_share") or None,
        adapter_rank=int(a.get("adapter_rank", 0) or 0),
        unique_layernorm=bool(a.get("unique_layernorm", False)),
        adapter_rank_ffn=adapter_rank_ffn, depth_emb=depth_emb,
        unshare_last_dec=bool(a.get("unshare_last_dec", False)),
    ).to(device)
    if any(k.startswith("ptr") for k in sd):
        model.enable_ptr()
    if any(k.startswith("beat") for k in sd):
        model.enable_beat(hires=any(k.startswith("beat.up") for k in sd))
    model.to(device)  # newly enabled heads default to CPU
    model.load_state_dict(sd)
    model.eval()
    # capability detection: prefer explicit training args (vocab-size inference
    # wrongly added prefix slots to models that never trained with them)
    if a:
        model._has_ctx = bool(a.get("ctx", False))
        model._has_sib = bool(a.get("sibling", False))
        model._has_style = bool(a.get("style", False))
        model._has_sync = bool(a.get("sync_token", False))
        model._has_plan = bool(a.get("plan", False))
        model._dual = bool(a.get("dual", False))
        model._slot = bool(a.get("slot", False)) or model._dual
    else:  # legacy fallback
        model._has_ctx = vocab_size > VOCAB.sep
        model._has_sib = vocab_size > VOCAB.sib
        model._has_style = False
        model._has_sync = False
        model._has_plan = False
        model._slot = False
        model._dual = False
    model._has_gctx = gctx
    return model


def load_hf(repo_or_dir, device="cuda"):
    """Load a Hugging Face SoftChartGenerator (safetensors) and attach the
    capability flags that generate_song() reads, so HF models are drop-in."""
    from .hf import SoftChartGenerator

    hf = SoftChartGenerator.from_pretrained(repo_or_dir).to(device).eval()
    m = hf.net
    caps = hf.capabilities or {}
    m._has_ctx = bool(caps.get("ctx", False))
    m._has_sib = bool(caps.get("sibling", False))
    m._has_style = bool(caps.get("style", False))
    m._has_sync = bool(caps.get("sync_token", False))
    m._has_plan = bool(caps.get("plan", False))
    m._dual = bool(caps.get("dual", False))
    m._slot = bool(caps.get("slot", False)) or m._dual
    m._has_gctx = m.gsum_proj is not None
    return m


def build_prefix(course=None, level=None, density_bucket=None, ctx_types=None,
                 uncond=False, sib_pairs=None, style=None, sync_band=None,
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
    if uncond:
        seq += [v.unk_cond] * 3
    else:
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


@torch.no_grad()
def decode_windows(model, mels, prefixes, device="cuda", temperature=1.0, top_p=0.95,
                   greedy=False, seed=0, cfg_w=0.0, gsum=None, pos_buckets=None,
                   lattice=False, n_cond=3, min_gap=MIN_GAP_FRAMES):
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
        unc = seqs.clone()
        unc[:, -n_cond:] = v.unk_cond
        seqs = torch.cat([seqs, unc], dim=0)
        memory = torch.cat([memory, memory], dim=0)

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
            ok = time_pos >= (last_time[idx] + min_gap).unsqueeze(1)
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
            pb = torch.tensor(
                [min(15, int(16 * starts[i + j] / max(T, 1))) for j in range(chunk.shape[0])],
                dtype=torch.long)
        evs = decode_windows(model, chunk, prefixes, device=device, greedy=greedy,
                             temperature=temperature, top_p=top_p, seed=seed + i,
                             cfg_w=cfg_w, gsum=g, pos_buckets=pb, lattice=lattice,
                             n_cond=4 if style is not None else 3)
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
    starts = list(range(0, max(T - 1, 1), WINDOW))
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

        gsum = torch.from_numpy(song_summary(mel.numpy()))

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

    if True:
        # rescue decoding: decoding occasionally EOSes a whole window early; if a
        # window is near-empty while its audio is musically active, redo it with
        # the opposite mode (sampling was empty -> greedy; greedy was empty ->
        # sampled retry, since a greedy redo would reproduce the same output)
        flux = [float(np.maximum(0, np.diff(w.numpy(), axis=1)).sum()) for w in wins]
        med = float(np.median(flux)) if flux else 0.0
        med_hits = float(np.median([len(ev) for ev in per_window])) if per_window else 0.0
        retry = [i for i, ev in enumerate(per_window)
                 if (len(ev) < 4 and flux[i] > 0.3 * med)
                 or (len(ev) < 0.4 * med_hits and flux[i] > 0.7 * med)]
        if retry:
            r_evs, _ = _decode_pass(
                model, [wins[i] for i in retry], [starts[i] for i in retry],
                course, level, density_bucket, None, device, not greedy, 0.9,
                top_p, seed + 7, 0.0, batch_windows, gsum=gsum, T=T,
                lattice=lattice, sib_default=sib_default, style=style_val,
                sync_band=sync_val, plan_blocks=plan_blocks, plan_default=plan_default,
                mode=("time" if dual else None))
            all_events = [e for i, w_ev in enumerate(per_window) if i not in retry
                          for e in [(starts[i] / FPS + f / FPS, c) for f, c in w_ev]]
            all_events += r_evs

    all_events.sort(key=lambda e: e[0])
    hits, spans = [], []
    open_span = None
    last_hit_t = -1.0
    for t, cls in all_events:
        if cls in HIT_CLASSES:
            if open_span is not None:
                # a span whose end never arrived would swallow every following
                # hit (visual audit: 20 s of silence) — force-close at SPAN_MAX
                if t - open_span[0] > SPAN_MAX.get(open_span[1], 6.5):
                    spans.append({"t0": round(open_span[0], 4),
                                  "t1": round(open_span[0] + SPAN_MAX.get(open_span[1], 6.5), 4),
                                  "type": open_span[1]})
                    open_span = None
                else:
                    continue  # no hits inside an open span
            if t - last_hit_t < 0.025:
                continue
            hits.append({"t": round(t, 4), "type": cls})
            last_hit_t = t
        elif cls in SPAN_CLASSES:
            if open_span is None:
                open_span = (t, cls)
        elif cls == "end":
            if open_span is not None and t - open_span[0] > 0.05:
                t1_span = min(t, open_span[0] + SPAN_MAX.get(open_span[1], 6.5))
                spans.append({"t0": round(open_span[0], 4), "t1": round(t1_span, 4),
                              "type": open_span[1]})
                open_span = None
    return {"hits": hits, "spans": spans, "course": course, "level": level,
            "density_bucket": density_bucket}


@torch.no_grad()
def generate_song_slot(model, mel, grid, course, level=None, density_bucket=None,
                       device="cuda", greedy=False, temperature=1.0, top_p=0.95,
                       seed=0, batch_windows=8, plan=None, on_progress=None):
    """Slot-mode generation: windows are anchored at the fitted barlines and the
    decoder emits exact TJA lattice indices (measure*96 + slot). No
    quantization step exists — 'hits_slots' ARE the chart.

    mel: (n_mels, T); grid: fit_grid() result (must be trustworthy).
    Returns {hits, spans (seconds, for rendering/metrics),
             hits_slots [(measure, slot, cls)], spans_slots, n_measures}.
    """
    from .vocab import MEAS_MAX, SLOTS

    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel.astype(np.float32))
    T = mel.shape[1]
    dur = T / FPS
    db = np.asarray(grid["downbeats"], np.float64)
    bar = float(grid["bar"])
    edges = np.append(db, db[-1] + bar)  # measure m spans [edges[m], edges[m+1])

    wins, metas = [], []  # meta = (first_measure_idx, K)
    j = 0
    while j < len(db):
        K = 0
        while (j + K < len(db) and K < MEAS_MAX
               and (edges[j + K + 1] - edges[j]) * FPS <= WINDOW):
            K += 1
        if K == 0:
            break
        t0 = edges[j]
        f0 = int(round(t0 * FPS))
        src = max(1, int(round((edges[j + K] - t0) * FPS)))
        x = mel[:, f0 : f0 + src]
        if x.shape[1] < src:
            x = torch.nn.functional.pad(x, (0, src - x.shape[1]), value=float(np.log(1e-5)))
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
            ph[1, i0:i1] = (frac * 4) % 1.0
        wins.append(torch.cat([x, ph], dim=0))
        metas.append((j, K))
        j += K

    sib_default = getattr(model, "_has_sib", False)
    style_val = -1 if getattr(model, "_has_style", False) else None
    plan_blocks = plan if getattr(model, "_has_plan", False) else None
    plan_default = getattr(model, "_has_plan", False) and plan is None

    events = []  # (measure_global, slot, cls) in decode order
    for i0 in range(0, len(wins), batch_windows):
        chunk = torch.stack(wins[i0 : i0 + batch_windows])
        prefixes = []
        for b_ in range(chunk.shape[0]):
            jj, KK = metas[i0 + b_]
            t0, t1 = edges[jj], edges[jj + KK]
            prefixes.append(build_prefix(
                course, level, density_bucket,
                mode=("slot" if getattr(model, "_dual", False) else None),
                sib_pairs=([] if sib_default else None), style=style_val,
                plan_slice=([(b[2], b[3]) for b in plan_blocks
                             if b[1] > t0 and b[0] < t1] if plan_blocks is not None
                            else ([] if plan_default else None))))
        evs = decode_windows(model, chunk, prefixes, device=device, greedy=greedy,
                             temperature=temperature, top_p=top_p, seed=seed + i0,
                             n_cond=4 if style_val is not None else 3, min_gap=2)
        for b_, w_ev in enumerate(evs):
            jj, KK = metas[i0 + b_]
            for g, cls in w_ev:
                if g < KK * SLOTS:
                    events.append((jj + g // SLOTS, g % SLOTS, cls))
        if on_progress:
            on_progress(min(i0 + batch_windows, len(wins)), len(wins))

    events.sort(key=lambda e: (e[0], e[1]))
    def _t(me, sl):
        return float(edges[me] + sl / SLOTS * (edges[me + 1] - edges[me])) \
            if me < len(edges) - 1 else float(edges[-1])

    hits, spans, hits_slots, spans_slots = [], [], [], []
    open_span = None
    last_key = None
    for me, sl, cls in events:
        t = _t(me, sl)
        if cls in HIT_CLASSES:
            if open_span is not None:
                if t - open_span[2] > SPAN_MAX.get(open_span[3], 6.5):
                    t1s = open_span[2] + SPAN_MAX.get(open_span[3], 6.5)
                    spans.append({"t0": round(open_span[2], 4), "t1": round(t1s, 4),
                                  "type": open_span[3]})
                    spans_slots.append((open_span[0], open_span[1], me, sl, open_span[3]))
                    open_span = None
                else:
                    continue
            if (me, sl) == last_key:
                continue
            hits.append({"t": round(t, 4), "type": cls})
            hits_slots.append((me, sl, cls))
            last_key = (me, sl)
        elif cls in SPAN_CLASSES:
            if open_span is None:
                open_span = (me, sl, t, cls)
        elif cls == "end":
            if open_span is not None and t - open_span[2] > 0.05:
                t1s = min(t, open_span[2] + SPAN_MAX.get(open_span[3], 6.5))
                spans.append({"t0": round(open_span[2], 4), "t1": round(t1s, 4),
                              "type": open_span[3]})
                spans_slots.append((open_span[0], open_span[1], me, sl, open_span[3]))
                open_span = None
    n_meas = (metas[-1][0] + metas[-1][1]) if metas else 0
    return {"hits": hits, "spans": spans, "hits_slots": hits_slots,
            "spans_slots": spans_slots, "n_measures": n_meas,
            "course": course, "level": level, "density_bucket": density_bucket}
