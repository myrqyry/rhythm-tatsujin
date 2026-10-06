"""PyTorch dataset: (mel window, token sequence) pairs from the preprocessed cache."""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import (FPS, MAX_TGT, MEAS_MAX, NOTE_CLASSES, SLOTS, VOCAB, WINDOW,
                    encode_window)

G_CHUNK = 172  # ~2s per whole-song summary chunk
G_LEN = 100  # fixed number of summary chunks (covers ~200s songs)


def song_summary(mel):
    """(n_mels, T) log-mel -> (G_LEN, n_mels) coarse whole-song summary."""
    n_mels, T = mel.shape
    n = min(G_LEN, max(1, T // G_CHUNK))
    out = np.full((G_LEN, n_mels), np.log(1e-5), dtype=np.float32)
    for k in range(n):
        out[k] = mel[:, k * G_CHUNK : (k + 1) * G_CHUNK].astype(np.float32).mean(axis=1)
    return out


class ChartWindowDataset(Dataset):
    """One item = one (song, course) chart; a random window is cropped per access."""

    def __init__(
        self,
        cache_dir,
        song_ids,
        train=True,
        cond_drop=0.15,
        spec_augment=True,
        windows_per_chart=8,
        use_ctx=False,
        aux=False,
        global_ctx=False,
        importance_sampling=False,
        tempo_aug=False,
        sibling=False,
        style=False,
        beat_head=False,
        sync_token=False,
        plan=False,
        mask_infill=0.0,
        complexity=False,
        slot=False,
        dual=False,
        dual_slot_p=0.65,
        beat_hires=False,
    ):
        self.cache = cache_dir
        self.train = train
        self.slot = slot
        self.dual = dual  # mixed slot/time batches with an explicit mode token
        self.dual_slot_p = dual_slot_p
        self._meas_cache = {}
        self.cond_drop = cond_drop if train else 0.0
        self.spec_augment = spec_augment and train
        self.use_ctx = use_ctx
        self.aux = aux
        self.global_ctx = global_ctx
        self.importance = importance_sampling and train
        self.tempo_aug = tempo_aug and train
        self.sibling = sibling
        self.beat_head = beat_head
        self.beat_hires = beat_hires
        self.sync_token = sync_token
        self.mask_infill = mask_infill if train else 0.0
        self.complexity = complexity
        self.plans = None
        if plan:
            with open(os.path.join(cache_dir, "plans.json")) as f:
                self.plans = json.load(f)
        self.styles = None
        if style:
            with open(os.path.join(cache_dir, "styles.json")) as f:
                self.styles = json.load(f)["styles"]
        with open(os.path.join(cache_dir, "index.json")) as f:
            index = {e["id"]: e for e in json.load(f)}
        self.items = []  # (sid, course, level, n_frames)
        for sid in song_ids:
            e = index.get(sid)
            if e is None:
                continue
            for c, info in e["courses"].items():
                self.items.append((sid, c, info["level"], e["n_frames"]))
        if self.slot and not self.dual:  # pure slot mode needs a grid per song
            keep = []
            for it in self.items:
                src = it[0].split("x")[0] if "x" in it[0] else it[0]
                if os.path.exists(os.path.join(cache_dir, f"{src}.beats.npz")):
                    keep.append(it)
            self.items = keep
        # virtual epoch length: several windows per chart
        self.mult = windows_per_chart if train else 1
        self._rng = random.Random(int(os.environ.get('SC_DATA_SEED', 1234)))

    def __len__(self):
        return len(self.items) * self.mult

    def _load(self, sid):
        mel = np.load(os.path.join(self.cache, f"{sid}.mel.npy"), mmap_mode="r")
        notes = np.load(os.path.join(self.cache, f"{sid}.notes.npz"))
        return mel, notes

    def _beats(self, sid):
        try:
            z = np.load(os.path.join(self.cache, f"{sid}.beats.npz"))
            return z["beats"], z["downbeats"]
        except Exception:
            return None, None

    def _measures(self, sid):
        """Slot mode: (downbeat_times, beats_per_measure, ok_mask) — measure m
        spans [db[m], db[m+1]). ok = sane meter (2..6 beats) and duration.
        4/4-only models pass meter_filter=4 at window time."""
        if sid in self._meas_cache:
            return self._meas_cache[sid]
        bts, dbs = self._beats(sid)
        out = None
        if dbs is not None and len(dbs) >= 3:
            db = np.asarray(dbs, np.float64)
            durs = np.diff(db)
            if bts is not None and len(bts):
                nb = np.searchsorted(bts, db[1:] - 1e-4) - np.searchsorted(bts, db[:-1] - 1e-4)
            else:
                nb = np.full(len(durs), 4)
            ok = (nb >= 2) & (nb <= 6) & (durs > 0.5) & (durs < 8.0)
            if ok.any():
                out = (db, nb.astype(np.int64), ok)
        self._meas_cache[sid] = out
        return out

    def _getitem_slot(self, i):
        """Slot-token window: measure-aligned crop; note positions are exact
        TJA lattice indices instead of audio frames. Lattice (v3) is
        SLOTS_PER_BEAT per beat, so any meter is exact (4/4 -> 96/measure,
        3/4 -> 72). Slot labels are tempo-INVARIANT, so tempo augmentation
        only touches the mel. Two phase channels (measure/beat sawtooth) are
        appended so the decoder reads the grid instead of inferring it."""
        from .vocab import BEATS_MAX, SLOTS_PER_BEAT

        sid, course, level, n_frames = self.items[i % len(self.items)]
        src_sid = sid.split("x")[0] if "x" in sid else sid
        tab = self._measures(src_sid)
        if tab is None:  # unusable grid: deterministic redirect to another item
            return self._getitem_slot((i + 9973) % len(self.items))
        db, nbm, okm = tab
        mel, notes = self._load(sid)
        times = notes[f"{course}_t"]
        classes = notes[f"{course}_c"]

        rng0 = random.Random(i * 31337 + (self._rng.randint(0, 1 << 30) if self.train else 0))
        rng = random.Random(i * 7919 + (self._rng.randint(0, 1 << 30) if self.train else 0))
        r = 1.0
        if self.tempo_aug and rng0.random() < 0.5:
            r = rng0.uniform(0.9, 1.111)

        # start measure: importance-weighted among valid measures
        cand = np.nonzero(okm)[0]
        cand = cand[db[cand] * FPS < max(n_frames - 1, 1)]
        if len(cand) == 0:
            return self._getitem_slot((i + 9973) % len(self.items))
        if self.train:
            if self.importance:
                lo = np.searchsorted(times, db[cand])
                hi = np.searchsorted(times, db[cand] + WINDOW / FPS * r)
                w = (hi - lo).astype(np.float64) + 3.0
                j = int(rng.choices(cand.tolist(), weights=w.tolist())[0])
            else:
                j = int(rng.choice(cand.tolist()))
        else:
            j = int(cand[(i * 2654435761) % len(cand)])

        # extend K whole measures while valid, inside audio, and within both
        # the frame window and the 72-beat token budget
        K = 0
        nbeats = 0
        while (j + K < len(okm) and okm[j + K]
               and nbeats + nbm[j + K] <= BEATS_MAX
               and (db[j + K + 1] - db[j]) * FPS / r <= WINDOW
               and db[j + K + 1] * FPS <= n_frames + FPS):
            nbeats += nbm[j + K]
            K += 1
        if K == 0:
            return self._getitem_slot((i + 9973) % len(self.items))
        t0, t_end = db[j], db[j + K]
        boff = np.concatenate([[0], np.cumsum(nbm[j : j + K])])  # beat offsets

        def to_slot(t, m):
            a, b = db[j + m], db[j + m + 1]
            s = int(np.floor(SLOTS_PER_BEAT * nbm[j + m] * (t - a) / (b - a) + 0.5))
            return min(boff[m] * SLOTS_PER_BEAT + s, WINDOW - 1)

        # notes -> exact lattice indices; keep frame positions for the aux head
        pairs, aux_frames = [], []
        seen = set()
        for m in range(K):
            a, b = db[j + m], db[j + m + 1]
            sel = (times >= a - 1e-4) & (times < b - 1e-4)
            for t, c in zip(times[sel], classes[sel]):
                g = to_slot(t, m)
                if (g, int(c)) not in seen:
                    seen.add((g, int(c)))
                    pairs.append((g, int(c)))
                    aux_frames.append((t - t0) / r * FPS)
        pairs_sorted = sorted(range(len(pairs)), key=lambda k: pairs[k][0])
        pairs = [pairs[k] for k in pairs_sorted]
        aux_frames = [aux_frames[k] for k in pairs_sorted]

        sib = None
        if self.sibling:
            order = ["easy", "normal", "hard", "oni", "ura"]
            sib_pairs = []
            if not (self.train and rng0.random() < 0.2):
                for c2 in reversed(order[: order.index(course)]):
                    if f"{c2}_t" in notes:
                        t2, c2c = notes[f"{c2}_t"], notes[f"{c2}_c"]
                        ev = []
                        for m in range(K):
                            a, b = db[j + m], db[j + m + 1]
                            s2 = (t2 >= a - 1e-4) & (t2 < b - 1e-4) & (c2c <= 3)
                            for t, c in zip(t2[s2], c2c[s2]):
                                ev.append((to_slot(t, m), int(c)))
                        if len(ev) > 12:
                            idx2 = np.linspace(0, len(ev) - 1, 12).astype(int)
                            ev = [ev[k] for k in idx2]
                        sib_pairs = ev
                        break
            sib = sib_pairs

        style = None
        if self.styles is not None:
            style = self.styles.get(f"{sid}:{course}", -1)

        plan_slice = None
        if self.plans is not None:
            blocks = self.plans.get(f"{sid}:{course}") or []
            plan_slice = [(b[2], b[3]) for b in blocks if b[1] > t0 and b[0] < t_end]

        seq, prefix_len = encode_window(
            VOCAB, course, level, pairs, cond_drop=self.cond_drop, rng=rng,
            sib_pairs=sib, style=style, plan_slice=plan_slice,
            mode=("slot" if self.dual else None),
        )
        if len(seq) > MAX_TGT:
            seq = seq[: MAX_TGT - 1] + [VOCAB.eos]
            if VOCAB.is_time(seq[-2]):
                seq = seq[:-2] + [VOCAB.eos]

        # mel crop [t0, t_end) -> stretch by r -> pad to WINDOW
        f0 = int(round(t0 * FPS))
        src_len = max(1, int(round((t_end - t0) * FPS)))
        x = mel[:, f0 : f0 + src_len].astype(np.float32)
        if x.shape[1] < src_len:
            x = np.pad(x, ((0, 0), (0, src_len - x.shape[1])), constant_values=np.log(1e-5))
        x = torch.from_numpy(x)
        tgt_len = min(WINDOW, max(1, int(round(src_len / r))))
        if tgt_len != src_len:
            x = torch.nn.functional.interpolate(
                x[None], size=tgt_len, mode="linear", align_corners=False)[0]
        if x.shape[1] < WINDOW:
            x = torch.nn.functional.pad(x, (0, WINDOW - x.shape[1]),
                                        value=float(np.log(1e-5)))
        if self.tempo_aug and rng0.random() < 0.3:
            k = rng0.randint(-4, 4)
            if k:
                x = torch.roll(x, k, dims=0)
                if k > 0:
                    x[:k] = x.min()
                else:
                    x[k:] = x.min()
        if self.spec_augment:
            for _ in range(2):
                w = rng.randint(0, 12)
                fq = rng.randint(0, x.shape[0] - w) if w else 0
                if w:
                    x[fq : fq + w] = x.mean()
            for _ in range(2):
                w = rng.randint(0, 24)
                fq = rng.randint(0, x.shape[1] - w) if w else 0
                if w:
                    x[:, fq : fq + w] = x.mean()
            x = x + (rng.random() * 1.38 - 0.69)
        # phase channels AFTER augmentation (the grid is clean metadata):
        # ch0 = measure phase 0..1, ch1 = beat phase 0..1; -1 marks padding
        ph = torch.full((2, WINDOW), -1.0)
        edges = (db[j : j + K + 1] - t0) / r * FPS
        for m in range(K):
            a, b = edges[m], edges[m + 1]
            i0, i1 = int(np.ceil(a - 1e-6)), min(int(np.ceil(b - 1e-6)), WINDOW)
            if i1 <= i0:
                continue
            frac = (torch.arange(i0, i1, dtype=torch.float32) - a) / max(b - a, 1e-6)
            ph[0, i0:i1] = frac
            ph[1, i0:i1] = (frac * float(nbm[j + m])) % 1.0
        x = torch.cat([x, ph], dim=0)

        aux_t = torch.zeros(0)
        if self.aux:
            aux_t = torch.zeros(WINDOW // 4)
            for f, (g, c) in zip(aux_frames, pairs):
                if NOTE_CLASSES[c] != "end":
                    aux_t[min(max(int(f), 0) // 4, WINDOW // 4 - 1)] = 1.0
            aux_t = torch.max(aux_t, 0.5 * torch.roll(aux_t, 1))
            aux_t = torch.max(aux_t, 0.5 * torch.roll(aux_t, -1))

        beat_t = torch.zeros(0)
        if self.beat_head:
            # measured LEAK (v15a): slot windows carry grid-phase INPUT
            # channels, so a beat head trained here just copies the channels
            # and collapses at inference (p50 723 ms). Mask beat supervision
            # (-1) on slot windows; the head learns from gridless time-mode
            # windows in dual training instead.
            res = WINDOW if self.beat_hires else WINDOW // 4
            beat_t = torch.full((res, 2), -1.0)

        in_t = torch.tensor(seq, dtype=torch.long)
        return (x, torch.tensor(seq, dtype=torch.long), prefix_len, aux_t,
                torch.zeros(0), 0, beat_t, in_t)

    def __getitem__(self, i):
        if self.dual:
            sid = self.items[i % len(self.items)][0]
            src = sid.split("x")[0] if "x" in sid else sid
            rngm = random.Random(i * 104729 + (self._rng.randint(0, 1 << 30) if self.train else 0))
            if self._measures(src) is not None and (not self.train or rngm.random() < self.dual_slot_p):
                return self._getitem_slot(i)
            # else: fall through to the time path (mode token + blank grid channels)
        elif self.slot:
            return self._getitem_slot(i)
        sid, course, level, n_frames = self.items[i % len(self.items)]
        mel, notes = self._load(sid)
        times = notes[f"{course}_t"]
        classes = notes[f"{course}_c"]

        # rhythm-preserving tempo augmentation: crop a longer/shorter source
        # window and resample it to WINDOW (mel-domain time stretch); note
        # times scale by the same factor, so all rhythmic relations survive.
        rng0 = random.Random(i * 31337 + (self._rng.randint(0, 1 << 30) if self.train else 0))
        r = 1.0
        if self.tempo_aug and rng0.random() < 0.5:
            r = rng0.uniform(0.9, 1.111)
        src_len = int(round(WINDOW * r))

        max_start = max(0, n_frames - src_len)
        rng = random.Random(i * 7919 + (self._rng.randint(0, 1 << 30) if self.train else 0))
        if self.train and max_start > 0:
            if self.importance:
                # importance sampling: bias window starts toward note-dense /
                # informative regions (uniform floor keeps sparse regions seen)
                cands = np.arange(0, max_start + 1, int(FPS))  # 1s stride
                lo = np.searchsorted(times, cands / FPS)
                hi = np.searchsorted(times, cands / FPS + WINDOW / FPS)
                w = (hi - lo).astype(np.float64) + 3.0
                start = int(rng.choices(cands.tolist(), weights=w.tolist())[0])
                start = min(max_start, start + rng.randint(0, int(FPS)))  # sub-second jitter
            else:
                start = rng.randint(0, max_start)
        else:  # deterministic pseudo-random window per item (not always the intro)
            start = (i * 2654435761) % (max_start + 1) if max_start > 0 else 0

        t0 = start / FPS
        t1 = (start + src_len) / FPS
        sel = (times >= t0) & (times < t1)
        frames = np.round((times[sel] - t0) * FPS / r).astype(np.int64)
        frames = np.clip(frames, 0, WINDOW - 1)
        pairs = list(zip(frames.tolist(), classes[sel].tolist()))

        sib = None
        if self.sibling:  # skeleton hint from the nearest easier course
            order = ["easy", "normal", "hard", "oni", "ura"]
            sib_pairs = []
            if not (self.train and rng0.random() < 0.2):  # p=0.2: train without hint
                for c2 in reversed(order[: order.index(course)]):
                    if f"{c2}_t" in notes:
                        t2, c2c = notes[f"{c2}_t"], notes[f"{c2}_c"]
                        s2 = (t2 >= t0) & (t2 < t1) & (c2c <= 3)  # hits only
                        f2 = np.clip(np.round((t2[s2] - t0) * FPS / r), 0, WINDOW - 1)
                        ev = list(zip(f2.astype(np.int64).tolist(), c2c[s2].tolist()))
                        if len(ev) > 12:  # even subsample to 12 slots
                            idx2 = np.linspace(0, len(ev) - 1, 12).astype(int)
                            ev = [ev[k] for k in idx2]
                        sib_pairs = ev
                        break
            sib = sib_pairs

        ctx = None
        if self.use_ctx:
            from .generate import CTX_LEN, HIT_CLASSES

            hit_ids = [NOTE_CLASSES.index(c) for c in HIT_CLASSES]
            prev = (times >= t0 - 3.0) & (times < t0)
            tail = [int(c) for c in classes[prev] if int(c) in hit_ids][-CTX_LEN:]
            if self.train and rng.random() < 0.1:
                tail = []  # simulate missing context (first window)
            elif self.train:
                tail = [c if rng.random() > 0.1 else rng.choice(hit_ids) for c in tail]
            ctx = [-1] * (CTX_LEN - len(tail)) + tail

        style = None
        if self.styles is not None:
            style = self.styles.get(f"{sid}:{course}", -1)

        sync_band = None
        beats_w = db_w = None
        if self.sync_token or self.beat_head:
            src_sid = sid.split("x")[0] if "x" in sid else sid
            bts, dbs = self._beats(src_sid)
            if bts is not None:
                # window-local, tempo-scaled beat grid
                bsel = (bts >= t0) & (bts < t1)
                beats_w = (bts[bsel] - t0) / r
                dsel = (dbs >= t0) & (dbs < t1)
                db_w = (dbs[dsel] - t0) / r
        if self.sync_token:
            sync_band = lhl_band(
                np.array([f / FPS for f, c in pairs if NOTE_CLASSES[c] != "end"]),
                beats_w, db_w)

        plan_slice = None
        if self.plans is not None:
            blocks = self.plans.get(f"{sid}:{course}") or []
            # blocks overlapping this window; times already in source timescale,
            # window is [t0, t1) in the same scale (r rescales note frames only,
            # block membership is decided in source time)
            plan_slice = [(b[2], b[3]) for b in blocks if b[1] > t0 and b[0] < t1]

        cplx = None
        if self.complexity:
            from .vocab import complexity_band
            cplx = complexity_band(pairs)

        seq, prefix_len = encode_window(
            VOCAB, course, level, pairs, cond_drop=self.cond_drop, rng=rng,
            ctx_types=ctx, sib_pairs=sib, style=style, sync_band=sync_band,
            plan_slice=plan_slice, complexity=cplx,
            mode=("time" if self.dual else None),
        )
        in_seq = None
        if self.mask_infill > 0 and rng.random() < self.mask_infill:
            # skeleton->color curriculum: decoder INPUT sees MASK where note
            # types were; gold targets keep the true types
            note_ids = set(VOCAB.note.values())
            in_seq = [VOCAB.mask if (i >= prefix_len and tok in note_ids) else tok
                      for i, tok in enumerate(seq)]
        if len(seq) > MAX_TGT:  # truncate overly dense windows, keep EOS
            seq = seq[: MAX_TGT - 1] + [VOCAB.eos]
            if VOCAB.is_time(seq[-2]):  # do not end on a dangling TIME token
                seq = seq[:-2] + [VOCAB.eos]

        x = mel[:, start : start + src_len].astype(np.float32)
        if x.shape[1] < src_len:
            x = np.pad(x, ((0, 0), (0, src_len - x.shape[1])), constant_values=np.log(1e-5))
        x = torch.from_numpy(x)
        if src_len != WINDOW:  # mel-domain time stretch to the fixed window size
            x = torch.nn.functional.interpolate(
                x[None], size=WINDOW, mode="linear", align_corners=False
            )[0]
        if self.tempo_aug and rng0.random() < 0.3:  # mild pitch shift (mel-bin roll)
            k = rng0.randint(-4, 4)
            if k:
                x = torch.roll(x, k, dims=0)
                if k > 0:
                    x[:k] = x.min()
                else:
                    x[k:] = x.min()
        if self.spec_augment:
            for _ in range(2):  # freq masks
                w = rng.randint(0, 12)
                f0 = rng.randint(0, x.shape[0] - w) if w else 0
                if w:
                    x[f0 : f0 + w] = x.mean()
            for _ in range(2):  # time masks
                w = rng.randint(0, 24)
                f0 = rng.randint(0, x.shape[1] - w) if w else 0
                if w:
                    x[:, f0 : f0 + w] = x.mean()
            x = x + (rng.random() * 1.38 - 0.69)  # gain +-6 dB in log space
        if self.dual:  # time mode carries no grid: phase channels = -1
            x = torch.cat([x, torch.full((2, WINDOW), -1.0)], dim=0)

        aux_t = torch.zeros(0)
        if self.aux:  # onset heatmap at encoder resolution (WINDOW // 4)
            aux_t = torch.zeros(WINDOW // 4)
            for f, c in pairs:
                if NOTE_CLASSES[c] != "end":
                    aux_t[min(f // 4, WINDOW // 4 - 1)] = 1.0
            aux_t = torch.max(aux_t, 0.5 * torch.roll(aux_t, 1))
            aux_t = torch.max(aux_t, 0.5 * torch.roll(aux_t, -1))

        beat_t = torch.zeros(0)
        if self.beat_head:
            res = WINDOW if self.beat_hires else WINDOW // 4
            div = 1 if self.beat_hires else 4
            beat_t = torch.zeros(res, 2)
            for arr, ch in ((beats_w, 0), (db_w, 1)):
                if arr is None:
                    continue
                for bt in arr:
                    k = int(bt * FPS) // div
                    if 0 <= k < res:
                        beat_t[k, ch] = 1.0
            for ch in range(2):
                col = beat_t[:, ch]
                beat_t[:, ch] = torch.max(col, 0.5 * torch.roll(col, 1))
                beat_t[:, ch] = torch.max(beat_t[:, ch], 0.5 * torch.roll(col, -1))

        gsum = torch.zeros(0)
        posb = 0
        if self.global_ctx:
            gsum = torch.from_numpy(song_summary(np.asarray(mel)))
            posb = min(15, int(16 * start / max(n_frames, 1)))

        in_t = torch.tensor(in_seq if in_seq is not None else seq, dtype=torch.long)
        return (x, torch.tensor(seq, dtype=torch.long), prefix_len, aux_t, gsum, posb,
                beat_t, in_t)


def lhl_band(note_times, beats, downbeats, band_width=0.6):
    """Longuet-Higgins & Lee (1984) style syncopation, simplified to the
    eighth-note grid, normalized per bar, bucketed into 6 bands."""
    if beats is None or len(beats) < 4 or len(note_times) < 2:
        return -1
    beat_len = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    grid = []  # (time, metrical level)
    dbset = set(np.round(downbeats, 3)) if downbeats is not None else set()
    for b in beats:
        lvl = 3 if round(float(b), 3) in dbset else 2
        grid.append((float(b), lvl))
        grid.append((float(b) + beat_len / 2, 1))  # eighth positions
    grid.sort()
    tol = 0.04
    score = 0.0
    for i, (gt_pos, lvl) in enumerate(grid[:-1]):
        has_note = np.any(np.abs(note_times - gt_pos) < tol)
        nxt_pos, nxt_lvl = grid[i + 1]
        nxt_note = np.any(np.abs(note_times - nxt_pos) < tol)
        if has_note and not nxt_note and nxt_lvl > lvl:
            score += nxt_lvl - lvl
    n_bars = max(1.0, len([g for g in grid if g[1] == 3]))
    return min(5, int(score / n_bars / band_width))


def collate(batch):
    xs, seqs, plens, auxs, gsums, posbs, beats, inseqs = zip(*batch)
    x = torch.stack(xs)
    maxlen = max(len(s) for s in seqs)
    tgt = torch.full((len(seqs), maxlen), VOCAB.pad, dtype=torch.long)
    loss_mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for i, (s, pl) in enumerate(zip(seqs, plens)):
        tgt[i, : len(s)] = s
        loss_mask[i, pl : len(s)] = True  # loss on events + EOS only
    aux = torch.stack(auxs) if auxs[0].numel() else None
    gsum = torch.stack(gsums) if gsums[0].numel() else None
    posb = torch.tensor(posbs, dtype=torch.long)
    beat = torch.stack(beats) if beats[0].numel() else None
    in_tgt = torch.full((len(inseqs), maxlen), VOCAB.pad, dtype=torch.long)
    for i, sq in enumerate(inseqs):
        in_tgt[i, : len(sq)] = sq
    return x, tgt, loss_mask, aux, gsum, posb, beat, in_tgt


def load_split_ids(cache_dir, val_songs=40, seed=42, use_aug=False):
    """Song-level split: train / val (carved from train) / test (provided).

    Augmented variants (speed-aug entries with "aug": true) are never used for
    val, and variants whose source song is in val are excluded from train
    (no leakage). The base split is stable regardless of augmentation.
    """
    with open(os.path.join(cache_dir, "index.json")) as f:
        index = json.load(f)
    base_train = sorted(e["id"] for e in index
                        if e["split"] == "train" and not e.get("aug"))
    test_ids = sorted(e["id"] for e in index if e["split"] == "test")
    rng = random.Random(seed)
    rng.shuffle(base_train)
    val_ids = base_train[:val_songs]
    train_ids = base_train[val_songs:]
    if use_aug:
        val_set = set(val_ids)
        train_ids = train_ids + sorted(
            e["id"] for e in index
            if e.get("aug") and e.get("src") not in val_set)
    return train_ids, val_ids, test_ids
