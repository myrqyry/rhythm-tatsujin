"""Encoder-decoder Transformer: log-mel windows -> chart token sequences."""

import math

import torch
import torch.nn as nn

from .vocab import N_MELS, VOCAB, WINDOW


def sinusoidal(length, dim):
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, dtype=torch.float32)[None]
    angle = pos / torch.pow(10000.0, 2 * i / dim)
    emb = torch.zeros(length, dim)
    emb[:, 0::2] = torch.sin(angle)
    emb[:, 1::2] = torch.cos(angle)
    return emb


class BeatHiRes(nn.Module):
    """Frame-resolution beat/downbeat head: encoder states (T/4) are
    upsampled 4x so supervision and peaks live on the 11.6 ms frame grid
    instead of 46 ms encoder bins (4x finer timing per peak)."""

    def __init__(self, d):
        super().__init__()
        self.up = nn.ConvTranspose1d(d, d // 2, 4, stride=4)
        self.out = nn.Conv1d(d // 2, 2, 3, padding=1)

    def forward(self, mem):  # (B, L, d) -> (B, 4L, 2)
        h = nn.functional.gelu(self.up(mem.transpose(1, 2)))
        return self.out(h).transpose(1, 2)


def _parse_pattern(spec, n):
    """Layer-sharing schedule. spec is either an int (number of UNIQUE
    physical layers, cycled to length n) or a comma string like "E0,E1,E0,E1"
    (labels; each distinct label is one physical layer). Returns a list of n
    physical-layer indices, e.g. [0,1,0,1] for 2-unique/4-physical-depth."""
    if spec is None:
        return list(range(n))  # no sharing: every layer unique
    if isinstance(spec, int):
        return [k % spec for k in range(n)]
    labels = [s.strip() for s in str(spec).split(",") if s.strip()]
    assert len(labels) == n, f"pattern {spec} has {len(labels)} labels, need {n}"
    order = {}
    out = []
    for lb in labels:
        if lb not in order:
            order[lb] = len(order)
        out.append(order[lb])
    return out


class Adapter(nn.Module):
    """Rank-r residual adapter applied after a (possibly reused) transformer
    block: x + gate * up(gelu(down(x))). gate init 0 so the whole stack starts
    identical to hard weight sharing; the adapter only differentiates a reused
    physical layer as training moves the gate off zero."""

    def __init__(self, d, rank):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gate * self.up(nn.functional.gelu(self.down(x)))


class SharedEncoder(nn.Module):
    """Encoder whose physical layers are reused according to a sharing pattern.
    Each POSITION in the depth keeps its own (a) rank-r adapter and (b) final
    LayerNorm — so ΔW starts at 0 (hard sharing) but every position can drift.
    With pattern=None this is exactly n unique layers + a norm, matching the
    stock nn.TransformerEncoder param count (adapters/pos-norms are opt-in).

    v1.7 depth-diversity options (all default-off, init == v1.6 behaviour):
      depth_emb: per-depth learned d-vector ADDED to the block input at each
        reuse (params = depth x d, zero-init so init == hard sharing).
      adapter_rank_ffn > 0: SPLIT adapters — instead of one post-block adapter,
        each depth owns a rank-`adapter_rank` adapter on the attention sublayer
        output and a rank-`adapter_rank_ffn` adapter on the FFN sublayer output
        (patterning lives in the FFN). Gate init 0 keeps init == hard sharing.
    """

    def __init__(self, d, nhead, ffn, dropout, pattern, adapter_rank=0,
                 unique_layernorm=False, adapter_rank_ffn=0, depth_emb=False):
        super().__init__()
        # `pattern` is the resolved list of physical-layer indices per depth
        n_phys = max(pattern) + 1
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nhead, ffn, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_phys)])
        self.plan = pattern
        self.split = adapter_rank_ffn > 0
        if self.split:  # v1.7: per-sublayer adapters (attention rank-a, FFN rank-f)
            self.attn_adapters = nn.ModuleList([
                Adapter(d, adapter_rank) if adapter_rank else nn.Identity()
                for _ in pattern])
            self.ffn_adapters = nn.ModuleList([
                Adapter(d, adapter_rank_ffn) for _ in pattern])
        else:  # v1.6: one residual adapter after the whole block
            self.adapters = nn.ModuleList([
                Adapter(d, adapter_rank) if adapter_rank else nn.Identity()
                for _ in pattern])
        self.depth_emb = (nn.Parameter(torch.zeros(len(pattern), d))
                          if depth_emb else None)
        self.pos_norms = nn.ModuleList([
            nn.LayerNorm(d) if unique_layernorm else nn.Identity()
            for _ in pattern])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, src_key_padding_mask=None):
        for k, phys in enumerate(self.plan):
            if self.depth_emb is not None:
                x = x + self.depth_emb[k]
            layer = self.layers[phys]
            if self.split:
                # norm_first decomposition of nn.TransformerEncoderLayer with
                # per-depth adapters applied to each sublayer OUTPUT (Houlsby
                # placement); at gate=0 this is bit-equal to the stock layer
                x = x + self.attn_adapters[k](
                    layer._sa_block(layer.norm1(x), None, src_key_padding_mask))
                x = x + self.ffn_adapters[k](layer._ff_block(layer.norm2(x)))
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
                x = self.adapters[k](x)
            x = self.pos_norms[k](x)
        return self.norm(x)


class SharedDecoder(nn.Module):
    """Decoder counterpart of SharedEncoder (see that docstring). Split mode
    puts rank-`adapter_rank` adapters on BOTH attention sublayers (self and
    cross) and rank-`adapter_rank_ffn` on the FFN sublayer."""

    def __init__(self, d, nhead, ffn, dropout, pattern, adapter_rank=0,
                 unique_layernorm=False, adapter_rank_ffn=0, depth_emb=False):
        super().__init__()
        n_phys = max(pattern) + 1
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, nhead, ffn, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_phys)])
        self.plan = pattern
        self.split = adapter_rank_ffn > 0
        if self.split:
            self.sa_adapters = nn.ModuleList([
                Adapter(d, adapter_rank) if adapter_rank else nn.Identity()
                for _ in pattern])
            self.ca_adapters = nn.ModuleList([
                Adapter(d, adapter_rank) if adapter_rank else nn.Identity()
                for _ in pattern])
            self.ffn_adapters = nn.ModuleList([
                Adapter(d, adapter_rank_ffn) for _ in pattern])
        else:
            self.adapters = nn.ModuleList([
                Adapter(d, adapter_rank) if adapter_rank else nn.Identity()
                for _ in pattern])
        self.depth_emb = (nn.Parameter(torch.zeros(len(pattern), d))
                          if depth_emb else None)
        self.pos_norms = nn.ModuleList([
            nn.LayerNorm(d) if unique_layernorm else nn.Identity()
            for _ in pattern])
        self.norm = nn.LayerNorm(d)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                tgt_is_causal=False):
        x = tgt
        for k, phys in enumerate(self.plan):
            if self.depth_emb is not None:
                x = x + self.depth_emb[k]
            layer = self.layers[phys]
            if self.split:
                x = x + self.sa_adapters[k](layer._sa_block(
                    layer.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal))
                x = x + self.ca_adapters[k](layer._mha_block(
                    layer.norm2(x), memory, None, None))
                x = x + self.ffn_adapters[k](layer._ff_block(layer.norm3(x)))
            else:
                x = layer(x, memory, tgt_mask=tgt_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          tgt_is_causal=tgt_is_causal)
                x = self.adapters[k](x)
            x = self.pos_norms[k](x)
        return self.norm(x)


class ChartModel(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        enc_layers=6,
        dec_layers=6,
        ffn=2048,
        dropout=0.1,
        vocab_size=None,
        max_tgt=None,
        aux=False,
        global_ctx=False,
        func_time=False,
        ptr=False,
        beat=False,
        in_ch=None,
        emb_factor=None,
        clean_phase=False,
        enc_share=None,
        dec_share=None,
        adapter_rank=0,
        unique_layernorm=False,
        adapter_rank_ffn=0,
        depth_emb=False,
        unshare_last_dec=False,
    ):
        super().__init__()
        from .vocab import MAX_TGT

        vocab_size = vocab_size or VOCAB.size
        max_tgt = max_tgt or MAX_TGT
        self.d_model = d_model
        # clean_phase (v1.6): the encoder never sees the 2 grid-phase channels
        # (in_ch stays N_MELS), and phase is injected into the DECODER memory
        # only, via a small projection. This removes the condition leak that
        # forced beat supervision to be masked on slot windows in v1.5.
        self.clean_phase = clean_phase
        self.in_ch = in_ch or N_MELS  # legacy slot mode appends 2 grid-phase channels
        front_ch = N_MELS if clean_phase else self.in_ch

        # conv frontend: (B, front_ch, T) -> (B, T/4, d)
        self.frontend = nn.Sequential(
            nn.Conv1d(front_ch, d_model, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            nn.GELU(),
        )
        enc_len = WINDOW // 4
        self.register_buffer("enc_pos", sinusoidal(enc_len, d_model), persistent=False)
        self.register_buffer("dec_pos", sinusoidal(max_tgt, d_model), persistent=False)

        # phase injection (clean_phase only): 2 grid-phase channels at frame
        # rate -> encoder rate (T/4) -> additive conditioning on the memory the
        # DECODER reads. The beat head reads the pre-injection (clean) memory.
        # phase_proj input = [measure_phase, beat_phase, valid_flag]. valid=0 on
        # time-mode windows (their raw phase is -1), giving the decoder a clean
        # zero-phase conditioning there; slot windows pass valid=1 + real phase.
        if clean_phase:
            self.phase_proj = nn.Sequential(
                nn.Conv1d(3, d_model, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            )
        else:
            self.phase_proj = None

        # sharing schedule (v1.6-small/tiny): reuse physical layers per pattern.
        # enc_share/dec_share: int (#unique layers) or "E0,E1,E0,E1" label string.
        self.enc_share = enc_share
        self.dec_share = dec_share
        if (enc_share is None and dec_share is None and not adapter_rank
                and not adapter_rank_ffn and not depth_emb and not unshare_last_dec):
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, ffn, dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, enc_layers,
                                                 nn.LayerNorm(d_model))
            dec_layer = nn.TransformerDecoderLayer(
                d_model, nhead, ffn, dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(dec_layer, dec_layers,
                                                 nn.LayerNorm(d_model))
        else:
            enc_plan = _parse_pattern(enc_share, enc_layers)
            dec_plan = _parse_pattern(dec_share, dec_layers)
            # v1.7: un-share the LAST decoder layer (pattern-shaping layer in
            # front of the generation head gets its own physical parameters);
            # no-op if that depth is already unique.
            if unshare_last_dec and dec_plan.count(dec_plan[-1]) > 1:
                dec_plan = dec_plan[:-1] + [max(dec_plan) + 1]
            self.encoder = SharedEncoder(d_model, nhead, ffn, dropout, enc_plan,
                                         adapter_rank, unique_layernorm,
                                         adapter_rank_ffn, depth_emb)
            self.decoder = SharedDecoder(d_model, nhead, ffn, dropout, dec_plan,
                                         adapter_rank, unique_layernorm,
                                         adapter_rank_ffn, depth_emb)

        # ALBERT-style factorized embeddings (v1.5): vocab->E->d. Position
        # tokens dominate the table (1728/1810 rows), so E=64 cuts the
        # embedding block ~3.5x; logits stay tied via the projected matrix.
        self.emb_factor = emb_factor
        if emb_factor:
            self.tok_emb = nn.Embedding(vocab_size, emb_factor, padding_idx=VOCAB.pad)
            nn.init.normal_(self.tok_emb.weight, std=0.02)
            self.emb_proj = nn.Linear(emb_factor, d_model, bias=False)
            nn.init.normal_(self.emb_proj.weight, std=1.0 / (emb_factor ** 0.5) * 0.16)
            self.out = None
        else:
            self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB.pad)
            nn.init.normal_(self.tok_emb.weight, std=0.02)  # keep tied logits well-scaled
            self.emb_proj = None
            self.out = nn.Linear(d_model, vocab_size, bias=False)
            self.out.weight = self.tok_emb.weight  # weight tying
        self.dropout = nn.Dropout(dropout)
        # auxiliary per-position onset-heatmap head on the encoder (v2)
        self.aux = nn.Linear(d_model, 1) if aux else None
        # pointer alignment head: each generated note must point to its audio
        # frame (differentiable provenance; explainability that trains alignment)
        self.ptr = nn.Linear(d_model, d_model) if ptr else None
        # beat/downbeat head: explicit metrical percept on the encoder.
        # beat="hires" builds the frame-resolution head (11.6 ms bins)
        self.beat = (BeatHiRes(d_model) if beat == "hires"
                     else (nn.Linear(d_model, 2) if beat else None))
        # song-level context: coarse whole-song summary + window position (v4).
        # Lets the model see beyond the window, so intentional gaps (a breath
        # before a strong section) are informed decisions, not failures.
        # v2 fix (after the v4 negative result): summary chunks get DEDICATED
        # positional embeddings instead of reusing window positions.
        if global_ctx:
            self.gsum_proj = nn.Linear(N_MELS, d_model)
            self.seg_emb = nn.Parameter(torch.zeros(2, d_model))
            self.pos_emb = nn.Embedding(16, d_model)  # window position bucket
            self.gpos = nn.Parameter(torch.randn(128, d_model) * 0.02)
        else:
            self.gsum_proj = None
        # functional time embeddings: the 1728 TIME tokens share a sinusoidal
        # basis + small projection instead of free embeddings (fewer params,
        # neighbouring times get similar representations)
        if func_time:
            self.register_buffer("time_basis", sinusoidal(WINDOW, 64), persistent=False)
            self.time_proj = nn.Linear(64, d_model)
            # match the 0.02-std scale of tok_emb rows (basis row norm ~ sqrt(32))
            nn.init.normal_(self.time_proj.weight, std=0.004)
            nn.init.zeros_(self.time_proj.bias)
        else:
            self.time_proj = None

    def encode(self, mel, gsum=None, pos_bucket=None):
        # mel: (B, C, T). clean_phase: C may be N_MELS (audio only) or N_MELS+2
        # (audio + 2 grid-phase channels); only the audio channels reach the
        # encoder, so the returned memory is CLEAN (phase-free) and safe for the
        # beat head. gsum: (B, G, n_mels) whole-song summary chunks;
        # pos_bucket: (B,) window-position bucket in [0, 16).
        audio = mel[:, :N_MELS] if self.clean_phase else mel
        h = self.frontend(audio).transpose(1, 2)  # (B, T/4, d)
        h = h + self.enc_pos[: h.shape[1]]
        if self.gsum_proj is not None and gsum is not None:
            g = self.gsum_proj(gsum) + self.seg_emb[1] + self.gpos[: gsum.shape[1]]
            h = h + self.seg_emb[0]
            p = self.pos_emb(pos_bucket).unsqueeze(1)  # (B, 1, d)
            h = torch.cat([p, g, h], dim=1)
        return self.encoder(self.dropout(h))

    def phase_mem(self, memory, mel):
        """Phase-conditioned copy of `memory` for the DECODER to cross-attend.
        Adds a projection of the grid-phase channels to the audio-frame slice
        of memory; the beat head keeps reading the clean `memory`. No-op unless
        clean_phase is on and phase channels are present. Time-mode windows
        (raw phase = -1) get valid=0 and zeroed phase -> ~zero conditioning."""
        nch = N_MELS
        if not self.clean_phase or mel.shape[1] < nch + 2:
            return memory
        ph = mel[:, nch:nch + 2]  # (B, 2, T) in [0,1); -1 marks no-grid frames
        valid = (ph[:, :1] >= 0).float()  # 1 where a grid exists, else 0
        ph_in = torch.cat([ph.clamp(min=0.0), valid], dim=1)  # (B, 3, T)
        p = self.phase_proj(ph_in).transpose(1, 2)  # (B, T/4, d)
        L = p.shape[1]
        out = memory.clone()
        out[:, -L:] = out[:, -L:] + p  # audio frames are the last L memory slots
        return out

    def emb_matrix(self):
        """Full embedding matrix; factorized and/or functional rows resolved."""
        E = self.tok_emb.weight
        if self.emb_proj is not None:
            E = self.emb_proj(E)
        if self.time_proj is not None:
            t = self.time_proj(self.time_basis)  # (WINDOW, d)
            E = torch.cat([E[: VOCAB.time0], t, E[VOCAB.time0 + WINDOW :]], dim=0)
        return E

    def decode(self, tgt_in, memory):
        # tgt_in: (B, L) token ids
        L = tgt_in.shape[1]
        E = self.emb_matrix()
        h = nn.functional.embedding(tgt_in, E) * math.sqrt(self.d_model) + self.dec_pos[:L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt_in.device)
        pad_mask = tgt_in == VOCAB.pad
        h = self.decoder(
            self.dropout(h),
            memory,
            tgt_mask=mask,
            tgt_key_padding_mask=pad_mask,
            tgt_is_causal=True,
        )
        return h @ E.T  # tied output projection (functional rows included)

    def enable_ptr(self):
        self.ptr = nn.Linear(self.d_model, self.d_model)

    def enable_beat(self, hires=False):
        self.beat = BeatHiRes(self.d_model) if hires else nn.Linear(self.d_model, 2)

    def forward(self, mel, tgt, return_aux=False, gsum=None, pos_bucket=None,
                return_extras=False, in_tgt=None):
        # in_tgt: optional corrupted decoder input (e.g. MASKed types for the
        # skeleton->color infill curriculum); gold targets stay = tgt
        memory = self.encode(mel, gsum=gsum, pos_bucket=pos_bucket)  # CLEAN
        dec_mem = self.phase_mem(memory, mel)  # phase-conditioned for the decoder
        dec_in = (in_tgt if in_tgt is not None else tgt)[:, :-1]
        L = dec_in.shape[1]
        E = self.emb_matrix()
        import math as _m
        h = nn.functional.embedding(dec_in, E) * _m.sqrt(self.d_model) + self.dec_pos[:L]
        dm = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        h = self.decoder(self.dropout(h), dec_mem, tgt_mask=dm,
                         tgt_key_padding_mask=dec_in == VOCAB.pad, tgt_is_causal=True)
        logits = h @ E.T
        if not (return_aux or return_extras):
            return logits
        outs = [logits]
        # aux/ptr/beat heads read the CLEAN memory (no phase leak)
        aux_mem = memory[:, -(WINDOW // 4):]
        outs.append(self.aux(aux_mem).squeeze(-1) if self.aux is not None else None)
        if return_extras:
            ptr_logits = None
            if self.ptr is not None:
                q = self.ptr(h)  # (B, L, d)
                ptr_logits = q @ aux_mem.transpose(1, 2) / _m.sqrt(self.d_model)
            beat_logits = self.beat(aux_mem) if self.beat is not None else None
            outs += [ptr_logits, beat_logits]
        return tuple(outs)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
