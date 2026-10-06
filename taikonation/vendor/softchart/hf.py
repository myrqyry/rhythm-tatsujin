"""Hugging Face Hub-compatible wrappers (safetensors + from_pretrained/push_to_hub).

Three model types, all <=16M params, MIT-licensed:
  SoftChartGenerator  audio log-mel  -> chart event tokens (the main model; pr12)
  SoftChartBeat       audio log-mel  -> beat/downbeat activations (barline anchor)
  SoftChartPlanner    block features -> per-block plan tokens (auto-planning)

Usage:
    from softchart.hf import SoftChartGenerator
    gen = SoftChartGenerator.from_pretrained("JacobLinCool/softchart-generator")
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .model import ChartModel, sinusoidal

_CARD = "See https://github.com/JacobLinCool/SoftChart — MIT licensed."


class SoftChartGenerator(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/JacobLinCool/SoftChart",
    license="mit",
    tags=["taiko", "rhythm-game", "chart-generation", "music", "audio-to-symbolic"],
):
    """Encoder-decoder chart generator. Config is stored as config.json and the
    full architecture (condition tokens, heads) is reconstructed on load."""

    def __init__(self, d_model=256, enc_layers=4, dec_layers=4, ffn=1024,
                 vocab_size=1802, aux=True, global_ctx=False, func_time=False,
                 ptr=False, beat=False, in_ch=None, emb_factor=None,
                 capabilities=None, nhead=8, clean_phase=False,
                 enc_share=None, dec_share=None, adapter_rank=0,
                 unique_layernorm=False, adapter_rank_ffn=0, depth_emb=False,
                 unshare_last_dec=False):
        super().__init__()
        # nhead and the v1.6/v1.7 sharing flags (enc_share/dec_share/adapter_rank/
        # unique_layernorm/adapter_rank_ffn/depth_emb/unshare_last_dec) and
        # clean_phase must be persisted in config.json: from_pretrained rebuilds
        # the architecture from these kwargs alone, before any weights are loaded.
        # Defaults reproduce the pre-v1.6 unshared architecture for older repos.
        self.net = ChartModel(
            d_model=d_model, nhead=nhead, enc_layers=enc_layers, dec_layers=dec_layers,
            ffn=ffn, vocab_size=vocab_size, aux=aux, global_ctx=global_ctx,
            func_time=func_time, ptr=ptr, beat=beat, in_ch=in_ch, emb_factor=emb_factor,
            clean_phase=clean_phase, enc_share=enc_share, dec_share=dec_share,
            adapter_rank=adapter_rank, unique_layernorm=unique_layernorm,
            adapter_rank_ffn=adapter_rank_ffn, depth_emb=depth_emb,
            unshare_last_dec=unshare_last_dec,
        )
        # capabilities: which condition axes this checkpoint was trained with,
        # so inference knows which prefix tokens to emit
        self.capabilities = capabilities or {}

    # delegate the generation interface to the inner net
    def encode(self, *a, **k):
        return self.net.encode(*a, **k)

    def decode(self, *a, **k):
        return self.net.decode(*a, **k)

    def forward(self, *a, **k):
        return self.net(*a, **k)

    @property
    def ptr(self):
        return self.net.ptr

    @property
    def beat(self):
        return self.net.beat

    @property
    def tok_emb(self):
        return self.net.tok_emb


class SoftChartPlanner(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/JacobLinCool/SoftChart",
    license="mit",
    tags=["taiko", "rhythm-game", "planning", "music"],
):
    """Song-level plan generator: block audio summaries -> (density, flag) tokens."""

    def __init__(self, feat_dim=261, d=192, layers=4, n_dens=8, n_flag=3):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d)
        self.course = nn.Embedding(5, d)
        self.register_buffer("pos", sinusoidal(128, d), persistent=False)
        enc = nn.TransformerEncoderLayer(d, 6, d * 4, 0.1, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, nn.LayerNorm(d))
        self.h_dens = nn.Linear(d, n_dens)
        self.h_flag = nn.Linear(d, n_flag)

    def forward(self, x, cid):
        h = self.inp(x) + self.course(cid)[:, None] + self.pos[: x.shape[1]]
        h = self.enc(h)
        return self.h_dens(h), self.h_flag(h)
