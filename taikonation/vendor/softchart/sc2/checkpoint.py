"""Strict generator-checkpoint metadata and architecture reconstruction.

V1.8 condition semantics cannot be recovered from tensor shapes.  A missing
``dual`` or ``ctx`` flag changes the decoder prefix while all weights still
load successfully, so release loading must reject incomplete or contradictory
metadata instead of guessing from state-dict keys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

from .vocab import N_MELS, VOCAB


# SoftChart 2.0 checkpoint args (2026-07-25 interface ruling): part of the
# REQUIRED set + presence checks. Pre-2.0 checkpoints lack the keys entirely,
# so parse_generator_checkpoint BACKFILLS them to False before validation —
# older checkpoints keep loading bit-identically. ``axis_knobs`` is a pure
# capability flag (big_rate/span_rate/stream prefix conditioning; no state
# keys of its own).
V2_MODEL_ARGS = frozenset({
    "stage_emb", "span_duration_head", "tempo_head", "rich_section_stats",
    "axis_knobs",
})

REQUIRED_GENERATOR_ARGS = frozenset({
    "d_model", "nhead", "dropout", "enc_layers", "dec_layers", "ffn",
    "emb_factor", "func_time", "aux", "global_ctx", "hierarchical_ctx",
    "section_layers", "section_ffn", "clean_phase", "enc_share",
    "dec_share", "adapter_rank", "unique_layernorm", "adapter_rank_ffn",
    "depth_emb", "unshare_last_dec", "align", "beat_head", "beat_hires",
    "ctx", "sibling", "style", "sync_token", "plan", "mask_infill",
    "complexity", "slot", "dual",
}) | V2_MODEL_ARGS

_BOOL_ARGS = frozenset({
    "func_time", "aux", "global_ctx", "hierarchical_ctx", "clean_phase",
    "unique_layernorm", "depth_emb", "unshare_last_dec", "align",
    "beat_head", "beat_hires", "ctx", "sibling", "style", "sync_token",
    "plan", "complexity", "slot", "dual",
}) | V2_MODEL_ARGS

GENERATOR_CAPABILITY_KEYS = (
    "ctx", "sibling", "style", "sync_token", "plan", "beat_head", "align",
    "aux", "func_time", "mask_infill", "complexity", "slot", "dual",
    "global_ctx", "hierarchical_ctx", "axis_knobs",
)

VOCAB_EXPANSION_SEED = 20260725  # deterministic init for appended token rows


def expand_vocab_rows(state, vocab_size, seed=VOCAB_EXPANSION_SEED):
    """Grow ``tok_emb.weight`` (and a tied ``out.weight``) to ``vocab_size``
    rows, initializing the appended rows N(0, 0.02) exactly like ChartModel's
    embedding init (deterministic via ``seed``).

    The vocabulary only ever grows append-only (vocab.py contract), so an
    older checkpoint's rows are a prefix of the current table. Returns a new
    mapping; tensors that need no change are passed through untouched.
    Raises when the checkpoint has MORE rows than requested (a newer-vocab
    checkpoint cannot be truncated to an older runtime).
    """
    if "tok_emb.weight" not in state:
        raise ValueError("state has no tok_emb.weight to expand")
    old_rows = int(state["tok_emb.weight"].shape[0])
    if old_rows > vocab_size:
        raise ValueError(
            f"checkpoint vocab has {old_rows} rows; cannot shrink to "
            f"{vocab_size} (newer-vocab checkpoint under an older runtime)")
    if old_rows == vocab_size:
        return state
    out = dict(state)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    grown = None
    for key in ("tok_emb.weight", "out.weight"):
        w = state.get(key)
        if w is None:
            continue
        if int(w.shape[0]) != old_rows:
            raise ValueError(
                f"{key} has {int(w.shape[0])} rows but tok_emb.weight has "
                f"{old_rows}; refusing to expand an inconsistent state")
        if grown is None:  # one shared block keeps tied weights tied
            grown = torch.empty(
                vocab_size - old_rows, int(w.shape[1]), dtype=torch.float32,
            ).normal_(mean=0.0, std=0.02, generator=gen)
        out[key] = torch.cat(
            [w, grown.to(device=w.device, dtype=w.dtype)], dim=0)
    return out


def _state_has(state: Mapping[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


def _require_presence(
    state: Mapping[str, torch.Tensor], prefix: str, expected: bool, label: str,
) -> None:
    present = _state_has(state, prefix)
    if present != expected:
        raise ValueError(
            f"checkpoint args/state disagree for {label}: "
            f"metadata={expected}, state={present}"
        )


def _positive_int(args: Mapping[str, object], name: str) -> int:
    value = args[name]
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"checkpoint arg {name} must be a positive integer")
    return int(value)


def _nonnegative_int(args: Mapping[str, object], name: str) -> int:
    value = args[name]
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"checkpoint arg {name} must be a non-negative integer")
    return int(value)


def parse_generator_checkpoint(checkpoint: Mapping[str, object]) -> tuple[
    Mapping[str, torch.Tensor], dict, dict, dict
]:
    """Validate and return ``state, args, model_config, capabilities``.

    The returned configuration is suitable for ``ChartModel``.  Tensor shapes
    are still checked a second time by its strict ``load_state_dict`` call.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("generator checkpoint must be a mapping")
    state = checkpoint.get("model")
    args = checkpoint.get("args")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("generator checkpoint model must be a non-empty mapping")
    if not isinstance(args, Mapping):
        raise ValueError("generator checkpoint args must be a mapping")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("generator state must map string keys to tensors")
    args = dict(args)
    for name in V2_MODEL_ARGS:  # pre-2.0 checkpoints: backfill, keep loading
        args.setdefault(name, False)
    missing = sorted(REQUIRED_GENERATOR_ARGS - set(args))
    if missing:
        raise ValueError(f"generator checkpoint lacks required args: {missing}")

    for name in _BOOL_ARGS:
        if not isinstance(args[name], bool):
            raise ValueError(f"checkpoint arg {name} must be boolean")
    for name in ("d_model", "nhead", "enc_layers", "dec_layers", "ffn",
                 "section_layers"):
        _positive_int(args, name)
    for name in ("emb_factor", "section_ffn", "adapter_rank",
                 "adapter_rank_ffn"):
        _nonnegative_int(args, name)
    dropout = args["dropout"]
    if (not isinstance(dropout, Real) or isinstance(dropout, bool)
            or not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0):
        raise ValueError("checkpoint arg dropout must be finite in [0, 1)")
    if args["enc_share"] is not None and not isinstance(args["enc_share"], (str, int)):
        raise ValueError("checkpoint arg enc_share must be null, integer, or string")
    if args["dec_share"] is not None and not isinstance(args["dec_share"], (str, int)):
        raise ValueError("checkpoint arg dec_share must be null, integer, or string")
    if args["global_ctx"] and args["hierarchical_ctx"]:
        raise ValueError("global_ctx and hierarchical_ctx are mutually exclusive")
    if args["hierarchical_ctx"] and not args["aux"]:
        raise ValueError("hierarchical_ctx requires aux skeleton supervision")
    if args["beat_hires"] and not args["beat_head"]:
        raise ValueError("beat_hires requires beat_head")
    if (args["slot"] or args["dual"]) and not args["clean_phase"]:
        raise ValueError("V1.8 slot/dual checkpoints require clean_phase")
    if args["complexity"]:
        raise ValueError(
            "complexity-conditioned checkpoints are not supported by the "
            "release generation API"
        )

    required_tensors = ("tok_emb.weight", "frontend.0.weight")
    absent = [name for name in required_tensors if name not in state]
    if absent:
        raise ValueError(f"generator state lacks required tensors: {absent}")
    token_weight = state["tok_emb.weight"]
    front_weight = state["frontend.0.weight"]
    if token_weight.ndim != 2 or front_weight.ndim != 3:
        raise ValueError("generator embedding/frontend tensors have invalid rank")

    d_model = int(args["d_model"])
    nhead = int(args["nhead"])
    if d_model % nhead:
        raise ValueError("checkpoint d_model must be divisible by nhead")
    vocab_size, embedding_width = map(int, token_weight.shape)
    if vocab_size > VOCAB.size:
        raise ValueError(
            f"checkpoint vocab has {vocab_size} rows; current vocabulary has "
            f"only {VOCAB.size} (newer-vocab checkpoint under an older runtime)"
        )
    if vocab_size < VOCAB.size:
        # Append-only vocab growth (SoftChart 2.0): older checkpoints load by
        # deterministically initializing the appended token rows.
        state = expand_vocab_rows(state, VOCAB.size)
        token_weight = state["tok_emb.weight"]
        vocab_size = VOCAB.size
    emb_factor = int(args["emb_factor"])
    expected_width = emb_factor or d_model
    if embedding_width != expected_width:
        raise ValueError(
            "checkpoint args/state disagree for emb_factor: "
            f"expected width {expected_width}, state has {embedding_width}"
        )
    if int(front_weight.shape[0]) != d_model:
        raise ValueError("frontend output channels disagree with d_model")
    expected_front_channels = (
        N_MELS if args["clean_phase"]
        else (N_MELS + 2 if (args["slot"] or args["dual"]) else N_MELS)
    )
    if int(front_weight.shape[1]) != expected_front_channels:
        raise ValueError(
            "frontend input channels disagree with clean slot/time metadata: "
            f"expected {expected_front_channels}, state has {front_weight.shape[1]}"
        )

    _require_presence(state, "emb_proj.", emb_factor > 0, "emb_factor")
    _require_presence(
        state, "section_encoder.", args["hierarchical_ctx"], "hierarchical_ctx")
    _require_presence(
        state, "skeleton_proj.", args["hierarchical_ctx"], "hierarchical skeleton")
    _require_presence(state, "gsum_proj.", args["global_ctx"], "global_ctx")
    _require_presence(state, "aux.", args["aux"], "aux")
    _require_presence(state, "phase_proj.", args["clean_phase"], "clean_phase")
    _require_presence(state, "time_proj.", args["func_time"], "func_time")
    _require_presence(state, "ptr.", args["align"], "align")
    _require_presence(state, "beat.", args["beat_head"], "beat_head")
    _require_presence(
        state, "beat.up.", args["beat_head"] and args["beat_hires"], "beat_hires")

    # SoftChart 2.0 presence checks (state-key prefixes per the 2026-07-25
    # interface ruling; axis_knobs carries no state keys)
    _require_presence(state, "stage_emb", args["stage_emb"], "stage_emb")
    _require_presence(state, "span_dur.", args["span_duration_head"],
                      "span_duration_head")
    _require_presence(state, "tempo_logp.", args["tempo_head"], "tempo_head")
    _require_presence(state, "tempo_k.", args["tempo_head"], "tempo_head")
    _require_presence(
        state, "section_encoder.inp_rich.",
        args["hierarchical_ctx"] and args["rich_section_stats"],
        "rich_section_stats")
    max_tgt = args.get("max_tgt")
    if max_tgt is not None and (not isinstance(max_tgt, Integral)
                                or isinstance(max_tgt, bool)
                                or int(max_tgt) <= 0):
        raise ValueError("checkpoint arg max_tgt must be a positive integer")

    beat: bool | str = (
        "hires" if args["beat_head"] and args["beat_hires"]
        else bool(args["beat_head"])
    )
    model_config = {
        "d_model": d_model,
        "nhead": nhead,
        "dropout": float(dropout),
        "enc_layers": int(args["enc_layers"]),
        "dec_layers": int(args["dec_layers"]),
        "ffn": int(args["ffn"]),
        "vocab_size": vocab_size,
        "aux": bool(args["aux"]),
        "global_ctx": bool(args["global_ctx"]),
        "hierarchical_ctx": bool(args["hierarchical_ctx"]),
        "section_layers": int(args["section_layers"]),
        "section_ffn": int(args["section_ffn"]) or None,
        "func_time": bool(args["func_time"]),
        "ptr": bool(args["align"]),
        "beat": beat,
        "in_ch": N_MELS + 2 if (args["slot"] or args["dual"]) else N_MELS,
        "emb_factor": emb_factor or None,
        "clean_phase": bool(args["clean_phase"]),
        "enc_share": args["enc_share"],
        "dec_share": args["dec_share"],
        "adapter_rank": int(args["adapter_rank"]),
        "unique_layernorm": bool(args["unique_layernorm"]),
        "adapter_rank_ffn": int(args["adapter_rank_ffn"]),
        "depth_emb": bool(args["depth_emb"]),
        "unshare_last_dec": bool(args["unshare_last_dec"]),
        "stage_emb": bool(args["stage_emb"]),
        "span_duration_head": bool(args["span_duration_head"]),
        "tempo_head": bool(args["tempo_head"]),
        "rich_section_stats": bool(args["rich_section_stats"]),
    }
    if max_tgt is not None:
        model_config["max_tgt"] = int(max_tgt)
    capabilities = {
        name: (bool(args[name]) if name != "hierarchical_ctx"
               else bool(args["hierarchical_ctx"]))
        for name in GENERATOR_CAPABILITY_KEYS
    }
    # BarScript prefixes ALWAYS carry the sync slot -- [BOS] course level dens
    # SYNC big_rate span_rate stream -- but train.py records the v1.8
    # `sync_token` flag, which sc2_train.sh never sets.  A loader that trusted
    # the recorded False would build a 7-token prefix for an 8-token model and
    # silently collapse every sync bucket onto one conditioning: no error
    # raised, just a dead knob.  Derive it from the mode here, at the single
    # point where capabilities are read off the checkpoint, so every consumer
    # (attach_generator_capabilities, the packager, the HF path) agrees.
    # v1.8 artifacts (mode absent, or slot/time) keep the recorded flag.
    if str(args.get("mode", "")) in ("b0", "c1"):
        capabilities["sync_token"] = True
    return state, args, model_config, capabilities


def attach_generator_capabilities(model, capabilities: Mapping[str, bool]) -> None:
    """Attach the exact semantic flags consumed by generation."""
    capabilities = dict(capabilities)
    # pre-2.0 capability dicts (released HF repos) lack the new knob flag
    capabilities.setdefault("axis_knobs", False)
    missing = sorted(set(GENERATOR_CAPABILITY_KEYS) - set(capabilities))
    if missing:
        raise ValueError(f"generator capabilities are incomplete: {missing}")
    invalid = sorted(
        name for name in GENERATOR_CAPABILITY_KEYS
        if not isinstance(capabilities[name], bool)
    )
    if invalid:
        raise ValueError(f"generator capabilities must be boolean: {invalid}")
    model._has_axis_knobs = bool(capabilities["axis_knobs"])
    model._has_ctx = bool(capabilities["ctx"])
    model._has_sib = bool(capabilities["sibling"])
    model._has_style = bool(capabilities["style"])
    model._has_sync = bool(capabilities["sync_token"])
    model._has_plan = bool(capabilities["plan"])
    model._dual = bool(capabilities["dual"])
    model._slot = bool(capabilities["slot"]) or model._dual
    model._hierarchical_ctx = bool(capabilities["hierarchical_ctx"])
    model._has_gctx = bool(capabilities["global_ctx"] or model._hierarchical_ctx)
