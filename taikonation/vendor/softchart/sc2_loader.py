"""Load and serve the BarScript release package.

This is the only bridge between the Space and ``softchart.sc2``.  It reads a
release package (``config.json``, ``spec.json``, ``model.safetensors``) and
returns an object whose ``generate`` call produces TJA text.

The package is resolved in one order, most explicit first -- ``$SC2_PACKAGE_DIR``,
a local directory, then the Hub repo (see ``resolve_package_dir``).  Whichever
source wins, the package directory is the whole source of truth from that point
on: every integrity check and every serving parameter is read out of it, so a
downloaded package and a local one are served identically or not at all.

The Hub repo is **private**, so the download needs a token.  When none can be
found the loader says exactly that, rather than reporting a missing model --
a token problem and an absent package have different fixes.

Every serving parameter comes from ``config.json``.  The package records not
just the values but the reasoning behind each one in ``config.json:
serving_basis``, so a constant re-typed in the Space would be a second,
unaudited contract.  ``serving_kwargs`` therefore mirrors the package's own
``scripts/sc2_package.py:serving_kwargs`` and passes the recorded values
explicitly into the decoder rather than letting module defaults apply.

Deployment grid: a ``/16`` bar-local lattice decoded with
``bar_denoms='supplied'`` -- the ``SHIP_DECISION`` ruling, and the ``default``
serving profile.  The package also carries a ``deploy_bpm_grid`` profile
(model-emitted denominators masked to divisors of 96); the model card does NOT
recommend it for this checkpoint (§2.8).  A ``/16`` grid makes triplets
arithmetically impossible: 3 does not divide 16.

Two things called "grid" meet here and must not be confused.  ``/16`` is the
BAR-LOCAL lattice: how finely a note may be placed inside one bar.  The BAR
TIMELINE -- where each barline falls in the audio -- is a separate object, and
``generate`` accepts it two ways:

``bpm=`` + ``offset_sec=``
    the grid is synthesized as a uniform tiling, ``bar_sec = (240/bpm)*num/den``
    from the first downbeat.  One scalar tempo for the whole song; a tempo
    change cannot be expressed.
``grid=``
    a caller-built supplied grid whose ``measure_edges`` are used verbatim.
    ``softchart.barscript_grid.barscript_grid_from_fit`` builds one from an
    estimated beat fit, so non-uniform barlines (and therefore tempo change)
    survive into both the decode and the ``#BPMCHANGE`` lines of the TJA.

The returned dict always reports the grid that was actually used, including
which of the two routes produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading

import numpy as np
import torch

from .sc2 import generate as sc2_generate
from .sc2.barscript import BarscriptSpec
from .sc2.checkpoint import (GENERATOR_CAPABILITY_KEYS,
                             attach_generator_capabilities)
from .sc2.generate import (SlotExportError, barscript_grid_from_bpm,
                           generate_song_barscript, require_slot_export)
from .sc2.model import ChartModel
from .sc2.tja import write_tja_slots
from .sc2.vocab import (COURSES, DENS_BUCKET_NPS, FPS, N_DENS, SLOTS,
                        V2_TOKEN_IDS, VOCAB, WINDOW)

CONFIG_SCHEMA = "softchart2_release_config_v1"

# The Hub repo the release is published to. PRIVATE: reading it needs a token
# (see resolve_token). Overridable so a fork or a re-upload does not need a code
# change, but the default is the published release.
DEFAULT_HUB_REPO = "JacobLinCool/softchart-barscript"

# The release's own directory name. This is the RUN identifier, not the public
# name: it is what the package, its manifest and every provenance record call
# this checkpoint, so it stays as built.
RELEASE_DIRNAME = "sc2_c1_v13_s1234_best"

# Exactly the files the release package consists of. snapshot_download is given
# this as its allow-list, so a repo that later grows unrelated files cannot pull
# them into the Space.
PACKAGE_FILES = ("MODEL_CARD.md", "config.json", "model.safetensors",
                 "preprocessor_config.json", "release_manifest.json",
                 "spec.json", "vocab.json")

# Files without which the package cannot be served at all.
REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json",
                  "spec.json", "vocab.json")

# The uniform bar lattice the release ships on (SHIP_DECISION / MODEL_CARD §3,
# "Recommended deployment configuration": --grid bpm --grid-bpm-denom 16).
SHIP_GRID_DENOM = 16

# md5 of every vendored BarScript module, as recorded by the release package in
# config.json:code.module_md5.  Checked at load time -- see verify_code().
VENDORED_MODULES = ("barscript.py", "checkpoint.py", "generate.py",
                    "meter.py", "model.py", "tja.py", "trace.py", "vocab.py")

# TJA `COURSE:` header literals, measured on the corpus rather than assumed.
# Over the 1 155 songs of JacobLinCool/taiko-1000-parsed, 229 carry an `ura`
# chart, and every one of them writes that chart's header as `Edit` (223) or
# the numeric `4` (6).  The string "Ura" appears zero times: it is the
# dataset's normalized course NAME, not the TJA header literal.  `tja.py`
# derives the header as `str(course).capitalize()`, which would emit
# `COURSE:Ura`, so the header is rewritten here -- in code this Space owns --
# and the vendored module stays byte-identical to the shipping tree.
TJA_COURSE_HEADER = {"easy": "Easy", "normal": "Normal", "hard": "Hard",
                     "oni": "Oni", "ura": "Edit"}

# Straight from MODEL_CARD.md §2.  These are the limitations a user must see
# before reading anything into a generated chart; the section numbers point at
# the measurement.  Do not soften them.
KNOWN_LIMITATIONS = (
    ("champion", "§0",
     "The champion comparison is INCONCLUSIVE: the challenger was run at one "
     "seed."),
    ("nesting", "§2.1",
     "The difficulty ladder is not nested. Independently decoded courses "
     "share 0.658/0.667/0.660 of their notes with the next harder course; "
     "authored charts share 0.95-0.99."),
    ("density", "§2.9",
     "Notes come out too dense: 102 sub-40 ms gaps across 29 charts (authored: "
     "2 across 1), 19 of them below the series record of 29.41 ms, and up to "
     "17 notes inside one second."),
    ("micro_spans", "§2.7",
     "One drumroll in seven is a micro-span (14.7 % vs 3.8 % authored)."),
    ("tuplets", "§2.12",
     "Fine-lattice and tuplet placement is much worse than average: 0.345 hit "
     "rate at positions whose denominator is divisible by 3."),
    ("grid", "§2.8",
     "The deployment grid is a uniform /16 lattice, so triplets are "
     "arithmetically impossible -- 3 does not divide 16."),
    ("density_knob", "§2.4",
     "At the top of the density knob, drumrolls are squeezed out: span yield "
     "falls to 33 % of authored. The highest setting is not recommended."),
    ("ura", "corpus",
     "Ura was trained on 208 charts against roughly 1 100 per other course; "
     "expect lower quality."),
)


class PackageError(RuntimeError):
    """The package directory is missing, incomplete or inconsistent."""


# ---------------------------------------------------------------------------
# locating the package


def deploy_package_dir():
    """``spaces/models/<release>`` -- the copy that travels with the Space.

    Named separately from :func:`local_candidates` so it stays available as the
    path to quote in a message even when the candidate list is empty.
    """
    spaces_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(spaces_dir, "models", RELEASE_DIRNAME)


def local_candidates():
    """On-disk directories searched for the package, in order.

    1. :func:`deploy_package_dir`.
    2. the frozen build artifact in the development worktree, for local runs
       from this checkout.
    """
    spaces_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(spaces_dir)
    return (
        deploy_package_dir(),
        os.path.join(repo_root, ".codex", "worktrees", "v18-8h", "artifacts",
                     "sc2", RELEASE_DIRNAME),
    )


def default_package_dir():
    """The package directory reachable WITHOUT the network.

    ``$SC2_PACKAGE_DIR`` if set, else the first populated local candidate, else
    the deploy location as a fallback so callers always get a path to name in a
    message.  This deliberately never downloads: it is what ``is_available``
    and the test suites use to ask whether a package is already on disk.
    Full resolution, Hub included, is :func:`resolve_package_dir`.
    """
    env = os.environ.get("SC2_PACKAGE_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    for candidate in local_candidates():
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return deploy_package_dir()


# ---------------------------------------------------------------------------
# the Hub fallback


def hub_repo_id():
    """The Hub repo the release is published to."""
    return os.environ.get("SC2_REPO_ID", DEFAULT_HUB_REPO)


def hub_enabled():
    """Whether the Hub is allowed as a source at all.

    ``SC2_NO_HUB=1`` pins the Space to on-disk packages, which is how a
    deployment proves it is not silently depending on a download.
    """
    if os.environ.get("SC2_NO_HUB", "").strip().lower() in ("1", "true", "yes"):
        return False
    return bool(hub_repo_id())


def resolve_token(token=None):
    """A Hub token, or ``None``.

    Explicit argument, then ``$HF_TOKEN`` / ``$HUGGING_FACE_HUB_TOKEN``, then
    whatever ``hf auth login`` cached.  On a Space the first is what a secret
    named ``HF_TOKEN`` becomes; locally the last is the usual case.
    """
    for candidate in (token, os.environ.get("HF_TOKEN"),
                      os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if candidate:
            return candidate
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    try:
        return get_token() or None
    except Exception:
        return None


def _no_token_error(repo_id):
    return PackageError(
        f"BarScript is published as a PRIVATE Hub repo ({repo_id}) and no "
        "HuggingFace token was found, so its weights cannot be downloaded. "
        "This is a credentials problem, not a missing model. Fix it in one of "
        "three ways: (1) set $HF_TOKEN to a token with read access to "
        f"{repo_id} -- on a Space that is Settings -> Variables and secrets -> "
        "New secret, named HF_TOKEN; (2) run `hf auth login` on this machine; "
        "or (3) point $SC2_PACKAGE_DIR at a local copy of the release package "
        "and skip the download entirely. Looked in: $HF_TOKEN, "
        "$HUGGING_FACE_HUB_TOKEN, and the HuggingFace token cache.")


def _hub_download_error(repo_id, exc):
    kind = type(exc).__name__
    hint = ""
    if kind in ("RepositoryNotFoundError", "GatedRepoError") \
            or "401" in str(exc) or "403" in str(exc):
        hint = (" The repo is PRIVATE, so this is what a token that is absent, "
                "expired, or lacking read access on it looks like -- the Hub "
                "reports a private repo as missing to callers who may not see "
                "it. Check the token before concluding the repo is gone.")
    elif kind == "LocalEntryNotFoundError" or "offline" in str(exc).lower():
        hint = (" Offline mode is on and the package is not in the local "
                "HuggingFace cache. Unset $HF_HUB_OFFLINE, or point "
                "$SC2_PACKAGE_DIR at a local copy.")
    return PackageError(
        f"could not download the BarScript package from {repo_id}: "
        f"{kind}: {exc}.{hint}")


def hub_snapshot_cached(repo_id=None, revision=None):
    """The cached snapshot directory for the repo, or ``None``. No network."""
    repo_id = repo_id or hub_repo_id()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hit = try_to_load_from_cache(repo_id, "config.json",
                                     revision=revision or None)
    except Exception:
        return None
    if isinstance(hit, str) and os.path.isfile(hit):
        return os.path.dirname(hit)
    return None


def download_from_hub(repo_id=None, revision=None, token=None):
    """Fetch the release package from the Hub and return its directory.

    Only :data:`PACKAGE_FILES` are fetched.  The snapshot is then checked
    against the ``release_manifest.json`` that travels with it, because these
    bytes crossed a network: the manifest records the sha256 of every file as
    built, so a truncated or substituted file is caught here rather than
    surfacing later as an unexplained decode.  Local packages are not
    re-hashed -- this check exists for the transport, and adding it to the
    on-disk paths would change how they already behave.
    """
    repo_id = repo_id or hub_repo_id()
    revision = revision or os.environ.get("SC2_REPO_REVISION") or None
    # The repo is PUBLIC, so a token is optional.  Refusing before the request
    # (as this did while it was private) turns "no credential configured" into
    # "model unavailable" on every deployment without a secret.  Try the
    # download; _no_token_error is raised only if the Hub actually refuses,
    # which is when the credential really is the problem.
    resolved = resolve_token(token)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PackageError(
            "huggingface_hub is required to download the BarScript package "
            f"from {repo_id}; install it or set $SC2_PACKAGE_DIR") from exc
    try:
        path = snapshot_download(repo_id=repo_id, repo_type="model",
                                 revision=revision, token=resolved,
                                 allow_patterns=list(PACKAGE_FILES))
    except Exception as exc:  # noqa: BLE001 -- re-raised as PackageError
        # 401/403/404 with no credential in hand: the repo is private (or was
        # made private again) and this IS a credentials problem -- say so with
        # the actionable message instead of a bare transport error.
        denied = any(t in f"{type(exc).__name__} {exc}"
                     for t in ("401", "403", "404", "RepositoryNotFound",
                               "GatedRepo", "Unauthorized"))
        if denied and resolved is None:
            raise _no_token_error(repo_id) from exc
        raise _hub_download_error(repo_id, exc) from exc
    missing = [n for n in REQUIRED_FILES
               if not os.path.isfile(os.path.join(path, n))]
    if missing:
        raise PackageError(
            f"the Hub repo {repo_id} is not a complete release package; "
            f"missing {missing}")
    verify_manifest(path, source=repo_id)
    return path


def verify_manifest(package_dir, *, source="package"):
    """Check a package's files against its own ``release_manifest.json``.

    A missing manifest is accepted -- not every package carries one -- but a
    manifest that disagrees with the bytes beside it is refused.
    """
    manifest_path = os.path.join(package_dir, "release_manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    bad = []
    for name, meta in (manifest.get("files") or {}).items():
        path = os.path.join(package_dir, name)
        if not os.path.isfile(path):
            if name in REQUIRED_FILES:
                bad.append(f"{name}: missing")
            continue
        size = os.path.getsize(path)
        if size != int(meta["size_bytes"]):
            bad.append(f"{name}: {size} bytes, manifest says {meta['size_bytes']}")
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != meta["sha256"]:
            bad.append(f"{name}: sha256 {digest.hexdigest()[:16]} != "
                       f"{meta['sha256'][:16]}")
    if bad:
        raise PackageError(
            f"the release package from {source} does not match its own "
            "release_manifest.json: " + "; ".join(bad))
    return manifest


def resolve_package_dir(*, allow_hub=True, token=None):
    """``(package_dir, source)`` -- the full resolution order.

    1. ``$SC2_PACKAGE_DIR`` -- an explicit request. If it holds no package this
       RAISES rather than falling through: someone who names a directory wants
       that directory, and silently downloading a different copy instead would
       hide the typo.
    2. a populated local candidate (:func:`local_candidates`).
    3. the Hub repo, downloaded (:func:`download_from_hub`).

    ``source`` is one of ``"env"``, ``"local"``, ``"hub"`` and is reported by
    ``SoftChart2.describe`` so a running Space can say where its weights came
    from.
    """
    env = os.environ.get("SC2_PACKAGE_DIR")
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise PackageError(
                f"$SC2_PACKAGE_DIR points at {path!r}, which has no "
                "config.json. Unset it to fall back to a local copy or the "
                f"Hub repo {hub_repo_id()}, or point it at a release package.")
        return path, "env"
    for candidate in local_candidates():
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate, "local"
    if allow_hub and hub_enabled():
        return download_from_hub(token=token), "hub"
    raise PackageError(
        "no BarScript release package found. Searched $SC2_PACKAGE_DIR and "
        f"{list(local_candidates())}"
        + ("; the Hub fallback is disabled by $SC2_NO_HUB."
           if not hub_enabled() else "; the Hub fallback was not attempted."))


# ---------------------------------------------------------------------------
# integrity


def _canonical_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vocab_tokens():
    """``tokens_by_id`` rebuilt from the VENDORED vocabulary.

    Mirrors ``scripts/sc2_package.py:vocab_contract`` field for field, so its
    sha256 can be compared with the one the package recorded.  A mismatch
    means the vendored ``vocab.py`` is not the table this checkpoint's
    embedding rows were trained against, and the ids in every prefix would be
    silently wrong.
    """
    tokens = [None] * VOCAB.size

    def assign(token_id, name):
        token_id = int(token_id)
        if not 0 <= token_id < VOCAB.size:
            raise PackageError(f"token id {token_id} out of range")
        if tokens[token_id] is not None:
            raise PackageError(f"duplicate token id {token_id}")
        tokens[token_id] = name

    for name in ("pad", "eos", "bos", "unk_cond"):
        assign(getattr(VOCAB, name), name.upper())
    for prefix, mapping in (("COURSE", VOCAB.course), ("LEVEL", VOCAB.level),
                            ("DENS", VOCAB.dens), ("NOTE", VOCAB.note)):
        for value, token_id in mapping.items():
            assign(token_id, f"{prefix}_{value}")
    for frame in range(WINDOW):
        assign(VOCAB.time0 + frame, f"TIME_{frame}")
    assign(VOCAB.sep, "SEP")
    assign(VOCAB.sib, "SIB")
    for prefix, mapping in (("STYLE", VOCAB.style), ("SYNC", VOCAB.sync),
                            ("PDENS", VOCAB.pdens), ("PFLAG", VOCAB.pflag),
                            ("CPLX", VOCAB.cplx), ("MODE", VOCAB.mode)):
        for value, token_id in mapping.items():
            assign(token_id, f"{prefix}_{value}")
    assign(VOCAB.plan, "PLAN")
    assign(VOCAB.mask, "MASK")
    for name, token_id in V2_TOKEN_IDS.items():
        assign(token_id, name)
    missing = [i for i, t in enumerate(tokens) if t is None]
    if missing:
        raise PackageError(f"vocabulary mapping has unmapped ids: {missing[:8]}")
    return tokens


def verify_code(config):
    """Refuse to serve with vendored modules that drifted from the package.

    The package's numbers describe ONE tree of serving code, identified by md5
    in ``config.json:code.module_md5``.  Vendored copies that no longer match
    are a different decoder, and nothing the model card says would still apply.
    """
    recorded = config.get("code", {}).get("module_md5", {})
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc2")
    bad = []
    for name in VENDORED_MODULES:
        want = recorded.get(f"src/softchart/{name}")
        if want is None:
            bad.append(f"{name}: package records no md5")
            continue
        got = _file_md5(os.path.join(here, name))
        if got != want:
            bad.append(f"{name}: {got} != {want}")
    if bad:
        raise PackageError(
            "vendored BarScript serving code does not match the release "
            "package; re-copy from the frozen shipping tree: " + "; ".join(bad))
    return {f"softchart/sc2/{n}": recorded[f"src/softchart/{n}"]
            for n in VENDORED_MODULES}


def verify_vocab(config, package_dir):
    """Cross-check the vendored token table against the package's record."""
    tokens = vocab_tokens()
    got = _canonical_sha256(tokens)
    want = config["vocab"]["mapping_sha256"]
    if got != want:
        raise PackageError(
            f"vendored vocab mapping sha256 {got} != package {want}")
    if int(config["vocab"]["size"]) != VOCAB.size:
        raise PackageError(
            f"vendored vocab size {VOCAB.size} != package "
            f"{config['vocab']['size']}")
    shipped = os.path.join(package_dir, "vocab.json")
    if os.path.isfile(shipped):
        with open(shipped, encoding="utf-8") as f:
            if json.load(f)["tokens_by_id"] != tokens:
                raise PackageError("vocab.json disagrees with the vendored "
                                   "token table")
    if int(config["vocab"]["window_frames"]) != WINDOW \
            or int(config["vocab"]["slots_per_4_4"]) != SLOTS:
        raise PackageError("vendored WINDOW/SLOTS disagree with the package")
    return got


def preprocessing_contract(package_dir):
    """The audio front end the package was trained and evaluated under."""
    path = os.path.join(package_dir, "preprocessor_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_preprocessing(package_dir, *, sr, n_fft, hop_length, n_mels,
                        fmin, fmax):
    """Assert a caller's log-mel front end matches the package contract.

    The Space computes its own mel (``app.load_logmel``); if any parameter
    drifts, the encoder sees a distribution it never trained on and the chart
    degrades silently.  Raises ``PackageError`` listing every mismatch.
    """
    c = preprocessing_contract(package_dir)
    want = {
        "sample_rate_hz": float(c["audio"]["sample_rate_hz"]),
        "n_fft": float(c["stft"]["n_fft"]),
        "hop_length": float(c["stft"]["hop_length"]),
        "n_mels": float(c["mel"]["n_mels"]),
        "fmin_hz": float(c["mel"]["fmin_hz"]),
        "fmax_hz": float(c["mel"]["fmax_hz"]),
    }
    got = {"sample_rate_hz": float(sr), "n_fft": float(n_fft),
           "hop_length": float(hop_length), "n_mels": float(n_mels),
           "fmin_hz": float(fmin), "fmax_hz": float(fmax)}
    bad = [f"{k}: {got[k]} != {want[k]}" for k in want if got[k] != want[k]]
    if bad:
        raise PackageError("log-mel front end disagrees with the package "
                           "contract: " + "; ".join(bad))
    return c


# ---------------------------------------------------------------------------
# the serving contract, read from the package


def serving_kwargs(config, profile="default"):
    """``generate_song_barscript`` keyword arguments this release ships.

    Field-for-field the same function as ``scripts/sc2_package.py:
    serving_kwargs``: values are passed EXPLICITLY so the contract the package
    recorded is the one that reaches the decoder, and never a module default
    that happens to agree today.
    """
    profiles = config.get("serving_profiles") or {"default": {"overrides": {}}}
    if profile not in profiles:
        raise PackageError(f"unknown serving profile {profile!r}; "
                           f"have {sorted(profiles)}")
    s = dict(config["serving"], **profiles[profile].get("overrides", {}))
    return {
        "span_min_sec": s["span_min_sec"],
        "span_min_beats": s["span_min_beats"],
        "degenerate_bar_sec": s["degenerate_bar_sec"],
        "min_onset_gap_sec": s["min_onset_gap_sec"],
        "bar_denoms": s["bar_denoms"],
        "bar_denom_mask": s["bar_denom_mask"],
        "greedy": s["greedy"],
        "temperature": s["temperature"],
        "plan_temperature": s["plan_temperature"],
        "top_p": s["top_p"],
    }


# ---------------------------------------------------------------------------
# supplied bar timelines


SUPPLIED_GRID_KEYS = ("bpm", "downbeats", "measure_edges", "measure_num",
                      "measure_den", "bar_denoms")


def _check_supplied_grid(grid, grid_denom):
    """Shape-check a caller-built supplied grid before it reaches the decoder.

    The decoder's own ``generate._barscript_grid`` validates the geometry
    (edges increasing, one edge per bar plus the terminal one, denominators in
    range) and is the authority on it, so this does not duplicate that work.
    What it adds is the check the decoder cannot make: that ``grid_denom`` --
    the lattice the CALLER believes it is serving, and the number reported to
    the user -- actually agrees with the denominators in the grid.  A grid
    built for a different lattice than the one being advertised would decode
    fine and be described wrongly.
    """
    if not isinstance(grid, dict):
        raise ValueError("grid must be a dict")
    missing = [k for k in SUPPLIED_GRID_KEYS if k not in grid]
    if missing:
        raise ValueError(f"supplied grid is missing {missing}; a BarScript "
                         f"grid needs {list(SUPPLIED_GRID_KEYS)}")
    denoms = [int(d) for d in grid["bar_denoms"]]
    if not denoms:
        raise ValueError("supplied grid has no bars")
    if grid_denom is not None and set(denoms) != {int(grid_denom)}:
        raise ValueError(
            f"supplied grid carries bar_denoms {sorted(set(denoms))} but "
            f"grid_denom={int(grid_denom)} was requested; pass "
            "grid_denom=None to serve a grid whose lattice varies per bar")
    return grid


def _supplied_grid_info(grid, duration_sec):
    """Provenance for a supplied grid, including whether tempo actually varies.

    ``n_distinct_bar_sec`` is the load-bearing field: a supplied grid is the
    only route that CAN carry a tempo change, but taking it does not mean a
    change was found.  Bar lengths are compared at microsecond resolution --
    the same rounding ``barscript_grid.barscript_grid_from_fit`` records -- so
    a grid that is uniform to within float noise reports one distinct length.
    """
    edges = np.asarray(grid["measure_edges"], dtype=np.float64)
    bar_sec = np.diff(edges)
    quarter = np.asarray([4.0 * int(n) / int(d) for n, d in
                          zip(grid["measure_num"], grid["measure_den"])],
                         dtype=np.float64)
    bar_bpm = 60.0 * quarter / bar_sec
    denoms = [int(d) for d in grid["bar_denoms"]]
    meters = sorted({(int(n), int(d)) for n, d in
                     zip(grid["measure_num"], grid["measure_den"])})
    info = {
        "source": "supplied",
        "denom": denoms[0] if len(set(denoms)) == 1 else None,
        "bpm": float(grid["bpm"]),
        "offset_sec": float(edges[0]),
        "duration_sec": float(duration_sec),
        "meter": [meters[0][0], meters[0][1]],
        "meter_varies": len(meters) > 1,
        "n_bars": len(denoms),
        "triplets_representable": all(d % 3 == 0 for d in denoms),
        "n_distinct_bar_sec": int(np.unique(np.round(bar_sec, 6)).size),
        "bar_bpm_min": float(bar_bpm.min()),
        "bar_bpm_max": float(bar_bpm.max()),
        "terminal_edge_sec": float(edges[-1]),
    }
    # provenance the grid builder attached, carried through untouched so the
    # caller does not have to hold on to the builder's return value
    for key in ("grid_source", "fit_quality", "grid_meta"):
        if key in grid:
            info[key] = grid[key]
    return info


class MotifConstraint:
    """Serve the motif-hard value the package RESOLVED, not a module default.

    ``generate._decode_barscript_window`` constructs ``BarscriptFSM(spec)``
    and exposes no parameter, so the module default would otherwise be the
    only value a decode could get.  The vendored ``generate.py`` is
    byte-identical to the shipping tree and stays that way, so the serving
    value is injected by wrapping ``BarscriptFSM.__init__`` for the duration
    of the decode -- the same mechanism, for the same reason, as
    ``scripts/sc2_package.py`` and ``scripts/sc2_generate_eval.py``.  The
    class object is untouched, so ``isinstance`` checks and any caller that
    passes ``motif_hard`` explicitly keep working.

    ``observed_state`` reports what was SERVED, so a decode that never built
    an FSM cannot be reported as compliant.
    """

    def __init__(self, motif_hard):
        self.motif_hard = bool(motif_hard)
        self.n_constructed = 0
        self.observed = set()

    def __enter__(self):
        from .sc2 import barscript as bs_mod

        self._mod = bs_mod
        self._orig = bs_mod.BarscriptFSM.__init__
        orig, owner = self._orig, self

        def init(fsm, spec, vocab=bs_mod.VOCAB, motif_hard=None):
            hard = owner.motif_hard if motif_hard is None else bool(motif_hard)
            owner.n_constructed += 1
            owner.observed.add(hard)
            orig(fsm, spec, vocab, hard)

        bs_mod.BarscriptFSM.__init__ = init
        return self

    def __exit__(self, *exc):
        self._mod.BarscriptFSM.__init__ = self._orig
        return False

    def observed_state(self):
        return {"n_fsm_constructed": self.n_constructed,
                "observed_motif_hard": sorted(self.observed),
                "verified": bool(self.n_constructed > 0
                                 and self.observed == {self.motif_hard})}


def motif_context(config):
    """The serving FSM configuration this package resolved at build time."""
    return MotifConstraint(
        config["train_provenance"]["motif_constraint"]["resolved_motif_hard"])


# ---------------------------------------------------------------------------
# loading


def load_release(package_dir, device="cpu"):
    """Rebuild the model from a package, reading nothing outside it.

    Returns ``(model, config, spec)``.  Ports ``scripts/sc2_package.py:
    load_release`` onto the vendored namespace; the checkpoint file, the
    architecture arguments, the capability flags and the label spec all come
    from ``package_dir``.
    """
    package_dir = os.path.abspath(package_dir)
    config_path = os.path.join(package_dir, "config.json")
    if not os.path.isfile(config_path):
        raise PackageError(
            f"no BarScript package at {package_dir!r} (config.json "
            "missing). Point $SC2_PACKAGE_DIR at a release directory, or "
            f"leave it unset to download {hub_repo_id()} from the Hub.")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PackageError(
            f"unsupported package schema {config.get('schema_version')!r}")

    verify_code(config)
    verify_vocab(config, package_dir)

    from safetensors.torch import load_file

    weights = os.path.join(package_dir, config["weights"]["file"])
    state = load_file(weights)
    # Weights ship bfloat16 (config.weights.cast_from == float32); the forward
    # pass runs under autocast, so they are restored to float32 storage, as
    # the package's own loader does.
    state = {k: (v.to(torch.float32) if v.is_floating_point() else v)
             for k, v in state.items()}
    model = ChartModel(**config["model"]).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    caps = dict(config["capabilities"])
    missing = sorted(set(GENERATOR_CAPABILITY_KEYS) - set(caps))
    if missing:
        raise PackageError(f"package capabilities are incomplete: {missing}")
    attach_generator_capabilities(model, caps)
    model._has_axis_knobs = bool(caps.get("axis_knobs", False))
    # config.serving_prefix_fix: train.py records sync_token=False while the
    # training prefix carries the SYNC slot. A loader that trusts the recorded
    # flag drops the slot and every sync bucket reaches the decoder as an
    # identical prefix.
    model._has_sync = True

    spec = BarscriptSpec.load(os.path.join(package_dir, "spec.json"))
    return model, config, spec


# ---------------------------------------------------------------------------
# the interface the app calls


def _course_header(text, course):
    header = TJA_COURSE_HEADER.get(str(course).lower())
    if header is None:
        return text
    out, n = re.subn(r"(?m)^COURSE:.*$", f"COURSE:{header}", text, count=1)
    if n != 1:
        raise PackageError("exported TJA has no COURSE header to normalize")
    return out


class SoftChart2:
    """A loaded BarScript release, ready to generate.

    Construct via :func:`load` (which caches by directory).  ``generate`` is
    the whole serving surface.
    """

    def __init__(self, model, config, spec, package_dir, device):
        self.model = model
        self.config = config
        self.spec = spec
        self.package_dir = package_dir
        self.device = device
        # Where the package came from: "env", "local", "hub" or "explicit".
        # Set by load(); a directly constructed instance has not been resolved.
        self.source = "explicit"
        self._lock = threading.Lock()

    # -- description ------------------------------------------------------

    @property
    def release_name(self):
        return self.config.get("release_name", "sc2")

    @property
    def parameters(self):
        return int(self.config["weights"]["parameters"])

    @property
    def courses(self):
        return list(COURSES)

    def serving(self, profile="default"):
        return serving_kwargs(self.config, profile)

    def limitations(self):
        """``[(key, section, text)]`` -- MODEL_CARD §2, for the UI."""
        return list(KNOWN_LIMITATIONS)

    def describe(self):
        return {
            "release_name": self.release_name,
            "parameters": self.parameters,
            "experimental": True,
            "package_dir": self.package_dir,
            "package_source": self.source,
            "hub_repo": hub_repo_id() if self.source == "hub" else None,
            "source_seed": self.config.get("source", {}).get("seed"),
            "source_step": self.config.get("source", {}).get("step"),
            "weights_dtype": self.config["weights"]["dtype"],
            "grid": {"source": "bpm", "denom": SHIP_GRID_DENOM,
                     "bar_denoms": self.serving()["bar_denoms"]},
            "serving": self.serving(),
            "motif_hard": self.config["train_provenance"][
                "motif_constraint"]["resolved_motif_hard"],
            "limitations": self.limitations(),
        }

    # -- conditions -------------------------------------------------------

    def density_bucket_from_nps(self, nps):
        """Notes-per-second -> DENS bucket, using the package's bucket width.

        MODEL_CARD §2.5: absolute bucket calibration is off on the density
        axis, so treat this as a knob, not a promise of a note count.
        """
        width = float(preprocessing_contract(self.package_dir).get(
            "density_bucket_nps", DENS_BUCKET_NPS))
        return int(min(N_DENS - 1, max(0, int(float(nps) / width))))

    # -- generation -------------------------------------------------------

    def generate(self, mel, course, level=None, *, bpm=None,
                 offset_sec=0.0, duration_sec=None, grid=None,
                 density_bucket=None,
                 density_nps=None, meter=(4, 4), grid_denom=SHIP_GRID_DENOM,
                 seed=0, big_rate=None, span_rate=None, stream=None,
                 sync_band=None, title="SoftChart", wave="song.ogg",
                 balloon_count=10, profile="default", out_path=None,
                 on_progress=None, **overrides):
        """Generate one chart and return it as TJA text plus provenance.

        ``mel``      (n_mels, T) natural-log mel, the front end described by
                     ``preprocessor_config.json`` (see ``check_preprocessing``).
        ``course``   one of ``COURSES`` (easy/normal/hard/oni/ura), or ``None``
                     to decode unconditioned.
        ``level``    TJA level, clamped to 1..12 by the prefix builder.
        ``bpm``      tempo of the SYNTHESIZED grid.  Mutually exclusive with
                     ``grid``; exactly one of the two must be given.
        ``offset_sec``  audio time of the FIRST DOWNBEAT, in seconds.  This is
                     the negative of the TJA ``OFFSET`` field.  Only meaningful
                     on the ``bpm`` route.
        ``grid``     a ready-made supplied grid -- ``{bpm, downbeats,
                     measure_edges, measure_num, measure_den, bar_denoms}`` --
                     used VERBATIM instead of tiling one from ``bpm``.  This is
                     the route that can carry a tempo change: the decoder and
                     ``write_tja_slots`` both read bar times out of
                     ``measure_edges``, so non-uniform bars stay non-uniform.
                     Build one with
                     ``softchart.barscript_grid.barscript_grid_from_fit``.
        ``duration_sec``  defaults to ``mel.shape[1] / FPS``.
        ``density_bucket`` DENS bucket 0..15, or ``density_nps`` in notes per
                     second, which is bucketed with the package's width.
        ``grid_denom``  bar-local lattice denominator teacher-forced into every
                     [BAR] header.  16 is the shipped deployment grid.  With a
                     supplied ``grid`` the grid's own ``bar_denoms`` govern and
                     ``grid_denom`` is only cross-checked against them.
        ``overrides``  serving parameters (``temperature``, ``top_p``,
                     ``min_onset_gap_sec``, ...).  The package's recorded
                     values apply unless one is named here; anything overridden
                     is reported back under ``serving_overrides``, because the
                     model card's numbers no longer describe that decode.

        Returns a dict:  ``tja`` (text), ``grid`` (the lattice actually used),
        ``serving`` (the kwargs that reached the decoder), ``serving_overrides``,
        ``motif`` (what the FSM was actually served), ``counters`` (decoder
        clamp/floor tallies), ``gen`` (the raw generation), plus ``course``,
        ``level``, ``density_bucket``, ``seed``, ``n_notes``, ``n_spans``,
        ``duration_sec`` and ``release_name``.

        Raises ``SlotExportError`` when a decoded position does not sit on the
        96-slot TJA export lattice; the generation is intact but no TJA can be
        written for it.
        """
        if course is not None and str(course).lower() not in COURSES:
            raise ValueError(f"unknown course {course!r}; "
                             f"choose one of {COURSES}")
        course = None if course is None else str(course).lower()
        if mel.ndim != 2:
            raise ValueError("mel must be (n_mels, T)")
        if (bpm is None) == (grid is None):
            raise ValueError(
                "pass exactly one of bpm= (synthesize a uniform grid) or "
                "grid= (use a supplied grid verbatim)")
        if duration_sec is None:
            duration_sec = mel.shape[1] / FPS
        if density_bucket is None and density_nps is not None:
            density_bucket = self.density_bucket_from_nps(density_nps)
        if density_bucket is not None:
            density_bucket = int(density_bucket)
            if not 0 <= density_bucket < N_DENS:
                raise ValueError(f"density_bucket must be in [0, {N_DENS})")

        kwargs = serving_kwargs(self.config, profile)
        unknown = sorted(set(overrides) - set(kwargs))
        if unknown:
            raise ValueError(f"not serving parameters: {unknown}")
        applied = {k: v for k, v in overrides.items()
                   if k in kwargs and v is not None and v != kwargs[k]}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

        if grid is None:
            grid = barscript_grid_from_bpm(bpm, float(offset_sec),
                                           float(duration_sec), meter=meter,
                                           denom=int(grid_denom))
            grid_info = {
                "source": "bpm",
                "denom": int(grid_denom),
                "bpm": float(bpm),
                "offset_sec": float(offset_sec),
                "duration_sec": float(duration_sec),
                "meter": [int(meter[0]), int(meter[1])],
                "n_bars": len(grid["bar_denoms"]),
                "triplets_representable": (int(grid_denom) % 3 == 0),
            }
        else:
            grid = _check_supplied_grid(grid, grid_denom)
            grid_info = _supplied_grid_info(grid, float(duration_sec))
        grid_info["bar_denoms"] = kwargs["bar_denoms"]
        grid_info["bar_denom_mask"] = kwargs["bar_denom_mask"]
        grid_info["profile"] = profile

        # config.serving_basis: span_bucket0_profile is null in this release
        # and is a KNOWN GAP (§2.7), not a tuned choice. A package that ever
        # ships one needs a builder here, so refuse rather than drop it.
        if self.config["serving"].get("span_bucket0_profile") is not None:
            raise PackageError(
                "package ships a span_bucket0_profile; this loader cannot "
                "build one and will not serve without it")

        motif = motif_context(self.config)
        with self._lock:
            with motif:
                gen = generate_song_barscript(
                    self.model, mel, grid, course, level=level,
                    density_bucket=density_bucket, spec=self.spec,
                    device=self.device, seed=int(seed), sync_band=sync_band,
                    big_rate=big_rate, span_rate=span_rate, stream=stream,
                    span_bucket0_profile=None,
                    on_progress=on_progress, **kwargs)

        require_slot_export(gen)
        tja = write_tja_slots(gen, grid, title=title, course=course or "oni",
                              level=level or 0, wave=wave,
                              balloon_count=balloon_count)
        tja = _course_header(tja, course or "oni")
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(tja)

        return {
            "tja": tja,
            "grid": grid_info,
            "serving": kwargs,
            "serving_overrides": applied,
            "motif": motif.observed_state(),
            "counters": {k: int(v) for k, v in gen["counters"].items()},
            "gen": gen,
            "course": course,
            "level": level,
            "density_bucket": density_bucket,
            "seed": int(seed),
            "n_notes": len(gen["hits"]),
            "n_spans": len(gen["spans"]),
            "duration_sec": float(duration_sec),
            "release_name": self.release_name,
            "experimental": True,
        }


_LOADED = {}
_LOAD_LOCK = threading.Lock()


def load(package_dir=None, device="cpu", *, token=None):
    """Load (and cache) the release package.

    With no ``package_dir`` the source is chosen by :func:`resolve_package_dir`,
    which may download from the Hub.  Either way the package that wins is put
    through the same gate: ``load_release`` verifies the eight vendored serving
    modules by md5 against ``config.json:code.module_md5``, cross-checks the
    vendored token table, and rejects an unknown package schema.
    """
    if package_dir is None:
        package_dir, source = resolve_package_dir(token=token)
    else:
        package_dir, source = os.path.abspath(package_dir), "explicit"
    key = (package_dir, device)
    with _LOAD_LOCK:
        if key not in _LOADED:
            model, config, spec = load_release(package_dir, device=device)
            release = SoftChart2(model, config, spec, package_dir, device)
            release.source = source
            _LOADED[key] = release
        return _LOADED[key]


def is_available(package_dir=None):
    """Whether the model can be offered, without loading any weights.

    True when a package is on disk, and also when the Hub fallback has a real
    chance of producing one -- the repo is already cached, or a token exists and
    the process is not offline.  Offering the entry on a resolvable-but-not-yet-
    downloaded Hub is deliberate: the failure a user then sees names the missing
    token, whereas hiding the entry would tell them the model does not exist.
    """
    if package_dir is not None:
        return os.path.isfile(os.path.join(package_dir, "config.json"))
    if os.path.isfile(os.path.join(default_package_dir(), "config.json")):
        return True
    if os.environ.get("SC2_PACKAGE_DIR"):
        # An explicit request that is not satisfiable: do not quietly answer it
        # from the Hub. resolve_package_dir raises with the path that was named.
        return False
    if not hub_enabled():
        return False
    if hub_snapshot_cached() is not None:
        return True
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() \
        in ("1", "true", "yes")
    if offline:
        return False
    # A PUBLIC repo needs no token.  Requiring one here (as this did while the
    # repo was private) silently drops the entry from the model list on any
    # deployment without a secret -- which is exactly the failure mode this
    # function's docstring says to avoid.  Offer it whenever the Hub is
    # reachable at all; download_from_hub still raises a named error if the
    # repo turns out to be private and no credential is present.
    return True


__all__ = ["SoftChart2", "SlotExportError", "PackageError", "load",
           "load_release", "is_available", "default_package_dir",
           "local_candidates", "resolve_package_dir", "download_from_hub",
           "hub_repo_id", "hub_enabled", "hub_snapshot_cached",
           "resolve_token", "verify_manifest",
           "serving_kwargs", "motif_context", "check_preprocessing",
           "preprocessing_contract", "verify_code", "verify_vocab",
           "vocab_tokens", "KNOWN_LIMITATIONS", "TJA_COURSE_HEADER",
           "SHIP_GRID_DENOM", "DEFAULT_HUB_REPO", "PACKAGE_FILES",
           "REQUIRED_FILES", "sc2_generate"]
