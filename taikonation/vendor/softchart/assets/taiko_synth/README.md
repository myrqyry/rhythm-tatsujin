# SoftChart Modal-Resonator Taiko Kit

This directory contains three deterministic, synthesized one-shot voices:

- `don.wav` — membrane hit
- `don-big.wav` — independent large membrane hit
- `katsu.wav` — rim hit

The voices are generated at 48 kHz by `scripts/build_taiko_synth.py`. The
signal path uses seeded noise and impulse excitations, banks of decaying modal
resonators, low-frequency pitch sweeps, band-limited stick transients, and
offline RMS-envelope calibration.

Developer-supplied reference WAVs are used only during the offline build to
calibrate envelopes and overall levels. Reference samples, paths, hashes, and
envelope arrays are not copied, mixed, embedded, recorded in the manifest, or
required at runtime. The bundled WAVs contain only mathematical synthesis.

`manifest.json` records the generated files, PCM hashes, onset anchors, and
scalar acoustic profiles. The runtime validates the complete bank before use
and performs one high-quality 48 kHz to preview-rate conversion in memory.

The synthesized assets are distributed under the repository's MIT license.
