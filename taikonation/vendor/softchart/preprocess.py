"""Convert the HF dataset into per-song cache files: log-mel (.npy) + notes (.npz).

Usage (on the training machine):
    python -m softchart.preprocess --out /workspace/softchart/cache [--split train]
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from .vocab import HOP, N_FFT, N_MELS, NOTE_CLASSES, NOTE_TYPE_MAP, SR, COURSES


def to_mono_22k(audio):
    """Handle both datasets<=3 dict audio and datasets>=4 torchcodec AudioDecoder."""
    import torch

    if isinstance(audio, dict):
        arr = np.asarray(audio["array"], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        sr = audio["sampling_rate"]
    else:  # torchcodec AudioDecoder
        samples = audio.get_all_samples()
        arr = samples.data.to(torch.float32).mean(dim=0).numpy()
        sr = samples.sample_rate
    if sr != SR:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return torch.from_numpy(np.ascontiguousarray(arr))[None]  # (1, T)


_mel_fb = None
_window = None


def logmel(wav):
    """(1, T) float32 -> (n_mels, frames) float16 log-mel (torch.stft + librosa fb)."""
    global _mel_fb, _window
    import torch

    if _mel_fb is None:
        import librosa

        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=SR / 2)
        _mel_fb = torch.from_numpy(fb)  # (n_mels, n_fft//2+1)
        _window = torch.hann_window(N_FFT)
    with torch.no_grad():
        spec = torch.stft(
            wav, N_FFT, hop_length=HOP, window=_window, center=True, return_complex=True
        )[0]
        power = spec.abs().pow(2)  # (freq, frames)
        m = torch.log(_mel_fb @ power + 1e-5)
    return m.numpy().astype(np.float16)


def extract_notes(course_struct):
    """course struct -> (times, classes, level) or None."""
    if course_struct is None or not course_struct.get("segments"):
        return None
    times, classes, bpms = [], [], []
    unknown = Counter()
    for seg in course_struct["segments"]:
        for n in seg.get("notes") or []:
            nt = n["note_type"]
            canon = NOTE_TYPE_MAP.get(nt) or NOTE_TYPE_MAP.get(nt.lower())
            if canon is None:
                unknown[nt] += 1
                continue
            times.append(n["timestamp"])
            classes.append(NOTE_CLASSES.index(canon))
            if n.get("bpm"):
                bpms.append(n["bpm"])
    if not times:
        return None
    order = np.argsort(times, kind="stable")
    return (
        np.asarray(times, np.float64)[order],
        np.asarray(classes, np.int8)[order],
        int(course_struct.get("level") or 0),
        float(np.median(bpms)) if bpms else 0.0,
        unknown,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    ds = load_dataset("JacobLinCool/taiko-1000-parsed")
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    os.makedirs(args.out, exist_ok=True)
    index = []
    unknown_total = Counter()

    for split in args.splits:
        d = ds[split]
        for i in range(len(d)):
            row = d[i]
            sid = f"{split}_{i:05d}"
            mel_path = os.path.join(args.out, f"{sid}.mel.npy")
            notes_path = os.path.join(args.out, f"{sid}.notes.npz")
            entry = {
                "id": sid,
                "split": split,
                "title": (row.get("metadata") or {}).get("TITLE", ""),
                "genre": (row.get("metadata") or {}).get("GENRE", ""),
                "courses": {},
            }
            arrays = {}
            for c in COURSES:
                ex = extract_notes(row.get(c))
                if ex is None:
                    continue
                times, classes, level, bpm, unknown = ex
                unknown_total.update(unknown)
                arrays[f"{c}_t"] = times
                arrays[f"{c}_c"] = classes
                entry["courses"][c] = {
                    "level": level,
                    "n_notes": int(len(times)),
                    "bpm": bpm,
                }
            if not arrays:
                print(f"skip {sid}: no charts")
                continue
            if not (os.path.exists(mel_path) and os.path.exists(notes_path)):
                try:
                    mel = logmel(to_mono_22k(row["audio"]))
                except Exception as e:
                    raise RuntimeError(f"{sid}: target-rate audio decode failed") from e
                np.save(mel_path, mel)
                np.savez(notes_path, **arrays)
                entry["n_frames"] = int(mel.shape[1])
            else:
                entry["n_frames"] = int(np.load(mel_path, mmap_mode="r").shape[1])
            index.append(entry)
            if i % 50 == 0:
                print(f"{split} {i}/{len(d)}", flush=True)

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)
    print("unknown note types:", dict(unknown_total))
    print(f"done: {len(index)} songs")


if __name__ == "__main__":
    main()
