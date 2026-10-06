import hashlib
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from .audio_processing import augment_spectrogram, get_audio_features
from .tokenization import TaikoTokenizer

INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
DIFFICULTY_MAP = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4, "unknown": 1}
_DIFFICULTY_ALIASES = {
    "easy": ("easy", "kantan"),
    "normal": ("normal", "futsuu", "futsu"),
    "hard": ("hard", "muzukashii", "muzukashi"),
    "oni": ("oni",),
    "ura": ("ura", "inner"),
}


def _explicit_difficulty(filename):
    lower = filename.lower()
    for difficulty, aliases in _DIFFICULTY_ALIASES.items():
        if any(alias in lower for alias in aliases):
            return difficulty
    return None


class TaikoTransformerDataset(Dataset):
    """Windowed, aligned audio/chart training data.

    Source arrays retain the original ~23.2 ms TaikoNation timing resolution.
    They are framed to the configured model resolution, normalized with
    statistics computed only from the training fold, then exposed as overlapping
    windows so every part of every song can contribute to training.
    """

    def __init__(
        self,
        all_samples,
        indices,
        tokenizer,
        genre_vocab,
        *,
        is_train=True,
        max_sequence_length=512,
        time_quantization_ms=46.4,
        source_resolution_ms=23.2,
        window_stride=None,
        cache_dir="./cache",
        prefetch=True,
        feature_mean=None,
        feature_std=None,
    ):
        self.is_train = is_train
        self.max_sequence_length = int(max_sequence_length)
        self.tokenizer = tokenizer
        self.genre_vocab = genre_vocab
        self.samples = [all_samples[i] for i in indices]
        self.source_resolution_ms = float(source_resolution_ms)
        self.time_quantization_ms = float(time_quantization_ms)
        requested_stride = int(window_stride or self.max_sequence_length)
        self.window_stride = max(1, min(requested_stride, self.max_sequence_length))
        self.cache_dir = cache_dir
        self.prefetch = prefetch
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)
        self._cache = {}
        self._cache_keys = []
        self._max_cache = 128
        self._token_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._preprocess_audio_async()
        self.windows = self._build_windows()

    def _cache_path(self, sample):
        source = os.path.realpath(sample["song_path"])
        key = f"{source}|{self.source_resolution_ms}|{self.time_quantization_ms}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"spec_{digest}.npy")

    def _preprocess_audio_async(self):
        """Pre-compute framed audio features with atomic cache writes."""
        print(f"Preprocessing audio features for {len(self.samples)} samples...")

        def process_sample(sample):
            cache_path = self._cache_path(sample)
            if os.path.exists(cache_path):
                return False
            audio_features = get_audio_features(
                sample["song_path"],
                source_resolution_ms=self.source_resolution_ms,
                frame_duration_ms=self.time_quantization_ms,
            )
            if audio_features is None:
                return False
            temp_path = cache_path + ".tmp.npy"
            np.save(temp_path, audio_features)
            os.replace(temp_path, cache_path)
            return True

        num_workers = min(mp.cpu_count(), 8)
        processed_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_sample, sample) for sample in self.samples]
            from tqdm import tqdm

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                if future.result():
                    processed_count += 1

        print(
            f"Processed {processed_count} new audio files, "
            f"{len(self.samples) - processed_count} from cache"
        )

    def _get_tokens(self, sample_idx):
        if sample_idx not in self._token_cache:
            tokens = self.tokenizer.tokenize(self.samples[sample_idx]["chart_path"])
            self._token_cache[sample_idx] = tuple(tokens)
        return self._token_cache[sample_idx]

    def _build_windows(self):
        windows = []
        for sample_idx, sample in enumerate(self.samples):
            cache_path = self._cache_path(sample)
            if not os.path.exists(cache_path):
                continue
            try:
                audio_len = int(np.load(cache_path, mmap_mode="r").shape[0])
                token_len = len(self._get_tokens(sample_idx))
            except Exception as exc:
                print(f"Warning: Could not index {sample['song_path']}: {exc}")
                continue

            usable = min(audio_len, token_len)
            if usable <= 0:
                continue

            if usable <= self.max_sequence_length:
                starts = [0]
            else:
                last_start = usable - self.max_sequence_length
                starts = list(range(0, last_start + 1, self.window_stride))
                if starts[-1] != last_start:
                    starts.append(last_start)

            windows.extend((sample_idx, start, usable) for start in starts)

        if not windows:
            raise RuntimeError("Dataset contains no aligned audio/chart windows")
        return windows

    def __len__(self):
        return len(self.windows)

    def _get_cached(self, sample_idx, cache_path):
        if sample_idx in self._cache:
            return self._cache[sample_idx]
        features = np.load(cache_path)
        if len(self._cache_keys) >= self._max_cache:
            evict = self._cache_keys.pop(0)
            del self._cache[evict]
        self._cache[sample_idx] = features
        self._cache_keys.append(sample_idx)
        return features

    def compute_feature_stats(self):
        """Compute per-band statistics from this dataset's songs only."""
        total = None
        total_sq = None
        count = 0
        for sample in self.samples:
            path = self._cache_path(sample)
            if not os.path.exists(path):
                continue
            features = np.load(path, mmap_mode="r").astype(np.float64)
            if features.ndim != 2 or features.shape[0] == 0:
                continue
            if total is None:
                total = np.zeros(features.shape[1], dtype=np.float64)
                total_sq = np.zeros(features.shape[1], dtype=np.float64)
            total += features.sum(axis=0)
            total_sq += np.square(features).sum(axis=0)
            count += features.shape[0]

        if count == 0 or total is None:
            raise RuntimeError("Could not compute training feature statistics")

        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
        std = np.sqrt(variance)
        return mean.astype(np.float32), std.astype(np.float32)

    def set_feature_stats(self, mean, std):
        self.feature_mean = np.asarray(mean, dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(std, dtype=np.float32), 1e-4)

    def __getitem__(self, idx):
        sample_idx, start, usable = self.windows[idx]
        sample = self.samples[sample_idx]
        cache_path = self._cache_path(sample)

        try:
            if self.prefetch:
                full_audio = self._get_cached(sample_idx, cache_path)
            else:
                full_audio = np.load(cache_path)
            full_tokens = self._get_tokens(sample_idx)
        except Exception as exc:
            print(f"Warning: Could not load sample {sample_idx}: {exc}")
            return None

        end = min(start + self.max_sequence_length, usable)
        audio_features = np.asarray(full_audio[start:end], dtype=np.float32).copy()
        token_ids = list(full_tokens[start:end])
        if audio_features.ndim != 2 or not token_ids:
            return None

        if self.feature_mean is not None and self.feature_std is not None:
            audio_features = (audio_features - self.feature_mean) / self.feature_std

        # Zero now means "training-set mean", so SpecAugment masking has a
        # meaningful neutral value rather than injecting an out-of-distribution
        # raw log-Mel value.
        if self.is_train:
            audio_features = augment_spectrogram(audio_features)

        valid_length = len(token_ids)
        pad_id = self.tokenizer.vocab["[PAD]"]
        previous_token = (
            self.tokenizer.vocab["[CLS]"] if start == 0 else int(full_tokens[start - 1])
        )

        if valid_length < self.max_sequence_length:
            pad_width = self.max_sequence_length - valid_length
            audio_features = np.pad(audio_features, ((0, pad_width), (0, 0)))
            token_ids.extend([pad_id] * pad_width)

        decoder_input = [previous_token] + token_ids[:-1]
        target = torch.tensor(token_ids, dtype=torch.long)
        padding_mask = target.eq(pad_id)

        try:
            return {
                "encoder_input": torch.from_numpy(audio_features).float(),
                "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
                "target": target,
                "padding_mask": padding_mask,
                "difficulty": torch.tensor(
                    DIFFICULTY_MAP.get(sample.get("difficulty", "oni").lower(), DIFFICULTY_MAP["oni"]),
                    dtype=torch.long,
                ),
                "genre_id": torch.tensor(
                    self.genre_vocab.get(sample.get("genre", "unknown"), 0),
                    dtype=torch.long,
                ),
            }
        except Exception as exc:
            print(f"Warning: Error creating tensors for sample {sample_idx}: {exc}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_transformer_data_loaders(config, fold_idx=0):
    genre_labels_path = "output/genre_labels.json"
    if os.path.exists(genre_labels_path):
        with open(genre_labels_path, "r") as handle:
            genre_labels = json.load(handle)
    else:
        genre_labels = {}
        print("Warning: genre_labels.json not found. Genres will be 'unknown'.")

    all_samples = []
    genres = {"unknown"}
    try:
        charts = sorted(os.listdir(INPUT_CHART_DIR))
        songs = os.listdir(INPUT_SONG_DIR)
        song_map = {song.split()[0]: song for song in songs}

        for chart_filename in charts:
            id_number = chart_filename.split("_")[0]
            song_filename = song_map.get(id_number)
            if not song_filename:
                continue
            genre = genre_labels.get(song_filename, "unknown")
            genres.add(genre)
            bracket_match = re.search(r"\[(.*?)\]", chart_filename)
            bracket_label = bracket_match.group(1) if bracket_match else chart_filename
            # The published TaikoNation corpus intentionally selected dense,
            # expert-level charts. Do not manufacture Easy/Ura labels from
            # density when the source file carries no explicit difficulty.
            difficulty = _explicit_difficulty(bracket_label) or "oni"
            all_samples.append(
                {
                    "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                    "song_path": os.path.join(INPUT_SONG_DIR, song_filename),
                    "genre": genre,
                    "difficulty": difficulty,
                }
            )
    except FileNotFoundError as exc:
        print(f"Error: Input directory not found - {exc}.")
        return None, None, None, None

    max_samples = config["training"].get("max_samples")
    if config.get("dry_run") and max_samples is None:
        max_samples = 8
    if max_samples is not None:
        all_samples = all_samples[: max(2, int(max_samples))]

    print(f"Found {len(all_samples)} paired song/chart samples.")
    if len(all_samples) < 2:
        raise ValueError("At least two paired song/chart samples are required")

    genre_vocab = {name: i for i, name in enumerate(sorted(genres))}
    n_splits = min(int(config["training"]["k_folds"]), len(all_samples))
    n_splits = max(n_splits, 2)
    train_indices, val_indices = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=42).split(all_samples)
    )[fold_idx]

    data_config = config["data"]
    tokenizer = TaikoTokenizer(
        time_quantization=data_config["time_quantization_ms"],
        source_resolution=data_config["source_resolution_ms"],
    )
    train_dataset = TaikoTransformerDataset(
        all_samples,
        train_indices,
        tokenizer,
        genre_vocab,
        is_train=True,
        max_sequence_length=data_config["max_sequence_length"],
        time_quantization_ms=data_config["time_quantization_ms"],
        source_resolution_ms=data_config["source_resolution_ms"],
        window_stride=data_config.get("window_stride", data_config["max_sequence_length"] // 2),
    )
    feature_mean, feature_std = train_dataset.compute_feature_stats()
    train_dataset.set_feature_stats(feature_mean, feature_std)

    val_dataset = TaikoTransformerDataset(
        all_samples,
        val_indices,
        tokenizer,
        genre_vocab,
        is_train=False,
        max_sequence_length=data_config["max_sequence_length"],
        time_quantization_ms=data_config["time_quantization_ms"],
        source_resolution_ms=data_config["source_resolution_ms"],
        window_stride=data_config.get("validation_window_stride", data_config["max_sequence_length"]),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print(
        f"Training windows: {len(train_dataset)}; validation windows: {len(val_dataset)}; "
        f"token resolution: {data_config['time_quantization_ms']} ms"
    )

    num_workers = max(0, min(mp.cpu_count(), int(config["training"].get("num_workers", 4))))
    val_workers = num_workers // 2
    prefetch_factor = int(config["training"].get("prefetch_factor", 2))
    pin_memory = torch.cuda.is_available()

    train_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    val_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "shuffle": False,
        "num_workers": val_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "persistent_workers": val_workers > 0,
    }
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
    if val_workers > 0:
        val_kwargs["prefetch_factor"] = prefetch_factor

    return (
        DataLoader(train_dataset, **train_kwargs),
        DataLoader(val_dataset, **val_kwargs),
        tokenizer,
        genre_vocab,
    )
