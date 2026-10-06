import numpy as np

def get_audio_features(audio_path, source_resolution_ms=23.2, frame_duration_ms=46.4):
    """
    Loads pre-computed audio features from a .npy file and frames them to align
    with the tokenization time steps.

    The original project stored audio features as numpy arrays, not raw audio.
    This function adapts to that format.

    Args:
        audio_path (str): Path to the .npy audio feature file.
        source_resolution_ms (float): The time resolution of each row in the source .npy file.
                                      23.2ms corresponds to 22050Hz / 512 hop_length.
        frame_duration_ms (float): Target duration of each model frame, matching the tokenizer.

    Returns:
        np.ndarray: A numpy array of shape (num_frames, feature_size), where each
                    row is a feature vector for a single time step.
        None: If an error occurs during loading or processing.
    """
    try:
        # 1. Load the pre-computed features from the .npy file
        audio_features = np.load(audio_path)
        if audio_features.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got shape {audio_features.shape}")

        # 2. Calculate how many source steps fit into one of our new frames.
        steps_per_frame = int(round(frame_duration_ms / source_resolution_ms))
        if steps_per_frame == 0: steps_per_frame = 1

        num_source_steps = audio_features.shape[0]
        num_frames = num_source_steps // steps_per_frame
        if num_frames == 0:
            return np.empty((0, audio_features.shape[1]), dtype=np.float32)

        # Vectorized framing keeps corpus preparation cheap even at the
        # finer default 46.4 ms model resolution.
        trimmed = audio_features[: num_frames * steps_per_frame]
        framed = trimmed.reshape(num_frames, steps_per_frame, audio_features.shape[1]).mean(axis=1)
        return framed.astype(np.float32, copy=False)

    except Exception as e:
        print(f"Error processing audio feature file {audio_path}: {e}")
        return None

def augment_spectrogram(spectrogram, freq_mask_param=27, time_mask_param=50):
    """
    Applies SpecAugment-style masking to a spectrogram.

    Args:
        spectrogram (np.ndarray): The input spectrogram of shape (time_steps, n_mels).
        freq_mask_param (int): The maximum width of the frequency mask.
        time_mask_param (int): The maximum width of the time mask.

    Returns:
        np.ndarray: The augmented spectrogram.
    """
    augmented_spec = spectrogram.copy()
    num_time_steps, num_mels = augmented_spec.shape

    # --- Frequency Masking ---
    f = int(np.random.uniform(low=0.0, high=min(freq_mask_param, num_mels - 1)))
    if f > 0:
        f0 = np.random.randint(0, num_mels - f)
        augmented_spec[:, f0:f0 + f] = 0

    # --- Time Masking ---
    t = int(np.random.uniform(low=0.0, high=min(time_mask_param, num_time_steps - 1)))
    if t > 0:
        t0 = np.random.randint(0, num_time_steps - t)
        augmented_spec[t0:t0 + t, :] = 0

    return augmented_spec
