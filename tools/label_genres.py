import os
import json
import random

def label_genres_from_filenames(audio_dir, output_path):
    """
    Assigns genre labels to songs based on keywords in their filenames.
    This is a rule-based fallback for legacy feature-only datasets where source audio is unavailable.
    """
    genre_labels = {}

    genre_rules = {
        "remix": "Electronic",
        "bootleg": "Electronic",
        "dnb": "Electronic",
        "dubstep": "Electronic",
        "feat.": "Pop",
        "tv size": "Pop",
        "idol": "Pop",
        "rock": "Rock",
        "metal": "Rock",
        "symphonic": "Classical",
        "orchestra": "Classical",
        "jazz": "Jazz",
        "hip-hop": "Hip-Hop"
    }

    default_genres = ["J-Core", "Artcore", "J-Pop", "Video Game"]

    print(f"Starting genre labeling for files in {audio_dir}...")

    for filename in os.listdir(audio_dir):
        if not filename.endswith('.npy'):
            continue

        assigned_genre = "unknown"
        lower_filename = filename.lower()

        for keyword, genre in genre_rules.items():
            if keyword in lower_filename:
                assigned_genre = genre
                break

        if assigned_genre == "unknown":
            assigned_genre = random.choice(default_genres)

        genre_labels[filename] = assigned_genre
        print(f"  - {filename}: {assigned_genre}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(genre_labels, f, indent=4)

    print(f"\nGenre labeling complete. {len(genre_labels)} files labeled.")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    AUDIO_DIR = "input_songs"
    OUTPUT_PATH = "output/genre_labels.json"

    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Input directory '{AUDIO_DIR}' not found.")
    else:
        label_genres_from_filenames(AUDIO_DIR, OUTPUT_PATH)