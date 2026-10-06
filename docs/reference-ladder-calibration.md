# Human Taiko ladder calibration notes

Source reviewed: user-provided OSZ archives, 2026-10-05.

These sets are reference evidence for difficulty conditioning and pattern evaluation.
They are not automatically part of training, and their audio is not committed.

## Reviewed sets

### KNOWER — Time Traveler (beatmap set 622136, Taiko mapper: Nifty)

| Difficulty | Notes | Notes/sec | Median IOI |
| --- | ---: | ---: | ---: |
| Kantan | 242 | 1.31 | 750 ms |
| Futsuu | 403 | 2.19 | 375 ms |
| Muzukashii | 675 | 3.67 | 188 ms |
| Oni | 869 | 4.72 | 187 ms |
| Inner Oni | 1169 | 6.35 | 125 ms |

Density relative to Oni is approximately 0.28 / 0.46 / 0.78 / 1.00 / 1.35.
The harder maps increasingly use half-beat and quarter-beat gaps rather than merely
adding arbitrary notes.

### WHOKILLEDXIX — spy (beatmap set 1484545, Taiko mapper: Nifty)

| Difficulty | Notes | Notes/sec | Median IOI |
| --- | ---: | ---: | ---: |
| Kantan | 256 | 1.90 | 450 ms |
| Futsuu | 488 | 3.63 | 300 ms |
| Muzukashii | 669 | 4.98 | 150 ms |
| Oni | 859 | 6.39 | 150 ms |
| Inner Oni | 944 | 7.02 | 150 ms |

Density relative to Oni is approximately 0.30 / 0.57 / 0.78 / 1.00 / 1.10.
Again, the difficulty ladder changes subdivision complexity as well as density.

### Nakanojojo — Bittersweet (beatmap set 700384)

The archive contains standard-osu maps plus two Taiko difficulties by Nifty. The
reference importer correctly ignores the non-Taiko maps.

| Difficulty | Notes | Notes/sec | Median IOI |
| --- | ---: | ---: | ---: |
| Muzukashii | 420 | 4.18 | 174 ms |
| Oni | 576 | 5.73 | 173 ms |

Muzukashii is about 0.73x the Oni note density, consistent with the two full
ladders above.

## Nifty-only slice currently imported

The local reference corpus now contains 298 Taiko maps across 65 beatmap sets. Of those, 221 maps across 48 sets are Nifty-authored. Guest difficulties such as `Nifty's Muzukashii` and `Nifty's Oni` are attributed to Nifty separately from the beatmap-set creator.

| Difficulty | Nifty maps | Mean notes/sec | Half-beat-or-faster share | Quarter-beat-or-faster share |
| --- | ---: | ---: | ---: | ---: |
| Easy/Kantan | 34 | 1.72 | 0.1% | 0.0% |
| Normal/Futsuu | 40 | 2.95 | 18.8% | 0.7% |
| Hard/Muzukashii | 42 | 4.17 | 41.9% | 8.8% |
| Oni | 49 | 5.44 | 60.4% | 20.0% |
| Ura/Inner Oni | 17 | 7.46 | 76.1% | 27.8% |

Two single-difficulty user-selected favorites are explicitly preserved as style exemplars rather than treated as incomplete ladders:

- Camellia — Key of timepieces / `Timeless Oni`: about 6.03 notes/sec, 69.0% half-beat-or-faster intervals, 32.8% quarter-beat-or-faster.
- PSYQUI — Hype feat. Such / `Adrenaline`: about 5.28 notes/sec, 73.5% half-beat-or-faster intervals, 46.5% quarter-beat-or-faster.

Nifty's named custom difficulties such as `Expressionism`, `burger gaming`, `NEUROTIC`, `WHERE YOU AT?`, `Fortes Fortuna Adiuvat`, `Ebb and Flow`, `Spade King`, and `Brutalism` remain custom style references instead of being guessed into standard difficulty labels.

## Current Nifty difficulty signal

Across the imported Nifty-authored maps with an Oni reference in the same set, mean density relative to that song's Oni chart is roughly:

- Easy/Kantan: 0.32x
- Normal/Futsuu: 0.54x
- Hard/Muzukashii: 0.77x
- Oni: 1.00x
- Ura/Inner Oni: 1.29x

This is a useful regression signal, not a generation quota. Musical density
depends on the song. More importantly, the human ladders show that difficulty
changes the distribution of rhythmic subdivisions and pattern complexity, not
only notes-per-second.

## Engineering consequence

Use multi-difficulty sets as grouped reference data: all difficulties sharing one
audio file must stay together for train/validation decisions. Never split their
individual charts independently, or validation will leak the same song/audio from
training.

Local import command:

    uv run taikonation-import-reference reference_osz/*.osz

It filters Mode:1, preserves real difficulty labels, extracts local assets only
under ignored output/reference_corpus/, and reports density/subdivision metrics.
