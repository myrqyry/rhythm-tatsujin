# Mapper style profiles

Local reference corpus snapshot from user-provided OSZ archives.

These statistics are descriptive references, not hard generation quotas. Song
selection differs between mappers, so stylistic conclusions should be validated
against more examples and playtesting.

## Current corpus

- Nifty: 221 Taiko maps across 48 beatmap sets, including 39 custom-name diffs.
- Ryxliee: 68 Taiko maps across 17 beatmap sets, including 8 custom-name diffs.

The importer writes the live machine-readable profiles to ignored
output/reference_corpus/mapper_profiles.json.

## Nifty vs Ryxliee

| Difficulty | Nifty NPS | Ryxliee NPS | Nifty density/Oni | Ryxliee density/Oni |
| --- | ---: | ---: | ---: | ---: |
| Easy | 1.72 | 1.54 | 0.32x | 0.33x |
| Normal | 2.95 | 2.83 | 0.54x | 0.59x |
| Hard | 4.17 | 3.87 | 0.77x | 0.82x |
| Oni | 5.44 | 4.93 | 1.00x | 1.00x |
| Ura | 7.46 | 6.05 | 1.29x | 1.17x |

Ryxliee's Normal and Hard maps stay closer to that song's Oni density, while
Nifty expands the density gap more aggressively at the top end.

| Difficulty | Nifty half-beat+ | Ryxliee half-beat+ | Nifty quarter-beat+ | Ryxliee quarter-beat+ |
| --- | ---: | ---: | ---: | ---: |
| Easy | 0.1% | 3.7% | 0.0% | 0.0% |
| Normal | 18.8% | 30.9% | 0.7% | 3.8% |
| Hard | 41.9% | 49.3% | 8.8% | 11.5% |
| Oni | 60.4% | 60.0% | 20.0% | 16.0% |
| Ura | 76.1% | 80.0% | 27.8% | 36.6% |

Ryxliee introduces faster subdivisions somewhat earlier in the ladder. At Ura,
Ryxliee's raw density remains lower than Nifty's while the quarter-beat-or-faster
share is slightly higher, suggesting a different balance of burst complexity
versus sustained density.

## Accent tendencies

Mean big-note ratios:

| Difficulty | Nifty | Ryxliee |
| --- | ---: | ---: |
| Easy | 5.8% | 10.6% |
| Normal | 3.5% | 5.8% |
| Hard | 2.9% | 4.3% |
| Oni | 2.3% | 5.1% |
| Ura | 1.4% | 6.6% |

Ryxliee currently shows a much stronger big-note/accent tendency in this corpus.

Mean Ka ratios are far more similar: both mappers stay around a roughly balanced
Don/Ka split across the ladder. That overlap is a candidate for a general
human-Taiko prior rather than a style-specific target.

## How to use this

For generated-map evaluation, keep three scores conceptually separate:

1. **Human Taiko plausibility** — patterns shared across strong mappers.
2. **Difficulty plausibility** — whether density and subdivision complexity fit
   the requested level.
3. **Mapper style proximity** — optional similarity to Nifty, Ryxliee, or another
   reference mapper.

Do not collapse these into one metric. A map can be excellent without imitating
one mapper, and a stylistically similar map can still be poor if its timing or
difficulty progression is wrong.


## Statistical follow-up

The first difficulty-controlled song-level statistical pass is recorded in
`docs/wolfram-mapper-analysis.md`. It finds a real multivariate mapper signal,
but the current six coarse metrics achieve only 59.1% held-out balanced
accuracy, so mapper-style scoring should remain a soft prior until richer rhythm
features improve generalization.
