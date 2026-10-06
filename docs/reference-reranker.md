# Human-reference candidate reranker

Date: 2026-10-06

Rhythm Tatsujin now has an experimental production selection layer around
SoftChart v1.7. It does not rewrite a generated chart or claim to measure
subjective quality directly. It generates several valid candidates and selects
the one that is most plausible under the local same-difficulty human reference
corpus.

## Idea-generator feature families

The structural feature pass intentionally crossed different mechanisms rather
than adding more variants of notes-per-second:

1. **Rhythm grammar fingerprint** — Taiko Don/Ka sequences crossed with
   n-gram/transition analysis.
2. **Burst and breath grammar** — inter-onset timing crossed with phrase-like
   burst/rest segmentation.
3. **Motif memory** — repeated 4/8-event windows crossed with mapper pattern
   reuse.
4. **Subdivision state machine** — beat-snap vocabulary crossed with
   transition/entropy analysis.
5. **Section envelope** — signal-envelope thinking crossed with note-density
   changes over equal song sections.

These are implemented as deterministic features in
`taikonation.reference.analyze_taiko_map`.

## Wolfram result

The second difficulty-controlled mapper analysis used 242 standard-difficulty
maps collapsed to 63 song-set vectors (46 Nifty, 17 Ryxliee), preserving the
same song-level leakage boundary as the first pass.

Among 13 new structural measurements, **subdivision entropy** was the clear
individual result:

- Hedges' g: **+0.904** (Nifty minus Ryxliee after difficulty normalization)
- Welch p: **0.00042**
- Bonferroni-adjusted p across 13 tests: **0.00545**

Mean color-run length and color-change ratio had moderate raw effects, but did
not survive the 13-test Bonferroni correction.

The full 13-feature structural family had Mahalanobis distance **1.533** with a
1,000-label permutation p of **0.0679** (null 95th percentile **1.600**).
Therefore the structural family as a whole is not treated as independently
established mapper separation yet. The corrected subdivision-entropy result is
stronger than the family-wide result.

## Production score

The candidate scorer uses one representative from each major feature family:

- notes/sec
- median inter-onset interval
- Ka ratio
- big-note ratio
- color-change ratio
- repeated 8-event motif share
- subdivision entropy
- mean burst length
- rest-interval share
- section-density change

For the requested difficulty, each feature is robustly centered/scaled from the
human corpus. Candidate distance is then measured to individual human maps.
The score is the mean distance to the k nearest human maps, rather than distance
to a single global centroid. This preserves multiple legitimate human mapping
styles instead of rewarding only an average-looking chart.

Lower scores mean closer to the human reference neighborhood. The number is a
selection heuristic, not a human quality rating.

## Tournament

Production generation now attempts:

- candidate 0: deterministic greedy SoftChart baseline
- candidates 1–3: near-greedy sampled SoftChart, temperature 0.65 / top-p 0.90,
  deterministic seeds 2001–2003

The greedy baseline is always eligible to win. If the human reference corpus is
missing or unusable, Rhythm Tatsujin returns the deterministic baseline without
spending time on sampled candidates.

## Empirical checks

### User's current song

`o-o-o-oh-some-typo-way-cover`, Oni:

| Candidate | Score | Hits |
| --- | ---: | ---: |
| Greedy seed 0 | 1.1116 | 358 |
| Sample seed 2001 | 1.1947 | 371 |
| Sample seed 2002 | **0.9732** | 364 |
| Sample seed 2003 | 0.9938 | 418 |

The scorer selected seed 2002, a **12.4% lower reference distance** than the
baseline.

### Nifty reference song, held-out set

`Vs. Lancer` (set 873963) was removed from the human scorer while its audio was
used for generation:

| Candidate | Score |
| --- | ---: |
| Greedy | 1.4552 |
| Sample 2001 | 1.2919 |
| Sample 2002 | **1.2456** |
| Sample 2003 | 1.6596 |

The tournament improved the held-out reference distance by **14.4%**.

### Ryxliee reference song, held-out set

`Kiss Me Again (Vampyx Remix) (Cut Ver.)` (set 2306354) was also excluded from
the scorer:

| Candidate | Score |
| --- | ---: |
| Greedy | **0.8415** |
| Sample 2001 | 0.8854 |
| Sample 2002 | 1.0272 |
| Sample 2003 | 1.0049 |

Here the deterministic chart was already best, and the tournament correctly
kept it.

## Live route proof

A real `/api/generate-chart` request produced chart id 2,
`2_reranker-smoke_oni.osu`.

Persisted receipt:

- selected quality score: **0.9732333407**
- attempted candidates: **4**
- selected seed: **2002**
- per-candidate scores persisted with the chart

The corresponding `.osz` endpoint was opened as a ZIP and passed integrity
checking. It contains:

- `2_reranker-smoke_oni.osu`
- `audio.wav`

## What this establishes

This establishes that reference-guided candidate selection is operational and
can choose a lower-distance candidate while preserving the existing greedy chart
when sampling is worse.

It does **not** establish that the selected chart is more fun or musically
better to a player. The next validation is human A/B playtesting. Candidate
scores are persisted specifically so later human ratings can test whether lower
reference distance predicts fun, musicality, playability, or coherence. If it
does not, the scorer should be changed rather than treated as ground truth.
