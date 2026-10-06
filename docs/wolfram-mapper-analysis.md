# Wolfram mapper-style analysis

Date: 2026-10-06

This is the first external statistical pass over the local human-reference corpus.
It asks a deliberately narrow question:

> After controlling for requested difficulty, do the current coarse chart metrics
> contain a reproducible Nifty-versus-Ryxliee mapper signal?

## Dataset and leakage control

Input is the ignored local reference corpus under `output/reference_corpus/`.

Only standard-difficulty maps authored by Nifty or Ryxliee were included. Custom
difficulty names were excluded from this test because they do not supply a
consistent difficulty control.

The source table contained 242 labeled standard-difficulty maps across 63 song
sets. To avoid treating several difficulties from the same song as independent
evidence:

1. Each metric was standardized within difficulty across the pooled human maps.
2. Standardized maps were averaged by mapper + beatmap set.
3. Statistical analysis used one vector per mapper/song set.

Final sample:

- Nifty: 46 song-set vectors
- Ryxliee: 17 song-set vectors

The six tested features were:

- notes per second
- median inter-onset interval
- Ka ratio
- big-note ratio
- half-beat-or-faster interval share
- quarter-beat-or-faster interval share

## Univariate results

Positive Hedges' g means larger for Nifty after difficulty normalization.

| Feature | Nifty mean z | Ryxliee mean z | Hedges' g | Welch p | Bonferroni p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Notes/sec | 0.079 | -0.365 | +0.476 | 0.079 | 0.472 |
| Median IOI | -0.101 | +0.360 | -0.645 | 0.038 | 0.226 |
| Ka ratio | +0.061 | -0.009 | +0.079 | 0.764 | 1.000 |
| Big-note ratio | -0.123 | +0.335 | -0.487 | 0.140 | 0.843 |
| Half-beat-or-faster | -0.065 | +0.379 | -0.478 | 0.206 | 1.000 |
| Quarter-beat-or-faster | -0.023 | +0.274 | -0.333 | 0.369 | 1.000 |

No single feature survives the six-test Bonferroni correction. Median inter-onset
interval is the strongest individual effect, but it should not be treated as an
independently established mapper signature from this sample.

## Joint signal

Wolfram computed a pooled-covariance Mahalanobis distance of:

- observed distance: **1.1352**

A 1,000-label permutation test preserving the mapper sample sizes produced:

- null mean: **0.7094**
- null 95th percentile: **1.0860**
- permutation p: **0.0370**

So the six features jointly contain more mapper separation than expected from
random mapper labels in this sample.

This is evidence for a real joint mapper signal, not evidence that the current
feature set is sufficient for reliable style identification.

## Held-out discrimination

A leave-one-song-set-out pooled-covariance nearest-centroid classifier was used
as a deliberately simple falsification check.

Confusion counts:

- Nifty -> Nifty: 30
- Nifty -> Ryxliee: 16
- Ryxliee -> Ryxliee: 9
- Ryxliee -> Nifty: 8

Performance:

- raw accuracy: **61.9%**
- Nifty recall: **65.2%**
- Ryxliee recall: **52.9%**
- balanced accuracy: **59.1%**

Because the corpus is imbalanced, raw accuracy is not the useful headline; an
always-Nifty classifier would score about 73% raw accuracy while having useless
Ryxliee recall. Balanced accuracy shows that the current coarse metrics only
weakly generalize to unseen song sets.

## Discriminant direction

For interpretation only, the pooled linear discriminant direction was normalized
by total absolute coefficient magnitude:

| Feature | Normalized coefficient |
| --- | ---: |
| Median IOI | -0.395 |
| Big-note ratio | -0.243 |
| Half-beat-or-faster | -0.220 |
| Quarter-beat-or-faster | -0.100 |
| Ka ratio | +0.035 |
| Notes/sec | -0.007 |

These coefficients are correlated-model weights, not independent feature
importance scores. Their useful qualitative message is that the current mapper
separation is carried much more by **timing texture and accent usage** than by
raw density or Don/Ka balance.

That agrees with the descriptive corpus observations:

- Nifty tends toward shorter inter-onset spacing at a given difficulty.
- Ryxliee tends toward more big-note/accent use.
- Ryxliee often introduces finer subdivisions without simply increasing raw
  notes/sec.
- Don/Ka balance remains very similar between the two mappers.

## Decision

Do **not** build a hard Nifty/Ryxliee style classifier or strong reranking weight
from these six features.

They are useful as soft priors and sanity checks, but the held-out result says
they miss much of what makes mapper style recognizable.

The next feature pass should focus on richer rhythm structure:

- Don/Ka n-grams and transition probabilities
- burst-length and rest-length distributions
- repeated 4/8/16-slot motifs
- motif reuse with variation
- section-to-section density dynamics
- subdivision-transition matrices
- roll/balloon placement relative to phrase boundaries
- note placement relative to audio onset/energy changes

The same evaluation protocol should be reused: difficulty normalization, one
vector per mapper/song set, permutation testing, and held-out song-level
validation. A richer feature family only earns production weight if it improves
held-out balanced accuracy without collapsing into song identity.
