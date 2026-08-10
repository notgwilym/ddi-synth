# notes: state at 8 August, week 3 of 8

## The question

Can an LLM generate a DDI relation-extraction training set, with no human annotation,
that trains BiomedBERT to a defensible micro-F1 on the real DDI-2013 test set.

What counts as answering it: one number on the held-out test set, obtained under a
protocol fixed in advance, with the human-trained baseline measured under the identical
protocol. Nothing else substitutes for that.

## Results, 8 August, dev set, three seeds

| arm | F1 | P | R | DrugBank | MedLine |
|---|---|---|---|---|---|
| human (26,785) | 0.800 (.005) | 0.768 | 0.835 | 0.804 | 0.699 |
| v13 (19,051) | 0.280 (.003) | 0.182 | 0.610 | 0.288 | 0.184 |
| v14 (18,482) | 0.379 (.007) | 0.302 | 0.511 | 0.384 | 0.310 |

Human reproduces the 0.800 from the masking runs and v13 reproduces 0.286, so the
pipeline has not drifted and both baselines are confirmed before v14 is read.

**v14 gains 0.099 F1, a 35% relative improvement, entirely through precision.** P rises
66% and R falls 16%. That is the signature the composition diagnosis predicted:
indiscriminate firing on small sentences was buying free recall, and removing it costs
that recall back. The mechanism is now confirmed empirically rather than inferred.

My prediction of 0.45 to 0.60 was wrong on the record.

MedLine improved 68% relative against DrugBank's 33%, which is the rewritten register
line paying off. v13's MedLine framings were mostly case reports; real MedLine is
abstract findings prose.

## False positive rate by source-sentence entity count

| bucket | pairs | human | v13 | v14 | v14 recall |
|---|---|---|---|---|---|
| 2 | 237 | 0.147 | 0.692 | 0.580 | 0.809 |
| 3 | 465 | 0.065 | 0.755 | 0.466 | 0.667 |
| 4-5 | 1006 | 0.045 | 0.687 | 0.359 | 0.583 |
| 6-9 | 999 | 0.041 | 0.359 | 0.130 | 0.301 |
| 10-19 | 744 | 0.046 | 0.370 | 0.051 | 0.171 |
| 20+ | 792 | 0.000 | 0.000 | 0.000 | 0.000 |

v14 halves the FPR at every bucket but remains an order of magnitude above human.

**The recall collapse above five entities is coverage, not a shortcut.** v14 caps at 5
entities, so it has never seen a sentence resembling buckets 6 and above, which hold
2,535 of 4,243 dev pairs. Note the contrast with v13, which still fired at 0.36 there
because CONEG produced 10-15 pair sentences; v14 fires at 0.05 because the construction
is out of distribution entirely.

**Fixing coverage buys almost nothing.** If v14 achieved its 4-5 bucket rates across all
higher buckets the result would be P 0.283, R 0.639, F1 0.392. Thirteen thousandths.
The extra recall arrives with proportional false positives because precision is what is
broken.

**In-distribution, restricted to buckets 2 through 5:** v14 P 0.315, R 0.668, F1 0.428
against human P 0.797, R 0.846, F1 0.821. Two false positives for every true one in the
region the generator was designed for. This is the number that matters, and it says
composition was a real constraint but not the binding one.

## Per class, v14 against human

| class | support | human F1 | v14 P | v14 R | v14 F1 |
|---|---|---|---|---|---|
| MECHANISM | 210 | 0.78 | 0.23 | 0.46 | 0.30 |
| EFFECT | 211 | 0.81 | 0.42 | 0.49 | 0.45 |
| ADVISE | 126 | 0.81 | 0.30 | 0.64 | 0.41 |
| INT | 19 | 0.81 | 0.22 | 0.58 | 0.32 |

The failure is uniform. No class collapses, recall is spread 0.46 to 0.64, and INT at
0.32 is comparable to MECHANISM, so the earlier worry about INT being too terse to
generate was unfounded and the INT scope fixes worked.

MECHANISM is worst at P 0.23 against EFFECT's 0.42, and both come from the same
machinery. The likely reading is that v14's MECHANISM content is too canonical: six
sites times two exposure directions, no named enzymes, no induction or inhibition
framing. A model trained on the narrow version over-applies the pattern.

## Partial-input probe

Matched on sentence count (2,429): human 0.152 (sd 0.020), v14 0.216, 0.223, 0.228 over
three subsample draws (sd ~0.01).

The gap of 0.070 is understated. Human was measured on 26,785 instances against v14's
~8,100, and the probe rises with data volume (full human 0.174 at 26k, human subsamples
of 1.6k anywhere from 0.00 to 0.17). Human is being measured at an advantage and still
scores lower.

The probe estimator is stable at full size (sd 0.020) and unusable below a few thousand
instances (sd 0.085, spread 0.22). Any threshold must be calibrated against size-matched
human data. The 0.15 I ran against for two rounds was invented.

## Leading hypothesis for the residual precision gap

v14's non-participants nearly always carry an r1 to r5 role, rendering into sentences as
"given earlier and stopped", "a comparator", "not permitted". Real corpus NONE pairs are
overwhelmingly unmarked co-mentions with nothing signalling non-participation. If the
model learned "role language nearby means NONE" it has learned a cue that does not exist
in the corpus, which would produce exactly the observed pattern: uniform across classes,
present in-distribution, and invisible to the composition gates because the hard-negative
rate is genuinely 0.50.

`P_ROLE` already exists as a parameter, so the test is: set it to 0.2, generate 300, and
check whether role vocabulary drops out of the probe's top features. About an hour end
to end, and a scale run plus retrain if it does.

Second candidate, not exclusive: MECHANISM content is too narrow, per the per-class
split. Adding a `via` enzyme slot and an induction/inhibition mode widens the class where
it costs most.

Third: label noise. The role-position leak affects ~2% of sentences, occasional ADVISE
scope drift supplies a reason the scope forbids, and some sentences assert something
about a drug whose pairs are labelled NONE. Plausibly 5% of labels are wrong, which
hurts but does not explain a factor of two on precision.

## What was established before today, still standing

**Masking.** Replacing every drug name with drug1/drug2 costs human 0.021 F1 and gains
synthetic 0.011. Drug identity contributes little, so vocabulary realism matters for
prose quality and reject rate rather than for classifier performance directly.

**Corpus structure.** Entities per sentence conditional on two or more: 0.41, 0.27, 0.14,
0.07, 0.04. Positives per multi-entity sentence: 0.448 zero, 0.375 one, 0.086 two. Types
DRUG 0.636, GROUP 0.224, BRAND 0.101, DRUG_N 0.038. DrugBank supplies ~94% of pairs.
Sentence-initial anaphora 5.7%. Token gap between interacting pairs: median 10, p90 57.

**Eight sentences carry 43% of training pairs**, from flattened HTML tables and bare
lists, contributing ~47% of all NONE pairs. This is evidence our loader skips
preprocessing the field applies as standard rather than a discovery about the corpus.
Still needs confirming against two or three BERT-era DDI papers, because if standard
preprocessing filters them, 0.8127 is not like-for-like with published figures.

## Critical review

**The gates predicted direction but not magnitude.** Every gate passed on v14 and the
result improved, so they were not worthless, but they cleared a dataset that scores 0.379
and they cannot distinguish that from one scoring 0.6. They are necessary and not
sufficient, and the last four rounds of prompt iteration were tuned against a probe whose
threshold I had invented and whose spread at that sample size was 0.22.

The training run should have happened at the second passing gate. It would have shown
the same thing and saved several days.

**The vocabulary problem remains open** and the masking result says it is lower priority
than it looks: names contribute 0.021 F1 for human, so absurd names cost reject rate and
drift rather than F1 directly.

**Paper framing.** Shortcut learning and partial-input baselines are textbook. The
contribution is not a new failure mode but a demonstration that the standard synthetic
data workflow has no diagnostic step between generation and training, with a worked case
where the defect cost 0.099 F1 and was invisible to aggregate metrics. Whether that is
worth a workshop paper depends on where the final number lands.

## Plan

**Next, in order.**

1. Role-marking test. `P_ROLE=0.2`, 300 specs, check the probe's top features. If role
   vocabulary drops out, regenerate at scale and retrain. Roughly two hours including
   the training run.
2. MECHANISM content widening: `via` enzyme slot, induction/inhibition mode. Cheap, and
   MECHANISM is the worst class by a clear margin.
3. Raise the entity cap to 8 or 10. Worth 0.013 on its own, so do it alongside something
   else rather than as its own iteration.

**Then branch.** If the role test moves precision materially, continue on generation. If
it does not, the residual gap is prose distribution rather than a specific cue, and the
honest next step is to characterise it rather than keep iterating: sample fifty v14 false
positives and read them against the sentences that produced them.

**Held for the writeup regardless.** Scale curve by subsampling the existing 6,000
sentence run. Mixing curve against human data. One sealed-test evaluation with the
protocol fixed in writing beforehand.

**Do not.** More prompt iteration without a training run to validate it. The verifier,
which is deliberately off the critical path. Enumeration generation, since the P gap is
not concentrated in large buckets. ChemProt transfer.

## Open decisions for Jake

Corpus statistics as design input. Composition, entity counts and label distribution come
from the training split, which is weak supervision. A PRIOR preset exists as the
hand-specified control arm; running both is one keyword.

`N_POSITIVES_BY_K` is hand-tuned to decorrelate positive rate from entity count and sits
in neither preset. Recorded in the manifest as such.

Few-shot exemplars. Twenty gold instances per class is eighty human annotations and
converts the claim from "no human annotation" to few-shot-guided generation, which is a
different and much better-populated setting. The DDI-2013 annotation guidelines contain
worked examples and are a published document rather than annotation labour, so those are
the defensible version. Either way it is testable as an arm: 300 with, 300 without,
compare distinct-4 and label fidelity.

Benchmark comparability, per the eight monster sentences above.

## Known defects in the current v14 run

Role-position instruction leaks into ~2% of sentences ("...these come after the main
statement"). Fold it into the block header.

Reject rate 5-7%, dominated by dropped GROUP names.

Occasional ADVISE scope drift supplying a reason the scope forbids.

Entity cap of 5, against a dev distribution where 60% of pairs come from sentences with
six or more.