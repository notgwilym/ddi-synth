# Project Notes: Synthetic Training Data for DDI Relation Extraction

University of Glasgow, IDA section. 8-week summer internship (week 2, July 2026).
Supervisor: Jake Lever. Intern: Gwilym.

---

Can a large LLM generate a drug-drug interaction training set from scratch, with no
human annotation, that trains a BERT classifier to a useful, honestly benchmarked
score on the real DDI-2013 test set?

This is a matching-and-benchmarking problem, not primarily a generation problem.
Clever prompting and elaborate negative-example taxonomies are not necessarily the
point. The point is a trustworthy number: "synthetic data reaches X% of the
human-annotated baseline on the real DDI-2013 test set."

---

## Background and motivation

Relation extraction normally needs expert-annotated corpora, which is a major
bottleneck when starting a new biomedical RE project. If an LLM can bootstrap a
training set from just a description of the target relations, that unblocks many
downstream projects.

## The task: DDI-2013

- Corpus: DDI-Extraction 2013. Two registers: DrugBank (terse drug-label prose) and
  MedLine (academic abstracts). Both appear in train and test, report separately.
- Formulation: relation extraction as text classification. For each candidate pair of
  drug entities in a sentence, wrap them in `[E1]..[/E1]` and `[E2]..[/E2]` markers
  and classify the pair.
- Labels: four positive classes (ADVISE, EFFECT, MECHANISM, INT) plus NONE. About 85%
  of candidate pairs are NONE.
- Headline metric: micro-F1 over the four positive classes only. Never include NONE in
  the headline, it is trivial and inflates the score. Always report per-class and
  per-register breakdowns alongside.
- INT is tiny. Per-class F1 on it is noisy until the full test set. It is also the
  class most likely to benefit from synthetic data, and label-as-input generation
  gives direct control over how much of it we make.

### Corpus quirk
At least one DrugBank doc contains duplicated sentence blocks (e.g. TRACRIUM),
inflating pair counts. Low count, confirmed by probe.

## Annotation rules that matter

Transcribed the official guidelines PDF (its text layer is broken, had to read it
visually). The rules that actually constrain generation:

- Negated interactions are NOT annotated (4.5.1). "X and Y do not interact" is NONE.
  So explicit non-interaction negatives are correct, not poison. This was an open
  question in week 1.
- If a sentence both affirms and negates an interaction, the affirmation wins (4.5.4).
- Sentences that merely report a study of an interaction, without confirming it, are
  not annotated (4.5.5).
- Speculative interactions ARE annotated regardless of certainty (4.5.2). "may
  interact", "suggests a possible interaction" all count.
- When an interaction fits several types, priority order is mechanism, then effect,
  then advise (4.5.11).
- Type definitions: mechanism is pharmacokinetic (absorption, distribution,
  metabolism, excretion, levels, clearance, AUC, half-life). Effect is a
  pharmacological effect, clinical finding, or pharmacodynamic mechanism. Advise is a
  recommendation about concomitant use. INT is an interaction asserted with no further
  detail.
- Not annotated as entities: enzymes (CYP3A4, P-glycoprotein), doses, dosage forms,
  routes of administration, foods and drinks.

## Core architectural principle

The trainer and the evaluation set are constant. Only the training data varies.

- Eval sets built once from human DDI, true negative ratio intact, never downsampled,
  never synthetic.
- Document-level split so pairs from one sentence cannot straddle splits.
- Dev is for iterating, val is for choosing between finalists, test is touched once
  around week 6.
- Negative downsampling applies to training only, and the ratio is a hyperparameter.

---

## Prior art

1. `Glasgow-AI4BioMed/synthetic_relex` (Jake, ~2024). LLM-as-annotator: Llama 3.3 70B
   labelled real PubTator sentences, distilled into BiomedBERT. Demo work. Worth
   revisiting as a label validator rather than a generator, see open questions.

2. `~/nfs/synth_data_creation/` nested-relations generator. Michael's work, nested
   relations from human-annotated data. Adjacent piece of Jake's larger puzzle.

3. MetaSynth (Riaz et al. 2025, arXiv 2504.12563). Meta-prompting with agent scaffolds
   for diverse synthetic data. Aimed at continual pre-training so it does not transfer
   directly, and the agentic scaffold costs about 3.6 minutes per document which would
   kill the scale curve. Two things worth taking: the finding that template prompting
   collapses to low diversity even with prior generations in context, and their
   contamination check (exact-match n-gram overlap at N = 1,2,3,5,10) which is the
   right tool for a leakage gate. Also calibrating: they fine-tuned BERT on synthetic
   vs real for three classification datasets and synthetic stayed behind real. Expect
   the same here.

Key distinction from prior work: this generates sentences from scratch, so it must
fabricate negatives. That is the hard, novel part.

---

## Approach

### Generating structurally correct data (current: clinical-vignette generation)

SUPERSEDED the original single-sentence flat-enumeration design (kept below as history).
The single-sentence approach produced template collapse ("A decreases the plasma
concentration of B" repeated) and, worse, STATED negatives ("whereas X and Y showed no
observable interaction") appended to every passage. Real DDI negatives are silent
co-mentions; the stated form is off-distribution and taught the classifier a spurious
cue. This was the main driver of the early precision/recall pathology.

Current design: multi-sentence clinical vignettes. The model is given a patient/document
framing, a drug pool, and target interaction TYPES (not named pairs), and writes a 2-4
sentence passage as a list of sentences, each with its own relations. Key properties:
- Non-interacting drugs get distinct clinical roles (comparator, background therapy,
  prior medication) woven into the narrative, not dumped in a trailing list. This
  produces natural, unstated, in-distribution negatives.
- Relations are declared PER SENTENCE (the model says which sentence each interaction is
  in), so no cross-sentence inference and no mention-id juggling.
- The model chooses which pool drugs realise each target type, so it picks plausible
  pairs instead of being forced to assert a false interaction.
- Realism of the interaction itself is explicitly NOT required (told not to deliberate);
  this keeps generation fast and, usefully, keeps novelty up (see leakage note). Known
  side effect: some generated mechanisms are chemically absurd (a monoclonal antibody
  "inhibiting hepatic metabolism"). Linguistically correct, chemically nonsense. Open
  design tension to raise with Jake, not yet resolved.

### Sentence-scoped parser (current)

`sample_to_instances` splits the passage into the model's declared sentences, resolves
entity spans within each, and enumerates pairs per sentence via the shared
`make_pair_instances`. Cross-sentence pairs are dropped by the same rule as the human
pipeline. Deliberate divergences from the human path, all synthetic-only:
- ONE INSTANCE PER NAME-PAIR per sentence (collapse repeated mentions to first
  occurrence). Human pipeline is mention-level (gold gives exact ids); synthetic
  relations are name-level, so name-keyed labelling + collapse avoids the
  self-pair junk and the same-pair-two-labels contradiction that mention-level
  enumeration produced on repeated-drug sentences.
- Self-name pairs (drug vs another mention of itself) are dropped.
- Relations whose two drugs land in different sentences are reject-and-logged, not
  silently dropped.
- Synthetic path does NOT use spaCy (model declares its own sentence boundaries);
  human path still uses spaCy. Deliberate.

Reject rate on the current pipeline is ~2% (v11). Residual rejects are model-side:
entity-string infidelity (singular/plural drift, spacing corruption), cross-sentence
mis-scoping, and substring collisions in `.find`-based span resolution. Acceptable;
tightening to <1% is v2.

### Class balance

The negative ratio falls out of the drug count and the vignette structure. IMPORTANT
FINDING (see Results): the vignette generator STRUCTURALLY under-produces NONE, because
each non-interacting drug tends to get its own sentence (one drug per sentence = zero
pairs = zero NONE). A dedicated co-mention negative generator (`CONEG`, empty relations,
several drugs per sentence) exists to top up NONE. But the mixing curve showed NONE
ratio is NOT the bottleneck, so this is now mainly a knob for matching dev, not a fix.

Labels are an input, not an output. The generator is told which class to write, giving
direct control over class balance. Positive-class composition can be matched to dev via
`label_dist` (tested: didn't move F1, see Results).

### Pipeline shape

Two stages, deliberately separated:

- Stage 1 (`generate_raw`) is expensive and non-deterministic. Calls the API
  concurrently and appends raw model output to `raw/<gen_id>.jsonl` as it arrives,
  flushing every line. A dead pod costs only the in-flight requests. Resume skips
  specs that already succeeded and retries ones that errored.
- Stage 2 (`build_dataset_from_raw`) is cheap and deterministic. Span resolution, pair
  enumeration, validation, manifest. Free to re-run whenever the resolver changes, no
  API calls.

Synthetic data flows through the same `make_pair_instances` as the human pipeline, via
shim objects that mimic the BratDocument interface. Marker insertion and pair
enumeration are the same tested code for both.

The model emits entity mentions without character offsets. Offsets are computed by
occurrence counting, because models cannot count characters reliably. Matching falls
back to case-insensitive, since the model routinely capitalises a name at the start of
a sentence.

### Provenance

Every dataset gets a manifest with a sha256 of the instances file, the generator
config, the vocab fingerprint, and the label distribution. Every training run records
`train_id` and `eval_id`. `run_training` refuses to evaluate on a synthetic set.

---

## Results so far

### Human baseline (70/15/15 document-level split, seed 42)

Winning config from the week 1 grid: random marker init, no negative downsampling,
3 epochs. Three seeds:

micro-F1 0.8127 +/- 0.0075, P 0.7928 +/- 0.0267, R 0.8345 +/- 0.0150

Recall sits above precision, and precision carries most of the seed-to-seed spread.
The model finds most real interactions but over-fires slightly.

Note: the earlier 0.855 figure was on the old 80/20 split and included the marker_init
axis, which has since been removed from the code (random won, difference was small).
Not comparable, do not cite it.

### Old v6 run (single-sentence pipeline) - HISTORY

v6 (single-sentence, stated-negatives, terms.json vocab): micro-F1 0.3158, P 0.54,
R 0.22. Precision-high/recall-low. Hypothesis at the time was formulaic negation, and
the vignette rebuild fixed the stated-negatives problem. But note the newer runs invert
the shape (recall-high, precision-low), so the story moved - see below. Keeping v6 only
as the pre-rebuild reference point.

### Current synthetic-only runs (vignette pipeline, real vocab)

v11 (uniform positive mix): micro-F1 0.247, P 0.15, R 0.62. Recall fine, precision is
the floor - the classifier over-fires on interactions. This is the shape that all
subsequent runs share.

Three controlled experiments to localise the gap:

1. MIXING CURVE (NONE ratio 0.30 -> 0.85, positives fixed): F1 flat 0.254 -> 0.277,
   precision flat ~0.16-0.19. => Class balance is NOT the bottleneck. The 13%-vs-87%
   NONE inversion hypothesis is FALSE. (Vignette generator under-produces NONE; topped
   up with the CONEG generator, but it doesn't help F1.)

2. POSITIVE-COMPOSITION MATCH (v13, dev proportions MECH .37/EFFECT .37/ADVISE .22/
   INT .03, vs the earlier uniform 1:1:1:1): micro-F1 0.286, essentially unchanged.
   => Positive-class distribution is NOT the bottleneck. INT F1 did ~double
   (0.077 -> 0.144) from de-dilution, but n=19 in dev so it barely moves micro.

3. SIZE-MATCHED HUMAN CONTROL (decisive). Human train subsampled to synthetic scale at
   0.85 NONE, same eval:

   | config | n | F1 | P | R |
   |---|---|---|---|---|
   | synthetic v13 @0.85 | ~19k | 0.286 | 0.186 | 0.613 |
   | human pos=300 @0.85 | 2000 | 0.460 | 0.416 | 0.514 |
   | human pos=800 @0.85 | 5333 | 0.653 | 0.568 | 0.769 |
   | human pos=1500 @0.85 | 10000 | 0.740 | 0.667 | 0.830 |
   | human pos=2573 @0.85 | 17153 | 0.789 | 0.735 | 0.852 |
   | human FULL (natural) | 26785 | 0.808 | 0.761 | 0.860 |

CONCLUSION. At matched size and NONE ratio, human 0.789 vs synthetic 0.286 - the ~0.50
gap is entirely SOURCE QUALITY, not quantity or distribution. Human reaches 0.46 with
just 300 real positives; synthetic has ~2933 positives and scores 0.29. Precision is
the failure mode everywhere: human precision climbs 0.42 -> 0.74 across the sweep,
synthetic is stuck ~0.19 regardless of anything tried. Synthetic examples are
individually far less informative than real ones. The problem is distribution OVERLAP
(synthetic positives/negatives don't teach a boundary that holds on real text), not
distribution proportion.

Key figure for the writeup: F1 vs positive count (human learning curve) with the
synthetic point sitting far below the curve. Makes the quality gap visual.

Per-class pattern (consistent across runs): recall fine (0.42-0.65), precision the
floor (0.09-0.29). EFFECT strongest (~0.40), MECHANISM weakest of the real three
(~0.22), INT worst and structurally so (defined by absence of detail, hard to generate,
n=19 noisy). Register split: DrugBank transfers ~2x better than MedLine (0.26 vs 0.13).

---

## Infrastructure

- Compute: GPUs via Launcher ephemeral pods, 2-hour idle timeout, container FS wiped
  on restart. Persistent storage at `/root/nfs`. `DDI_DATA_ROOT` points there so raw
  generations and instance files survive.
- LLM endpoint: OpenAI-compatible at `http://api.llm.apps.os.dcs.gla.ac.uk/v1`, key in
  `$IDA_LLM_API_KEY`. Structured output works via `client.responses.parse` with a
  Pydantic `text_format`.
- Models: gpt-oss-120b for real runs, gpt-oss-20b for pipeline debugging when 120b is
  asleep and slow to start.
- `max_output_tokens` must be generous - currently 4000. Reasoning tokens count against
  that budget, so a tight cap truncates the JSON and returns status=incomplete. Learned
  twice: 600 -> 70% failed; and 3000 at reasoning=high on large multi-word-name pools
  -> incomplete truncations. Generation uses reasoning=low (it's a writing task); the
  VERIFIER uses reasoning=high (a judgement task - see Open questions).
- The Responses API route (`responses.parse`) is WORKER-SPECIFIC. Some workers 404 on
  it (llama-3-8b did) while gpt-oss-120b serves it. Ping `responses.parse` on the exact
  model before a big run. gpt-oss endpoint is a single shared H100, cold-starts on idle,
  and can be unschedulable under contention ("resource not available").
- Batch API: not worth pursuing. vLLM's batch support is mainly offline via `run_batch`
  over a JSONL file, which needs server-side access we do not have. The OpenAI Batch
  API exists for cost discounts and rate limits, neither of which apply on a
  self-hosted cluster, and it trades away latency. vLLM already does continuous
  batching internally, so concurrent requests are already batched at the engine level.
  Effort is better spent on `reasoning_effort` and worker count.
- Editor: VS Code Remote-SSH into the pod, notebooks against the pod GPU with
  `ddi/*.py` editable alongside, `%autoreload 2`.

---

## Vocabulary

CURRENT: DrugBank + WHO-ATC (`datasets/other/DrugBank.csv`, `WHO-ATC-DDD.csv`).
Replaced the old `terms.json` chemical-synonym dump (reagents, file paths, industrial
chemicals) which was badly off-distribution from DDI-2013's real drug entities.

Three-tier build in `build_vocab`:
- Drugs: DrugBank common names (cap-at-one canonical name per drug) + ATC 7-char substance leaves.
- Groups: ATC 4-5 char subgroup names (quinolones, beta blocking agents, etc.). This
  closed a real gap - DDI-2013 annotates group entities heavily and the old vocab had
  almost none. ~0.3 group sampling (approximate prior from guidelines, NOT measured
  from corpus data, to keep the from-scratch claim clean).
- Anatomical top-level ATC codes (1-3 char) dropped.

Filtering: base filter (codes, formulae, >6 words, 2+ digits) + admin-token stoplist
(combinations, other, products, substances, reagents, equipment...) + non-drug stoplist
(crab, pollen, vaccine, spp...). Group word-cap of 3 trades away some legit long class
names for cleanliness. Final: ~14.9k drugs, ~425 groups, all reading as real
drugs/classes. Provenance (sources + p_group + hash) recorded in the vocab fingerprint.

Residual noise (few, low priority v2): endogenous compounds and foods leak through
(Avocado, N-Acetylglucosamine); a DrugBank type/category filter would remove the class.

NOTE for leakage: real drug names raise n-gram overlap with the real corpus vs the old
junk vocab. That is expected vocabulary overlap, not memorised phrasing (read the N=8
tail, not N=1). Re-run the leakage gate on the real-vocab data before any external
result. Leakage gate was deleted at one point as "pointless on junk vocab" - it is NOT
pointless on real vocab, re-add before showing a number externally.

---

## Generation quality notes

Rejection rate went from 28% to 2% over a few iterations. What caused rejections:

- Unicode typography. The model writes non-breaking hyphens and narrow spaces in the
  sentence while emitting plain ASCII in the entity list, so exact matching fails.
  Normalising both sides fixed most of the 28%.
- Markdown bold around entity names.
- Decoding degeneration, "mandatory mandatory mandatory..." for hundreds of tokens.
  Caught by a repetition detector in stage 2.
- Case mismatch between entity text and sentence.
- Duplicate entity ids, usually every entity given the literal id "text".

Do not add a positional fallback for relation arguments. The model uses 0-based and
1-based indexing inconsistently, and guessing wrong silently mislabels the pair.
Text-based fallback is unambiguous and safe.

Prompt lessons:

- Imperative phrasing in the rules ("must appear", "must not interact") made the model
  echo "mandatory" and "mandates" obsessively into the sentences. Declarative phrasing
  reduced it a lot.
- The model will write the literal label name into the sentence ("a clear MECHANISM
  interaction") unless told not to. That would let the classifier read the label word
  directly, score well on synthetic, and collapse on real text.
- 120b follows the non-participant rule much better than 20b. On the same specs, 20b
  wrote things like "must not be combined with salicylates, anesthetics, or X",
  asserting interactions between drugs that then get labelled NONE. 120b did not.
  Model size matters specifically for negative-label honesty.

---

## Open questions

Label fidelity is now the LEADING candidate for the quality gap. Because the label is
an input, the generator is anchored - told to write EFFECT it labels its output EFFECT
regardless. So requested-vs-emitted agreement measures compliance, not correctness.

The verifier (blind relabeller) is BUILT and CALIBRATED. Fresh call, no sight of the
requested label, given the marked sentence + the 4.5.x guideline definitions, returns a
label. Key results:
- At reasoning_effort=low it massively over-called interactions (NONE recall 0.45-0.48)
  - it reads interactions into everything. Adding rules/definitions did NOT fix this.
- At reasoning_effort=HIGH: NONE recall 0.81, overall accuracy 0.82 on human dev vs
  gold. The reasoning step (does this sentence actually ASSERT an interaction between
  THESE two) is what it was missing at low. Reasoning effort was the lever, not the
  prompt.
- Self-calibration: hand-labelled 30 dev instances blind vs gold, 27-28/30. Misses only
  on genuinely ambiguous boundaries (MECH/EFFECT, INT/NONE), no directional bias. So
  manual adjudication of the verifier's disagreements is trustworthy. (Also: the task is
  a LINGUISTIC judgement not a pharmacological one - no biomed expertise needed, the
  guidelines resolve it. Confirmed on the hard cases.)
- Verifier is weak on INT (recall 0.58-0.63), structurally - INT is defined by ABSENCE
  of detail, and a careful reasoner tends to find detail.

THE verifier experiment (next session): does verifier-pruning improve downstream F1?
Verify v13 positives at reasoning=high, keep agreements, retrain, compare to the 0.286
unpruned baseline. Three-arm to be rigorous: pruned / unpruned / random-equal-size. The
bar is downstream F1 vs random-equal-size (controls for "less data"), NOT annotator
quality metrics. Pre-commit to the reading: if pruned ~= raw, the positives were already
faithful and the gap is TEXT realism (templating), not label fidelity - which points at
de-templating the positives instead.

---

## Next steps

- [x] Problem understood, prior art scoped, supervisor aligned
- [x] Infrastructure, harness, human baseline (0.8127)
- [x] Generation pipeline rebuilt: vignette generation, sentence-scoped parser,
      one-instance-per-name-pair, real DrugBank+ATC vocab
- [x] CONEG negative generator (natural co-mention NONE)
- [x] Verifier built + calibrated (0.82 @ reasoning=high)
- [x] Mixing curve -> class balance ruled out
- [x] Positive-composition match -> positive distribution ruled out
- [x] Size-matched human control -> gap is per-example QUALITY, not quantity/distribution

NEXT (priority order):
- [ ] VERIFIER-PRUNING EXPERIMENT (the main one). Three-arm: pruned / unpruned /
      random-equal-size, bar = downstream F1 vs random-equal-size. Tests label-fidelity
      hypothesis against the 0.286 baseline.
- [ ] If pruning doesn't help: de-template the positives (prompt diversity). Positives
      are formulaic ("Co-administration of X with Y resulted in..."); real DDI text is
      varied. Templated positives -> classifier learns template not relation -> overfires.
- [ ] Re-add the leakage gate (deleted; now needed on real vocab - see Vocabulary note).
- [ ] ChemProt second-corpus replication (pre-empts the single-corpus reviewer objection;
      now realistic given the pipeline is stable).
- [ ] Best config against the real DDI TEST set, once, ~week 6. Test still sealed.

RAISE WITH JAKE:
- Paper viability + venue/deadline (BioNLP workshop at ACL likely) + authorship.
- The chemically-absurd-mechanism tension (realism vs novelty/leakage).
- Shared-H100 availability (was down ~40h once; a real constraint on a timeboxed project).

DEPRIORITISED (were in the old plan, now lower value given the quality finding):
- Scale curve - the human control shows quantity isn't the issue, so scaling synthetic
  won't close the gap. Only worth it to show synthetic saturation, not as a fix.
- Zero-shot vs few-shot, 120b-vs-20b generator ablation, vocab ablation - all secondary
  to the quality/fidelity question now.