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

### Generating structurally correct data

Prompt the LLM with N drugs and one specified interacting pair, then enumerate all
pairs programmatically. Pairs not listed in the relations fall through to NONE. This
gives in-distribution negatives for free and forces the classifier to attend to the
entity markers rather than sentence topic.

The negative ratio falls out of the drug count. With one positive relation and N drugs
there are C(N,2) pairs, so N=3 gives 67% NONE, N=4 gives 83%, N=5 gives 90%. Current
distribution {3: 0.25, 4: 0.45, 5: 0.2} lands around 85%, matching the real corpus
without any downsampling.

Labels are an input, not an output. The generator is told which class to write. This
gives direct control over class balance, which matters for INT.

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

### First synthetic-only run (v6, 2000 specs, gpt-oss-120b)

micro-F1 0.3158, P 0.5431, R 0.2226. Train size 13968 instances.

Per class: ADVISE 0.421, INT 0.414, EFFECT 0.287, MECHANISM 0.264.
Per register: DrugBank 0.316, MedLine 0.308.

That is 39% of the human baseline, untuned, first attempt.

The interesting part is the shape. Precision 0.54 against recall 0.22 is the opposite
of the human baseline. The model is reasonably right when it fires but barely fires.
That points at the negatives rather than the positives.

Leading hypothesis: formulaic negation. Almost every generated sentence appends
something like "whereas X and Y showed no observable interaction". Real DDI negatives
are usually just drugs co-occurring in a list with no comment at all. The classifier
may have learned that explicit disclaimer language means NONE, and defaults to NONE
when that cue is absent on real text. Fix is in the prompt: stop the model appending a
non-interaction clause.

INT scoring second best on 19 support is encouraging for the class-control argument.

Registers scoring almost identically suggests synthetic transfers evenly rather than
favouring the DrugBank style it superficially imitates.

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
- `max_output_tokens` must be generous, around 3000. Reasoning tokens count against
  that budget, so a tight cap truncates the JSON and returns status=incomplete. Set it
  to 600 at one point and 70% of the batch failed.
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

`terms.json` is a chemistry synonym dump, about 262k chemical terms plus genes and
diseases. After regex filtering (identifiers, InChIKeys, UNII codes, ATC codes, CAS
numbers, molecular formulae, dyes, reference standards, mojibake, consumer products)
roughly 200k remain, but most are still industrial chemicals, solvents and dyes rather
than drugs. No regex separates "Dinonyl adipate" from "Clioquinol Impurity 6".

Kept as-is for now, deliberately. The classifier only sees the marked span, and varied
unpredictable entity strings arguably prevent lexical shortcutting. An LLM filter pass
exists in `scripts/filter_vocab_llm.py` if a cleaner lexicon is wanted. "Clean lexicon
vs raw chemical pool" is a cheap one-line ablation.

Drug class terms (NSAIDs, corticosteroids, MAO inhibitors etc.) are a hand-written list
in `vocab.py`, since terms.json is thin on them and they are a large slice of real DDI
entities. This is a hand-curated artefact, which weakens the portability claim
slightly. Worth a sentence in the writeup. Bootstrapping the list from the task
description via the LLM is straightforward future work.

Genes and diseases are not DDI entities and are not used as entities. They are
available as context terms the model may mention but must not annotate.

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

Label fidelity is the biggest unknown. Because the label is an input, the model is
anchored: told to write EFFECT, it will label its output EFFECT more or less
regardless of what it actually wrote. Agreement between requested and emitted label
measures compliance, not correctness, so it is nearly useless as a quality signal.
Seen at least once where a sentence used clear mechanism language ("altered the plasma
concentration of") while carrying an INT label.

Fix: blind relabelling. Take generated sentences with markers inserted, and in a fresh
call with no mention of what was requested, ask the model to classify the pair.
Agreement between requested and blind label is a real fidelity number. A few hundred
per config is enough. Plus a manual spot check of about 50 against the guidelines,
which is the only ground truth available and calibrates whether the blind relabeller
can be trusted.

If fidelity turns out to be, say, 70%, that is not a failure. It explains the gap to
the human baseline and makes the final number interpretable.

---

## Next steps

- [x] Problem understood, prior art scoped, supervisor aligned
- [x] Infrastructure: pod persistence, caches, BiomedBERT cached
- [x] Harness running
- [x] Human baseline on the 70/15/15 split
- [x] Synthetic generation pipeline, structured, from scratch
- [x] First synthetic-only run
- [ ] Fix formulaic negation in the prompt, rerun, see if recall moves
- [ ] Blind relabel validator plus manual spot check
- [ ] Leakage gate (exact-match n-gram overlap against dev/val/test, written back into
      the manifest, which already has a `leakage_report` field reserved)
- [ ] Negative strategy ablation
- [ ] Scale curve: 1k, 5k, 20k, 100k. Where does it saturate?
- [ ] Mixing curve: human {0, 10, 25, 50, 100%} against synthetic. Target finding: N
      synthetic examples is worth M human annotations.
- [ ] Zero-shot vs few-shot generation
- [ ] Vocab ablation: LLM-filtered lexicon vs raw chemical pool
- [ ] Generator model ablation: 120b vs 20b, given the non-participant finding
- [ ] Best config against the real DDI test set, once, around week 6