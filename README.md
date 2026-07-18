# Project Brief: Synthetic Training Data for DDI Relation Extraction

*University of Glasgow, IDA section. 8-week summer internship (week 1, ~July 2026).*
*Supervisor: Jake Lever. Intern: Gwilym.*

---

Can a large LLM generate a drug–drug interaction training set **from scratch** — with
no human annotation — that trains a BERT classifier to a **useful, honestly benchmarked**
score on the **real DDI-2013 test set**?


This is a matching-and-benchmarking problem, not primarily a generation problem.
Clever prompting and elaborate negative-example taxonomies are not necessarily the point. The point is a trustworthy number: "synthetic data reaches X% of the
human-annotated baseline on the real DDI-2013 test set." 

---

## Background & motivation

Relation extraction (RE) normally needs expert-annotated corpora which is a major bottleneck when starting a new biomedical RE project. If an LLM can bootstrap a training set from just a description of the target relations, that unblocks many downstream projects. 

## The task: DDI-2013

- **Corpus:** DDI-Extraction 2013. Drug–drug interactions. Two registers:
  **DrugBank** (terse drug-label prose) and **MedLine** (academic abstracts).
  Both appear in train and test; report them separately.
- **Formulation (from the tutorial):** relation extraction as text classification.
  For each candidate *pair of drug entities in a sentence*, wrap them in `[E1]..[/E1]`
  and `[E2]..[/E2]` markers and classify the pair.
- **Labels:** 4 positive classes — `ADVISE`, `EFFECT`, `MECHANISM`, `INT` — plus
  `NONE` (no interaction). ~85% of candidate pairs are `NONE`.
- **Headline metric:** micro-F1 over the **four positive classes only**. Never
  include NONE in the headline (it's trivial and inflates the score). Always report
  per-class and per-register breakdowns alongside.
- **`INT` is tiny** (~16 in the tutorial val split). Per-class F1 on it is noise
  until evaluated on the full test set. It is also the class most likely to *benefit*
  from synthetic data.

### Note
At least one DrugBank doc contains duplicated sentence blocks (e.g. TRACRIUM), inflating pair counts. Likely very low count, after simple probe. 

## Core architectural principle

**The trainer and the evaluation set are CONSTANTS. Only the training data varies.**

Every experiment — human, synthetic, mixtures, ablations — is just a different list
of `{text, label}` records fed to one fixed `train_and_eval()` and scored on one fixed
human eval set. This is what makes any two runs comparable. Violate it and comparisons
become meaningless.

- Eval set: built once from human DDI, true (~15:1) negative ratio intact, **never
  downsampled, never synthetic, never touched until final test.**
- Negative downsampling applies to **training only**, and the ratio is itself a
  hyperparameter to sweep (it directly controls the precision/recall trade-off).
- Hold out the official DDI **test** set until final experiments (~week 6). Tune on
  validation only.

---

## Prior art

**1. `Glasgow-AI4BioMed/synthetic_relex`** (GitHub + HF dataset, Jake, ~2024).
LLM-as-*annotator*: Llama 3.3 70B *labelled* real PubTator sentences; distilled into
BiomedBERT. Per the supervisor: a **demo**, never matched to a real task, **never
benchmarked**, uses a pre-API generation path that won't run on the current endpoint.

**2. `~/nfs/synth_data_creation/` nested-relations generator** (`relation_schema.yml`,
`nested_relation_maker.ipynb`, `terms.json`, `nested_relations_dataset.json`).
Michael's work (nested relations from human-annotated data).
Adjacent piece of Jake's larger puzzle.

**Key distinction:** prior work *labelled real sentences*. This project *generates
sentences from scratch*, which means it must **fabricate negatives** — sentences where
drugs co-occur but no interaction is asserted. That is the hard, novel part.

---

## Approach & experimental plan

### Generating structurally-correct data (the critical detail)
DDI classifies *pairs within sentences*, and most negatives are non-interacting pairs
*inside sentences that also contain a positive*. So: prompt the LLM to produce a
sentence with N drugs where a specified subset of pairs interact and the rest do not,
then enumerate **all** pairs as instances (one positive, rest negative). This yields
in-distribution negatives for free and forces the model to attend to the entity markers
rather than sentence-level topic. Emit the **same `{text, label, source, sent_id}`
record format** as the human pipeline so both flow through identical code.

### Negative-example strategies to ablate 
Intra-sentence non-participants (highest value); shared-property distractors
("both metabolised by CYP3A4"); explicit non-interaction ("no interaction was
observed"); comparative; co-medication lists; minimal-perturbation negatives.
Check the DDI annotation guidelines: is a *negated* interaction labelled NONE or a
positive class? This determines whether "explicit non-interaction" negatives are
correct or poison.

### Experiments (rough plan)
1. Human-only baseline (the ruler). Across seeds, with error bars.
2. Synthetic-only, size-matched to human train.
3. Negative-strategy ablation.
4. Scale curve: 1k -> 5k -> 20k -> 100k. Where does it saturate?
5. Zero-shot vs few-shot generation.
6. **Mixing curve**: human {0, 10, 25, 50, 100%} x synthetic.
   Target finding: "N synthetic examples ≈ M human annotations."
7. Best config -> real DDI **test** set, once.

---

## Known bugs in the tutorial code

1. **Marker mismatch:** inserts `[E1]` but tokenizer adds `[Arg1]` — added tokens
   never used, markers shredded into subwords.
2. **`itertools.product` -> both (A,B) and (B,A):** creates a mirrored, mislabelled
   NONE duplicate of every positive. Use `combinations` over offset-sorted entities.
3. **Metric includes NONE:** `compute_metrics` uses macro-F1 over all 5 classes;
   headline must be micro-F1 over the 4 positives.

## Marker-initialisation finding (already learned the hard way)

Naively "fixing" the marker mismatch by adding `[E1]` etc. to the vocab made F1
*worse* (0.69->0.48). Reason: new tokens get ~N(0, 0.02²) random embeddings and, under
a short/low-LR budget, never train enough to be useful — the model goes blind to the
markers. The broken subword version worked because `[`, `e`, `##1`, `]` are all
*pretrained*. **Fix:** add the tokens but initialise each new embedding to the **mean
of its subword pieces**. Sweep `{subword, random, mean}` to confirm. 

Also: the P=0.35/R=0.79 precision collapse seen early on is the train/eval negative-ratio
mismatch (trained ~5:1, evaluated ~15:1) — not a bug, a hyperparameter. Sweep neg_ratio.

---

## Infrastructure (set up, week 1)

- **Compute:** various GPUs via Launcher **ephemeral pods** (2-hour idle timeout,
  container FS wiped on restart). Persistent storage at `/root/nfs`.
- **Persistence:** venv, repo, caches all live on `/root/nfs`. HF cache at
  `/root/nfs/launcher-cache/huggingface` (writable, **already contains BiomedBERT** —
  no download). VS Code server persisted by the platform. `.bashrc` sets `HF_HOME`,
  `PIP_CACHE_DIR`, `IDA_LLM_API_KEY`, activates venv, cd's to repo.
- **LLM endpoint:** `gpt-oss-120b`, OpenAI-compatible API
  (`http://api.llm.apps.os.dcs.gla.ac.uk/v1`, key in `$IDA_LLM_API_KEY`).
  Best throughput ~7 rows/s at 256 concurrent threads. Check for a
  `reasoning: {effort: low}` param — output/input ratio ~2.3x suggests wasted CoT.
- **Editor:** VS Code Remote-SSH into the pod; notebooks run against the pod GPU with
  `ddi/*.py` editable alongside. `%autoreload 2`.
- **Git:** GitHub key set up for pushing run records off the ephemeral pod.
  Uni GitLab key (`~/.ssh/gitlab`) and DCS key (`~/.ssh/id_rsa`) already exist.

---

## Current status (end of week 1 setup)

- [x] Problem understood, prior art found and scoped, supervisor aligned
- [x] Infrastructure: pod persistence, caches, BiomedBERT confirmed cached
- [ ] Harness running
- [ ] Human annotations baseline
- [ ] Synthetic generation (structured, from scratch)
- [ ] Ablations, scale curve, mixing curve

**Next concrete step:** get `build_human()` producing a decent human baseline micro-F1 across
3 seeds.