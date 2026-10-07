# ddi-synth

Code, run records, a report and linked tutorial for an EPSRC vacation internship in the AI4BioMed Lab, University of Glasgow (summer 2026, supervised by Jake Lever).

The project asks how a large language model is best used to produce training data for drug-drug interaction (DDI) extraction: by writing labelled sentences, or by labelling real ones. A BiomedBERT classifier is fine-tuned on each kind of data and scored on DDI-2013.

## Result

For this task, an LLM is better spent labelling and checking real text than writing new text.

Test scores, micro-F1 over the four positive labels, five seeds each:

| Training data | Human-labelled sentences | F1 | Share of full human supervision |
|---|--:|--:|--:|
| Generated sentences (best generator) | 0 | 0.528 | 66% |
| Real sentences, LLM labels | 0 | 0.680 | 84% |
| Real sentences, LLM labels checked by a second LLM pass | 0 | 0.728 | 90% |
| Human labels | 1,000 | 0.745 | 92% |
| Human labels plus checked LLM labels on the rest | 1,000 | 0.773 | 96% |
| Human labels | 2,402 (all) | 0.806 | 100% |


## Tutorial

Alongside the report there is a tutorial, `tutorial/ddi_tutorial.ipynb`, that walks through the analyses behind the report on samples of the project's own data. No GPU or API access needed.

```bash
git clone https://github.com/notgwilym/ddi-synth.git
cd ddi-synth/tutorial
jupyter notebook ddi_tutorial.ipynb
```

The first cell installs its dependencies, downloads the spaCy model and fetches the DDI-2013 corpus into the repository root. Should work with Python 3.10 and 3.12.

## Repository

| Path | Contents |
|---|---|
| `ddi/` | the pipeline: corpus loading and pair construction, generator prompts and specifications, labelling and checking, training and evaluation |
| `scripts/` | entry points for generation, labelling, the validation grid, the test run and the report's checks |
| `tutorial/` | the tutorial notebook, its helper module and the data samples it reads |
| `runs/` | one JSON record per training run, with a diff of the code it ran against |
| `reports/` | the data split, the validation grid, the test selection and results, and `experiment_log.html`, a rendered record of every run |
| `datasets/` | dataset manifests and the drug vocabularies (DrugBank names, WHO-ATC groups) |
| `notes/`, `logs/` | project notes and console logs |

## Running the full pipeline

Reproducing the experiments, rather than the tutorial, needs a GPU for BiomedBERT and an OpenAI-compatible endpoint serving `gpt-oss-120b`:

```bash
pip install -r requirements.txt
export IDA_LLM_BASE_URL=...   # endpoint URL
export IDA_LLM_API_KEY=...
```

`requirements.txt` pins the versions used for the reported runs (Python 3.10).

## Licence and data

The code is released under the GNU General Public License (GPL) v3 (see `LICENSE`).

The DDI-2013 corpus (Herrero-Zazo et al., 2013) is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International licence](https://creativecommons.org/licenses/by-nc/4.0/). Files in this repository that contain corpus sentences, including `tutorial/samples/labelled.jsonl` and `runs/frames/`, are derived from it and are under the same licence: attribution required, non-commercial use only. The derived files re-split the corpus's training directory and record one label per pair of drug mentions. The corpus itself is not included and is downloaded at run time.

> Herrero-Zazo, M., Segura-Bedmar, I., Martínez, P. and Declerck, T. (2013). The DDI corpus: an annotated corpus with pharmacological substances and drug-drug interactions. *Journal of Biomedical Informatics*, 46(5), 914–920.

Generated sentences were written by `gpt-oss-120b` from specifications in this repository.

## Acknowledgements

Funded by University of Glasgow SoCS Internship. Supervised by Jake Lever, AI4BioMed Lab, School of Computing Science, University of Glasgow.
