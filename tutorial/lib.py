"""Plumbing for the tutorial notebook.

The rule for what lives here rather than in the notebook: if a reader would copy it
unchanged into their own project, it belongs here; if it encodes an idea about why
synthetic relation data fails, it belongs in the notebook where it can be read. So this
file loads data, flattens records and trains models, and the notebook computes every
diagnostic itself.

Every record, synthetic or human, has the same shape, so any diagnostic written once
works on all six datasets:

    {"version", "sent_id", "sentence", "n_entities", "pairs": [{"text", "label"}], "spec"}

`spec` is the generation spec for synthetic sentences and None for human ones.
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SAMPLES = HERE / "samples"
VERSIONS = ["v13", "v14", "v15", "v17", "v18"]
LABELS = ["NONE", "MECHANISM", "EFFECT", "ADVISE", "INT"]

_MARK = re.compile(r"\[/?E[12]\]")
_SPAN = re.compile(r"\[(E[12])\](.*?)\[/\1\]", re.S)


def load_samples(version):
    """One generator version's sample, as sentence records."""
    path = SAMPLES / f"{version}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python tutorial/make_samples.py` on a machine that "
            f"has the full datasets, or pull the committed samples.")
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def sample_manifest():
    """Which versions kept their generation specs, and whether their prompts can still
    be re-rendered from the current code."""
    return json.loads((SAMPLES / "manifest.json").read_text())


def human_sentences(split="train", n=None, seed=0):
    """Real corpus sentences in the same record shape as the synthetic samples.

    Built with the project's own `build_human`, which makes one instance per unordered
    pair. That matters: the earlier sentence-level tutorial makes one per ordered pair,
    so mixing the two builders silently doubles the human pair counts.
    """
    import ddi.data as ddi_data
    # CORPUS is relative to the working directory, and the notebook runs from tutorial/.
    # load_brat_docs reads it at call time, so pointing it at the repo root here is enough.
    ddi_data.CORPUS = str(REPO / "DDICorpusBrat")
    from ddi.data import build_human
    from ddi.verify_binary import group_by_sentence, _sentence_spans, _strip

    train, dev, val = build_human()
    rows = {"train": train, "dev": dev, "val": val}[split]
    grouped = group_by_sentence(rows)
    ids = sorted(grouped)
    if n is not None:
        ids = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    return [{"version": f"human-{split}", "sent_id": sid,
             "sentence": _strip(grouped[sid][0]["text"]),
             "n_entities": len(_sentence_spans(grouped[sid])),
             "pairs": [{"text": r["text"], "label": r["label"]} for r in grouped[sid]],
             "spec": None}
            for sid in ids]


def results():
    """Logged F1 for every arm the notebook quotes, recomputed from runs/*.json."""
    return json.loads((HERE / "results.json").read_text())


def flatten(records):
    """Sentence records to the pair instances the classifier trains on."""
    return [{"sent_id": r["sent_id"], "text": p["text"], "label": p["label"]}
            for r in records for p in r["pairs"]]


def sentence_label(record):
    """POS if any pair in the sentence is positive. The diagnostics work at sentence
    level because that is the level a generator makes its choices at."""
    return "POS" if any(p["label"] != "NONE" for p in record["pairs"]) else "NONE"


def masked(record):
    """The sentence with every drug mention replaced by DRUG, recovered from the marked
    spans in its pairs. Used wherever a diagnostic must not be able to see drug names."""
    text = record["sentence"]
    spans = set()
    for p in record["pairs"]:
        stripped, pos = "", 0
        for m in _SPAN.finditer(p["text"]):
            pre = _MARK.sub("", p["text"][pos:m.start()])
            stripped += pre
            start = len(stripped)
            stripped += m.group(2)
            spans.add((start, start + len(m.group(2))))
            pos = m.end()
    # Merge overlaps before replacing. Brat allows nested and discontinuous entities, and
    # replacing overlapping spans one at a time works only by luck of the order.
    merged = []
    for b, e in sorted(spans):
        if merged and b < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((b, e))
    for b, e in reversed(merged):
        text = text[:b] + "DRUG" + text[e:]
    return text


class GuessGame:
    """Show masked sentences, hide their labels, score the reader's guesses.

    Plumbing only. The point of the game is what the reader notices while playing it.
    """

    def __init__(self, records, n=10, seed=0):
        rng = random.Random(seed)
        pos = [r for r in records if sentence_label(r) == "POS"]
        neg = [r for r in records if sentence_label(r) == "NONE"]
        k = n // 2
        self.items = rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(n - k, len(neg)))
        rng.shuffle(self.items)

    def show(self):
        for i, r in enumerate(self.items):
            print(f"{i:>2}.  {masked(r)}\n")

    def score(self, guesses):
        truth = [sentence_label(r) for r in self.items]
        guesses = [g.upper().strip() for g in guesses]
        right = sum(g == t for g, t in zip(guesses, truth))
        print(f"{right} of {len(truth)} correct\n")
        for i, (g, t, r) in enumerate(zip(guesses, truth, self.items)):
            frame = (r.get("spec") or {}).get("frame") or \
                    "+".join((r.get("spec") or {}).get("assertions") or []) or "-"
            mark = " " if g == t else "x"
            print(f" {mark} {i:>2}. guessed {g:<4} was {t:<4}  construction: {frame}")
        return right / len(truth)


def train_eval(train_records, eval_records, seeds=(0,), epochs=3):
    """Train BiomedBERT on sentence records and score it. Optional throughout: every
    number the notebook argues from is already in results.json, so nothing here is
    needed to follow the argument. It exists so a reader can reproduce the shape of a
    result on a sample, not to reproduce the headline figures."""
    from ddi.train import train_and_eval

    base = {"model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "epochs": epochs, "lr": 2e-5, "batch_size": 32, "max_length": 256,
            "neg_ratio": None, "render_mode": "markers"}
    tr, ev = flatten(train_records), flatten(eval_records)
    out = []
    for s in seeds:
        m = train_and_eval({**base, "seed": s, "dataset": "tutorial"}, tr, ev)
        out.append({"seed": s, "f1": m["micro_f1_pos"],
                    "p": m["micro_p_pos"], "r": m["micro_r_pos"]})
    return out
