# %% [markdown]
# # Can an LLM write training data for a BERT model?
#
# ### Gwilym Hughes, Summer 2026 Internship
#
# The main reason for looking towards synthetic training data to train AI models is that annotation is the bottleneck to starting new projects. Getting an LLM to write labelled examples instead skips the bottleneck. In theory the knowledge a large LLM has in its weights should be sufficient to generate good training data for a smaller model like BERT, i.e. distillation. This notebook is the result of an 8-week internship trying this for drug-drug interaction extraction.
#
# The material is the project's own: 400 sentences from each of five generator versions, 400 real sentences labelled three ways (by a person, by the model, and by the model after checking its own positive labels), and the scores behind every figure in the write-up. It assumes the earlier sentence-level relation extraction tutorial, which turns a DDI sentence into pair instances and fine-tunes BiomedBERT on them.

# %% [markdown]
# ## Setup
#
# Nothing below needs a GPU or an API key. The corpus download supplies the real sentences in section 1.

# %%
# !pip install -q bioc spacy scikit-learn pandas matplotlib
# !python -c "import spacy; spacy.cli.download('en_core_web_sm')"
# ![ -d ../DDICorpusBrat ] || (cd .. && curl -sL -o DDICorpus.zip "https://github.com/isegura/DDICorpus/raw/refs/heads/master/DDICorpus-2013(BRAT).zip" && unzip -oq DDICorpus.zip)

# %%
import math, random, re
from collections import Counter, defaultdict

import pandas as pd

import lib
from lib import load_samples, human_sentences, sentence_label, masked, flatten

synth = {v: load_samples(v) for v in lib.VERSIONS}
human = human_sentences("train", n=400, seed=0)
data = {"human": human, **synth}
R = lib.results()
{k: len(v) for k, v in data.items()}

TAGS = re.compile(r"\[(E[12])\](.*?)\[/\1\]")

def pair_names(text):
    found = {t: re.sub(r"\[/?E[12]\]", "", name) for t, name in TAGS.findall(text)}
    return found.get("E1"), found.get("E2")

# %% [markdown]
# ## 1. The task
#
# A sentence arrives with its drug mentions already marked, and every pair of mentions needs one of five labels: MECHANISM for a pharmacokinetic change, EFFECT for a clinical or pharmacodynamic consequence, ADVISE for a recommendation about taking the two together, INT for an interaction stated without detail, and NONE. The label belongs to the pair, not the sentence, so a sentence naming four drugs produces six instances, and each can carry a different label. Here is a real one.

# %%
ex = next(r for r in human if 3 <= r["n_entities"] <= 4 and sentence_label(r) == "POS")
print(ex["sentence"], "\n")
pd.DataFrame([{"drug A": a, "drug B": b, "label": p["label"]}
              for p in ex["pairs"] for a, b in [pair_names(p["text"])]])

# %% [markdown]
# About five pairs in six in the corpus are NONE, so scores are micro-F1 over the four positive labels; counting NONE would reward a model that never predicts anything else.
#
# Every dataset below, generated or real, is held as a list of records of this shape: the sentence, the number of drug mentions, every pair with its label, and, for generated sentences, the specification that produced it. Real sentences are built with the project's pair construction, one instance per unordered pair. The earlier tutorial makes one per ordered pair, twice as many, so the two builders must not be mixed in any comparison.

# %%
def shape(record):
    def show(v):
        if isinstance(v, list):
            return f"list of {len(v)}"
        if isinstance(v, dict):
            return "specification: " + ", ".join(sorted(v))
        if isinstance(v, str) and len(v) > 60:
            return v[:57] + "..."
        return v
    return {k: show(v) for k, v in record.items()}

pd.DataFrame({"generated (v18)": shape(synth["v18"][0]), "real": shape(human[0])})

# %% [markdown]
# The generated sentences are samples of 400 from the datasets the logged results were measured on, each joined back to its specification wherever the raw generation file survived; v13's did not. Scores come from `results.json`, rebuilt from the run records with the ids of the runs behind each figure. `lib.py` holds only loading, masking and training; every measurement is computed in the cells below.

# %%
cols = ["dataset_id", "gen_id", "sentences_available", "sentences_sampled",
        "raw_file_found", "specs_joined"]
man = pd.DataFrame(lib.sample_manifest()).T
display(man[[c for c in cols if c in man.columns]])

v18 = lib.results()["trajectory"]["v18"]
print(f"v18 scored {v18['f1']} (sd {v18['sd']}) over {v18['n_seeds']} seeds, "
      f"from runs {', '.join(v18['run_ids'])}")

# %% [markdown]
# ## 2. Writing labelled sentences
#
# The generator never asks the model for a label. It builds a specification, a structured description of one sentence that lists the drugs it must mention and which pairs among them stand in which relation, renders it as a prompt, and stores the model's sentence beside the specification. A deterministic second stage finds each drug in the sentence, tags every pair, and copies each pair's label from the specification. The specification is written before the sentence exists, so it is the gold annotation: the labels are correct by construction.

# %%
rec = next(r for r in synth["v18"]
           if r["spec"] and r["spec"].get("positives") and r["n_entities"] >= 3)
spec = rec["spec"]
surface = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", spec.get("entities") or []))   # keys run A, B, C in list order

print("assertions ", spec.get("assertions"))
print("modifiers  ", spec.get("modifiers"))
print("variants   ", spec.get("variants"))
print("\ngold, as the specification states it")
for p in spec["positives"]:
    a, b = p["between"]
    print(f"  {surface.get(a, a)}  {p['label']}  {surface.get(b, b)}")
print("\nthe model wrote\n ", rec["sentence"], "\n")
pd.DataFrame([{"drug A": a, "drug B": b, "label": p["label"]}
              for p in rec["pairs"] for a, b in [pair_names(p["text"])]])

# %% [markdown]
# The pairs the specification names carry its labels and every other pair is NONE, so one generated sentence supplies negative examples as well as positive ones.
#
# ### 2.1 What each generator produced

# %%
def describe(records):
    pairs = flatten(records)
    per = [len(r["pairs"]) for r in records]
    per.sort()
    return {"sentences": len(records),
            "pairs": len(pairs),
            "pairs/sentence mean": round(sum(per) / len(per), 2),
            "pairs/sentence median": per[len(per) // 2],
            "positive pairs": round(sum(p["label"] != "NONE" for p in pairs) / len(pairs), 3),
            "positive sentences": round(sum(sentence_label(r) == "POS" for r in records) / len(records), 3)}

pd.DataFrame({k: describe(v) for k, v in data.items()}).T

# %% [markdown]
# In v13 almost every sentence contains one pair and that pair is almost always positive: the generator did exactly what it was asked, which was to write a sentence in which two given drugs interact. Real sentences carry several drugs, and most of the pairs among them are NONE. The gap between the mean and the median for real text comes from a few sentences that list dozens of drugs and produce hundreds of pairs each.
#
# ### 2.2 The pair-count shortcut
#
# The rule below sees only how many drugs a sentence mentions, never its words.

# %%
def my_rule(record):
    # edit this: it sees only how many drugs the sentence contains
    return "POS" if record["n_entities"] == 2 else "NONE"

def score_rule(rule, records):
    return sum(rule(r) == sentence_label(r) for r in records) / len(records)

{k: round(score_rule(my_rule, v), 3) for k, v in data.items()}

# %% [markdown]
# The rule scores well on v13 and badly on real text. A classifier trained on v13 finds the same rule, because it is the shortest path to a low loss on that data, and applies it to real sentences where it no longer holds. The table below breaks the positive rate down by the number of drugs. The empty cells in v13's column are the finding itself: v13 never wrote a sentence with more than two drugs.

# %%
def positive_rate_by_entities(records):
    buckets = defaultdict(list)
    for r in records:
        for p in r["pairs"]:
            buckets[min(r["n_entities"], 6)].append(p["label"] != "NONE")
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(buckets.items())}

pd.DataFrame({k: positive_rate_by_entities(v) for k, v in data.items()}) \
  .rename_axis("drugs (6 = six or more)").fillna("none written")

# %% [markdown]
# ### 2.3 In detail: a label that can be read off the construction
#
# v14 added drugs that take no part in the relation, which removed the pair-count shortcut. Its precision rose and its recall did not (scores in 2.4). The cause is visible in a single v17 sentence.
#
# v17 built each sentence from one of nineteen named constructions, called frames, derived from the annotation guidelines. The cell picks the frame that most often produces positive sentences and one short sentence built from it.

# %%
v17 = [r for r in synth["v17"] if r["spec"]]
focus = Counter(r["spec"]["frame"] for r in v17 if sentence_label(r) == "POS").most_common(1)[0][0]
one = next(r for r in v17 if r["spec"]["frame"] == focus and 3 <= r["n_entities"] <= 5)
keys = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", one["spec"].get("entities") or []))
fill = lambda v: re.sub(r"\{([A-Z])\}", lambda m: keys.get(m.group(1), m.group(0)), str(v))

print("frame:", focus)
print("\nthe specification asked for")
for k, v in one["spec"].get("says") or []:
    print(f"  {k}: {fill(v)}")
print("\nthe model wrote\n ", one["sentence"], "\n")
pd.DataFrame([{"drug A": a, "drug B": b, "label": p["label"]}
              for p in one["pairs"] for a, b in [pair_names(p["text"])]])

# %% [markdown]
# Nothing here is wrong. The specification asked for a relation, the model wrote it, and the pair carries the label the specification gave it. The problem appears only across sentences. Here are the sentence labels of every sample sentence built from the same frame, then of every frame.

# %%
same = [r for r in v17 if r["spec"]["frame"] == focus]
print(f"{len(same)} sample sentences use '{focus}':", dict(Counter(sentence_label(r) for r in same)), "\n")
pd.crosstab(pd.Series([r["spec"]["frame"] for r in v17], name="frame"),
            pd.Series([sentence_label(r) for r in v17], name="sentence label"))

# %% [markdown]
# Every row has a zero in one column. Each frame always produces the same sentence label, so knowing the frame settles the label.
#
# The uncertainty coefficient turns that observation into a number. With $Y$ the sentence label and $F$ the frame, the entropy of the label is $H(Y) = -\sum_y p(y)\log_2 p(y)$, and the entropy left once the frame is known is $H(Y \mid F) = \sum_f p(f)\,H(Y \mid F = f)$. The coefficient is the share of the label's uncertainty that the frame removes:
#
# $$U(Y \mid F) = \frac{H(Y) - H(Y \mid F)}{H(Y)}$$
#
# It is 1 when the frame determines the label and 0 when the frame says nothing about it. It is not sklearn's `normalized_mutual_info_score`, which divides by the average of both entropies: a nineteen-way frame carries far more entropy than a two-way label, so that measure reports about 0.4 even when every frame is pure. Computed by hand from the counts above:

# %%
labels = Counter(sentence_label(r) for r in v17)
n = sum(labels.values())
H_Y = -sum(c / n * math.log2(c / n) for c in labels.values())

by_frame = defaultdict(Counter)
for r in v17:
    by_frame[r["spec"]["frame"]][sentence_label(r)] += 1
H_Y_given_F = sum(sum(c.values()) / n * -sum(k / sum(c.values()) * math.log2(k / sum(c.values()))
                                            for k in c.values() if k)
                  for c in by_frame.values())

print(f"H(Y)     = {H_Y:.3f} bits   from {dict(labels)}")
print(f"H(Y | F) = {H_Y_given_F:.3f} bits   every frame is pure, so each term is zero")
print(f"U(Y | F) = {(H_Y - H_Y_given_F) / H_Y:.3f}")

# %% [markdown]
# The same measure, as a function used for every comparison that follows:

# %%
def entropy(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c)

def uncertainty_coefficient(pairs):
    """U(label | x): the fraction of the label's entropy that x removes.
    1.0 means x determines the label outright, 0.0 means it says nothing about it.

    This is not sklearn's normalized_mutual_info_score, which divides by the mean of
    both entropies and so reports about 0.4 for a 19-way construction that fully
    determines a binary label. The asymmetric version asks the question that matters
    here, which is whether knowing the construction settles the label."""
    joint = Counter(pairs)
    px, py = Counter(), Counter()
    for (x, y), c in joint.items():
        px[x] += c; py[y] += c
    n = sum(joint.values())
    mi = sum(c / n * math.log2((c / n) / ((px[x] / n) * (py[y] / n)))
             for (x, y), c in joint.items())
    return mi / entropy(py)

round(uncertainty_coefficient([(r["spec"]["frame"], sentence_label(r)) for r in v17]), 3)

# %% [markdown]
# A label fixed by the frame would do no harm if the frame were invisible in the finished sentence, since the classifier never sees the specification. The test is whether the words give the frame away. The cell masks every drug name, trains a bag-of-words model to name the frame, and checks the sentence from above against the words the model relies on most for its frame.

# %%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
X = vec.fit_transform([masked(r) for r in v17])
y = [r["spec"]["frame"] for r in v17]
clf = LogisticRegression(max_iter=3000).fit(X, y)
names = vec.get_feature_names_out()
top = [names[i] for i in np.argsort(clf.coef_[list(clf.classes_).index(focus)])[::-1][:12]]

print("the sentence, masked\n ", masked(one), "\n")
print(f"strongest features for '{focus}':", ", ".join(top))
print("of those, present in this sentence:", ", ".join(w for w in top if w in masked(one).lower()) or "none")
acc = cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=3).mean()
print(f"\nframe recovered from masked text: {acc:.3f} over {len(set(y))} frames (chance {1 / len(set(y)):.3f})")

# %% [markdown]
# On this 400-sentence sample the frame is recovered three times in four against a chance rate of one in nineteen; on the full 5,702-sentence dataset the same test reaches 0.961. The difference is training data per frame: three-fold cross-validation over 400 sentences leaves about fourteen examples of each. The chain is complete in both cases. The words name the frame and the frame names the label, a path from surface to answer that never asks whether the two drugs interact.
#
# The same test works on a person. The game below shows ten masked v17 sentences and asks whether each asserts an interaction; the answers print the frame beside each one.

# %%
game = lib.GuessGame(synth["v17"], n=10, seed=1)
game.show()

# %%
my_guesses = ["POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE"]   # edit
game.score(my_guesses)

# %% [markdown]
# The same game on real sentences:

# %%
real_game = lib.GuessGame(human, n=10, seed=1)
real_game.show()

# %%
my_real_guesses = ["POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE"]   # edit
real_game.score(my_real_guesses)

# %% [markdown]
# I scored 9 of 10 on each, so accuracy does not separate the two. The route does. Every generated NONE was given away by the wording attached to a drug outside the relation, such as "part of the background regimen", "used if the first choice is unsuitable" or "prescribed for an unrelated condition", and needed no pharmacology; each real answer came from reading the relation. Both misses were conventions rather than pharmacology: v17's `incompatible` frame, which the generator labels NONE although the one corpus sentence using it is positive, and on the real side what looked like a guideline convention case.
#
# The most repeated phrases show where those words come from.

# %%
def top_ngrams(records, n=4, k=10):
    c = Counter()
    for r in records:
        w = re.sub(r"[^\w\s]", " ", masked(r).lower()).split()
        c.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(k)

pd.DataFrame({"v17": [f"{g}  ({c})" for g, c in top_ngrams(synth["v17"])],
              "human": [f"{g}  ({c})" for g, c in top_ngrams(human)]})

# %% [markdown]
# v17 described every drug outside the relation with one of seven fixed phrases, and those phrases supply its most repeated 4-grams; real text's are drug enumerations. Its negative frames also avoided effect vocabulary, so words such as *caused* and *prolonged* mark positives and denial language marks NONE, neither of which holds in real text.
#
# The corpus shows which constructions carry labels. 300 real sentences were classified by construction, once, by `gpt-oss-120b` at temperature 0, and committed in `runs/frames/`. For each construction the table gives how often it is the main clause of a sentence, how often it appears at all, and its positive rate in the corpus against v17.

# %%
import json
from pathlib import Path

gold = [json.loads(l) for l in (lib.REPO / "runs/frames/gold_frames_n300_seed0.jsonl").read_text().splitlines() if l]
gold = [g for g in gold if not g.get("error") and g.get("confident", True)]

primary, present, corpus_pos = Counter(), Counter(), defaultdict(list)
for g in gold:
    primary[g["primary"]] += 1
    for f in {g["primary"], *(g.get("also_present") or [])}:
        present[f] += 1
        corpus_pos[f].append(g["has_pos"])

v17_pos = defaultdict(list)
for r in synth["v17"]:
    if r["spec"]:
        v17_pos[r["spec"]["frame"]].append(sentence_label(r) == "POS")

rows = []
for f in sorted(present, key=lambda f: -present[f]):
    c = sum(corpus_pos[f]) / len(corpus_pos[f])
    s = sum(v17_pos[f]) / len(v17_pos[f]) if v17_pos[f] else None
    rows.append({"construction": f, "main clause": primary[f], "present in": present[f],
                 "corpus positive": round(c, 3),
                 "v17 positive": None if s is None else round(s, 3),
                 "gap": None if s is None else round(abs(s - c), 3)})
pd.DataFrame(rows)

# %% [markdown]
# Appositive and coordinate constructions appear in real sentences but are never the main clause. They package an assertion rather than make one, yet v17 sampled them as alternatives to mechanism and denial, and made appositive sentences positive every time against a corpus rate near four in ten.

# %% [markdown]
# ### 2.4 The fix: two axes
#
# v18 splits the inventory. Assertions bind a pair of drugs and carry a label; a sentence draws one or two, the second usually negative. Modifiers carry no label and are drawn independently: headings, repeated mentions, dose details, appositives. The cell runs the project's v18 code; building and rendering a specification needs no API call.

# %%
from ddi.vocab import build_vocab
from ddi import prompt_v18 as p18

vocab = build_vocab()
specs = p18.make_specs(3000, vocab=vocab, seed=0)

s = specs[3]
print(f"assertions {s['assertions']}   modifiers {s['modifiers']}   variants {s['variants']}")
print(f"gold {s['positives'] or 'all NONE'}\n")
print(p18.render(s))

# %% [markdown]
# The same coefficient for v17's frames, v18's assertion sets, and the corpus's constructions. The target is the corpus, not zero: real constructions carry some information about the label, and a generator at zero would be writing denials as often positive as negative.

# %%
from ddi import prompt as p17

v17_specs = p17.make_specs(3000, vocab=vocab, seed=0)
pd.Series({
    "v17  frame": uncertainty_coefficient([(x["frame"], bool(x["positives"])) for x in v17_specs]),
    "v18  assertion set": uncertainty_coefficient(
        [("+".join(sorted(x["assertions"])), bool(x["positives"])) for x in specs]),
    "corpus  construction": uncertainty_coefficient([(g["primary"], g["has_pos"]) for g in gold]),
}).round(3)

# %% [markdown]
# v18's coefficient is 0.458 against the corpus's 0.464. In the logged runs the change took F1 from 0.391 to 0.486 [dev], and the gain was recall, 0.403 to 0.628, with precision nearly unchanged: the classifier stopped relying on a cue that real text does not contain. The scores of every version:

# %%
pd.DataFrame(R["trajectory"]).T[["f1", "sd", "p", "r", "n_seeds"]]

# %% [markdown]
# The v13 figure is one seed on a version of v13 padded with extra negatives, not the raw generation sampled above. All five are development scores. The human baseline on the same set is about 0.80, which is not comparable with the 0.85 to 0.90 in much of the DDI-2013 literature: those papers delete trivially negative pairs before training and evaluation, and this project does not.
#
# ### 2.5 Extending the generator
#
# An assertion is a function from the sentence's entities and a random generator to what the sentence must say, which pairs are positive and with which label, and which entities it concerns. Its `kind` selects the stylistic axes the renderer samples, so it must be one of these; anything else raises a KeyError inside `make_specs`.

# %%
sorted(p17.AXES_FOR_KIND)

# %% [markdown]
# A new assertion for protein-binding displacement, a mechanism the inventory lacks, registered and measured without generating anything:

# %%
def f_displacement(ents, rng):
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("acts", f"{{{a}}}"), ("acted on", f"{{{b}}}"),
                     ("what changes", "binding to plasma proteins"), ("which way", "displaced")],
            "positives": [(a, b, "MECHANISM")],
            "focus": [a, b]}

saved = dict(p18.ASSERTIONS)
try:
    p18.ASSERTIONS["displacement"] = {"n": 2, "w": 0.06, "f": f_displacement}
    extended = p18.make_specs(3000, vocab=vocab, seed=0)
finally:
    p18.ASSERTIONS.clear(); p18.ASSERTIONS.update(saved)

drawn = [x for x in extended if "displacement" in x["assertions"]]
alone = [x for x in drawn if len(x["assertions"]) == 1]
print(f"drawn in {len(drawn)} of {len(extended)} specs")
print(f"sentence positive rate when it is the only assertion: "
      f"{sum(bool(x['positives']) for x in alone) / len(alone):.3f}")

# %% [markdown]
# Alone, the new positive assertion yields a positive sentence about three times in four, not every time. v18's `unlisted` variant retargets about a quarter of positive assertions at a drug outside the sentence's list, which makes them NONE, so a new assertion inherits that impurity and cannot become a new shortcut. All of this costs nothing, because a specification can be built and measured before any sentence is generated.

# %% [markdown]
# ## 3. Labelling real sentences
#
# The other use of the model keeps real sentences and their marked drugs and asks only for the labels. The annotator sees eight pairs per request, with the guideline conventions in its prompt. Here is one sentence from the labelled sample, with the gold label and the model's label for every pair.

# %%
lab = lib.load_labelled()   # built on the pod by make_samples.py --labelled

def fp_rejected(p):
    return p["llm"] != "NONE" and p["gold"] == "NONE" and p["verdict"] == "rejected"

pool = [r for r in lab if any(fp_rejected(p) for p in r["pairs"]) and len(r["pairs"]) >= 3]
story = min(pool, key=lambda r: len(r["pairs"]))
print(story["sentence"], "\n")
pd.DataFrame([{"drug A": a, "drug B": b, "gold": p["gold"], "model": p["llm"]}
              for p in story["pairs"] for a, b in [pair_names(p["text"])]])

# %% [markdown]
# At least one pair here is positive to the model and NONE in the gold labels. Across the sample the pattern is the same: the model finds most real interactions and adds interactions the annotators did not mark. The label-level scores, the model's labels scored against gold with the task metric:

# %%
def prf(pairs, key):
    tp = sum(1 for p in pairs if p[key] != "NONE" and p[key] == p["gold"])
    fp = sum(1 for p in pairs if p[key] != "NONE" and p[key] != p["gold"])
    fn = sum(1 for p in pairs if p["gold"] != "NONE" and p[key] != p["gold"])
    P, Rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(P, 3), "recall": round(Rc, 3),
            "F1": round(2 * P * Rc / max(P + Rc, 1e-9), 3), "positives": tp + fp}

pairs = [p for r in lab for p in r["pairs"]]
print("gold positives:", sum(p["gold"] != "NONE" for p in pairs), "of", len(pairs), "pairs")
prf(pairs, "llm")

# %% [markdown]
# On the full labelled pool the annotator reached precision 0.623 and recall 0.881 against gold [dev]. Trained on, those labels cost 0.126 F1 against the same sentences with human labels [test]:

# %%
T = R["test"]
readable = {"test-bfull-none": "human labels", "test-b0-llm": "model labels", "test-b0-llmf": "model labels, checked"}
pd.DataFrame({readable[k]: T[k] for k in ["test-bfull-none", "test-b0-llm"]}).T[["f1", "sd", "p", "r", "n_seeds"]]

# %% [markdown]
# ## 4. In detail: checking the model's labels
#
# The model's errors are mostly false positives, so the check asks about positives only. A second call, the verifier, receives the sentence with every drug mention numbered and one question per pair: would the guidelines annotate an interaction between these two mentions? A pair the verifier rejects is demoted to NONE. This is the exact question it was sent for the sentence above.

# %%
from ddi.verify_binary import SYSTEM, build_batches, render

asked = [{"sent_id": story["sent_id"], "text": p["text"], "label": p["llm"]}
         for p in story["pairs"] if p["llm"] != "NONE"]
print("\n".join(SYSTEM.splitlines()[:9]), "\n  ...\n")
print(render(build_batches(asked)[0]))

# %% [markdown]
# Mentions are numbered rather than named because the same drug named twice is two mentions, and the guidelines let only one of them take part in an interaction. Only the model's positive pairs are asked about, so the numbering counts the mentions in those pairs. The verdicts, and the labels after demotion:

# %%
pd.DataFrame([{"drug A": a, "drug B": b, "gold": p["gold"], "model": p["llm"],
               "verifier": p["verdict"], "after": p["final"]}
              for p in story["pairs"] for a, b in [pair_names(p["text"])]])

# %% [markdown]
# In this sentence the verifier rejects a pair that is NONE in gold, and demotion corrects it. Whether that holds across the sample decides whether the check works: the rejected positives should be NONE in gold far more often than the accepted ones.

# %%
llm_pos = [p for p in pairs if p["llm"] != "NONE"]
def gold_none(ps):
    k = sum(p["gold"] == "NONE" for p in ps)
    return f"{k:>4} of {len(ps):<4} ({k / max(len(ps), 1):.0%})"
print("model positives that are NONE in gold  ", gold_none(llm_pos))
print("  rejected by the verifier             ", gold_none([p for p in llm_pos if p["verdict"] == "rejected"]))
print("  accepted by the verifier             ", gold_none([p for p in llm_pos if p["verdict"] == "accepted"]))
print("  not asked, the request failed        ", gold_none([p for p in llm_pos if p["verdict"] == "not asked"]))

# %% [markdown]
# NOTE: state the result once `labelled.jsonl` exists. Give the share of rejected positives that are NONE in gold against the share among accepted ones, and say in one sentence whether the verifier is removing false positives or rejecting at random.
#
# The check also has a cost: a rejected pair that is positive in gold loses a true interaction.

# %%
cost = [(r, p) for r in lab for p in r["pairs"] if p["verdict"] == "rejected" and p["gold"] != "NONE"]
print(f"{len(cost)} rejected pairs are positive in gold")
if cost:
    r, p = cost[0]
    a, b = pair_names(p["text"])
    print(f"\n{r['sentence']}\n\n{a} and {b}: gold {p['gold']}, model {p['llm']}, demoted to NONE")

# %% [markdown]
# NOTE: describe the example above in one sentence, and what kind of pair the verifier wrongly rejects.
#
# Scored against gold before and after demotion:

# %%
pd.DataFrame({"before checking": prf(pairs, "llm"), "after checking": prf(pairs, "final")})

# %% [markdown]
# On the test set, the classifier trained on checked labels moved from precision 0.578 and recall 0.824 to 0.676 and 0.790 [test], and checking recovered 39% of the label cost from section 3:

# %%
pd.DataFrame({readable[k]: T[k] for k in ["test-bfull-none", "test-b0-llmf", "test-b0-llm"]}).T[["f1", "sd", "p", "r", "n_seeds"]]

# %% [markdown]
# ## 5. Spending annotation
#
# With some human-labelled sentences and more without labels, the question is what to add. The grid fixed a human budget of 0 to 1,000 sentences and added nothing, model labels on the remaining sentences, checked model labels, generated v18 text, or both. These are validation scores. Every arm trained for at least 500 optimisation steps; with a fixed three epochs, the smallest human-only arms collapse to predicting NONE for every pair, which makes anything added to them look far more valuable than it is.

# %%
grid = R["grid"]
adds = [("none", "human only"), ("llm", "+ LLM labels"), ("llmf", "+ LLM labels, verified"),
        ("synth", "+ generated v18"), ("both", "+ LLM labels and v18")]
budgets = [0, 100, 250, 500, 1000]
tab = pd.DataFrame({label: {b: grid[str(b)].get(a, {}).get("f1") for b in budgets}
                    for a, label in adds}).rename_axis("human-labelled sentences")
ax = tab.plot(marker="o", figsize=(7, 4), ylabel="validation micro-F1", xticks=budgets)
ax.set_xticklabels(budgets)
tab.round(3).fillna("-")   # no human-only arm exists at a budget of 0

# %% [markdown]
# The configurations validation chose, scored once on the test set:

# %%
order = ["test-b0-synth", "test-b0-llm", "test-b0-llmf", "test-b1000-none", "test-b1000-llmf", "test-bfull-none"]
names = {"test-b0-synth": "generated v18", "test-b0-llm": "model labels",
         "test-b0-llmf": "model labels, checked", "test-b1000-none": "1,000 human only",
         "test-b1000-llmf": "1,000 human + checked model labels", "test-bfull-none": "all human"}
tt = pd.DataFrame({names[k]: {"human sentences": int(T[k]["human_sentences"]), "F1": T[k]["f1"], "sd": T[k]["sd"],
                              "P": T[k]["p"], "R": T[k]["r"]} for k in order}).T
tt["share of all human"] = (tt["F1"] / T["test-bfull-none"]["f1"]).round(3)
tt

# %% [markdown]
# With no human labels, checked model labels reach 90% of full human supervision, within 0.02 of 1,000 human-labelled sentences on their own; with 1,000 human sentences, 42% of the set, they reach 96%. Generated text reaches 65%, and added to model labels it lowered F1 at every budget on validation. The write-up gives these results with their uncertainties.

# %% [markdown]
# ## 6. Applying this to another task
#
# The machinery carries over unchanged: specification, render and generate; the two-stage split between stored model output and deterministic instance building; the verifier's numbered-mention question; and the diagnostics in section 2, which need only a specification with a construction field and a sample of real sentences classified the same way. What is specific to DDI is the label set, the guideline conventions in the annotator's and verifier's prompts, the drug vocabulary, and the pair structure.
#
# Checks worth running before generating anything:
#
# - Compare model labels on real text with generated text on the same evaluation set before improving either.
# - Build specifications and measure each construction's label purity and the uncertainty coefficient against a classified corpus sample; both are free.
# - Train every comparison to a minimum number of optimisation steps, not a fixed number of epochs.
# - Before trusting a filter, count what it removes by category.
# - Choose on a held-out split, freeze the choice, and test once.

# %% [markdown]
# ## Optional: train on a sample
#
# Needs a GPU. Reproduces the shape of the pair-count result on the 400-sentence samples, not the logged figures.

# %%
# dev = human_sentences("dev")
# for v in ["v13", "v14", "v18"]:
#     print(v, lib.train_eval(synth[v], dev, seeds=(0,)))
