# %% [markdown]
# # Can an LLM write training data for a BERT model?
#
# ### Gwilym Hughes, Summer 2026 Internship
#
# The main reason for looking towards synthetic training data to train AI models is that annotation is the bottleneck to starting new projects. Getting an LLM to write labelled examples instead skips the bottleneck. In theory the knowledge a large LLM has in it's weights should be sufficient to generate good training data for a smaller model like BERT, ie. distillation. This notebook is the result of an 8-week internship trying this for drug-drug interaction extraction.
#
# DRAFT: This notebook assumes the earlier sentence-level relation extraction tutorial, which shows how a DDI sentence becomes a set of pair instances and how BiomedBERT is trained on them, and it does not repeat that material. What it adds is the data the internship actually produced: a sample of real output from each of five generator versions, drawn from the datasets the logged results were measured on, so that every claim below can be checked against the sentences that produced it rather than against a reconstruction written for the occasion.

# %% [markdown]
# ## Setup
#
# DRAFT: Everything up to the final optional section runs on a laptop without a GPU and without calling any LLM, because the generated data is already in `samples/` and the logged scores are in `results.json`. The corpus download is only needed for the human comparison sentences.

# %%
# !pip install -q bioc spacy scikit-learn pandas
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
{k: len(v) for k, v in data.items()}

# %% [markdown]
# ## What each generator actually produced
#
# DRAFT: Before any model is trained, the simplest description of a dataset is how many drugs its sentences mention, how many pairs that produces, and what fraction of those pairs are positive, and it is worth looking at this table for a while before reading on because most of the story of the internship is already in it.

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
# DRAFT: The v13 row is the one to look at. Almost every sentence it wrote contains exactly one pair, and that pair is almost always positive, which is exactly what was asked of it (write a sentence in which these two drugs interact) and exactly what makes it useless, because real sentences contain several drugs and most of the pairs among them are NONE. The mean against the median for the human row is also worth a moment: a small number of sentences enumerate dozens of drugs and produce hundreds of pairs each, so the mean is dragged far above what a typical sentence looks like.

# %% [markdown]
# ## Shortcut one: count the drugs
#
# DRAFT: Before looking at the diagnostic, write a classifier yourself using nothing but the number of drugs in the sentence, with no access to its words, and see how it does on each dataset.

# %%
def my_rule(record):
    # edit this: it sees only how many drugs the sentence contains
    return "POS" if record["n_entities"] == 2 else "NONE"

def score_rule(rule, records):
    return sum(rule(r) == sentence_label(r) for r in records) / len(records)

{k: round(score_rule(my_rule, v), 3) for k, v in data.items()}

# %% [markdown]
# DRAFT: A rule that ignores every word in the sentence does very well on v13 and badly on real text. A classifier trained on v13 finds the same rule, because it is the shortest route to a low loss on that data, and then applies it to real sentences where it no longer holds. This is the composition shortcut, and the table below is the diagnostic that exposed it. The empty cells in v13's column are not missing data but the finding itself, since the generator never once wrote a sentence mentioning more than two drugs.

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
# ## Shortcut two: read the construction
#
# DRAFT: Fixing the composition shortcut by adding drugs that take no part in the relation improved precision but left recall flat, which at the time looked like a failed fix and was in fact a second shortcut becoming the binding constraint. To see it, play the game below. Each sentence has its drug names replaced with DRUG, and the task is to say whether the sentence asserts an interaction between any two of them.

# %%
game = lib.GuessGame(synth["v17"], n=10, seed=1)
game.show()

# %%
my_guesses = ["POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE"]   # edit
game.score(my_guesses)

# %% [markdown]
# DRAFT: Now play the same game on real sentences.

# %%
real_game = lib.GuessGame(human, n=10, seed=1)
real_game.show()

# %%
my_real_guesses = ["POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE", "POS", "NONE"]   # edit
real_game.score(my_real_guesses)

# %% [markdown]
# DRAFT: Scored on accuracy alone, the two games come out about the same for anyone who knows the domain, and I got 9 of 10 on each. What differs is how the answers were reached. On the generated sentences every NONE was given away by the wording used to describe a drug that takes no part in the relation, phrases such as "part of the background regimen", "used if the first choice is unsuitable" and "prescribed for an unrelated condition", so no pharmacology was needed at all, whereas on the real sentences each answer had to come from reading the relation itself. A classifier takes the cheaper route whenever one exists, and the generated data offered one.
#
# Both of my misses were conventions rather than pharmacology. On the generated side it was v17's `incompatible` construction, which the generator always labels NONE while the one real corpus sentence using it is positive, and on the real side it looked like a guideline convention case. The measurement that makes the difference in route precise asks how much of the uncertainty about a sentence's label is removed by knowing its construction.

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
    here, which is whether knowing the construction tells you the label."""
    joint = Counter(pairs)
    px, py = Counter(), Counter()
    for (x, y), c in joint.items():
        px[x] += c; py[y] += c
    n = sum(joint.values())
    mi = sum(c / n * math.log2((c / n) / ((px[x] / n) * (py[y] / n)))
             for (x, y), c in joint.items())
    return mi / entropy(py)

v17_frames = [(r["spec"]["frame"], sentence_label(r)) for r in synth["v17"] if r["spec"]]
round(uncertainty_coefficient(v17_frames), 3)

# %% [markdown]
# DRAFT: A value of 1.0 would be harmless on its own if the construction were invisible in the finished sentence, but it is not, and a bag-of-words model with the drug names masked out can name the construction from the text.

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

with_frame = [r for r in synth["v17"] if r["spec"]]
X = CountVectorizer(ngram_range=(1, 2), min_df=2).fit_transform([masked(r) for r in with_frame])
y = [r["spec"]["frame"] for r in with_frame]
acc = cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=3).mean()
print(f"construction recovered from masked text: {acc:.3f} over {len(set(y))} classes "
      f"(chance {1 / len(set(y)):.3f})")

# %% [markdown]
# DRAFT: On this 400-sentence sample the construction is recovered about three times in four against a chance rate of one in nineteen. On the full 5,702-sentence dataset the same test reaches 0.961, the difference most likely being the amount of training data per class rather than anything about the text, since three-fold cross-validation over 400 sentences leaves about fourteen examples of each construction to learn from.
#
# So the words give away the construction and the construction gives away the label, which is a complete route from surface to answer that never passes through the question of whether these two drugs interact. The most repeated phrases in the generated text show where the words came from.

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
# DRAFT: v17 described each non-participating drug using one of seven fixed phrases, such as "started before the others and since stopped", and those seven phrases supplied the most repeated 4-grams in the entire generated corpus. Worse, every negative construction had been written to avoid effect vocabulary, so the classifier learned that words like caused and prolonged mean positive and that denial language means NONE, neither of which holds in real text.

# %% [markdown]
# ## Finding the fix in the real corpus
#
# DRAFT: The way out came from classifying a sample of real corpus sentences by construction, which was done once for 300 sentences and is committed in `runs/frames/`, and then asking two questions of each construction: whether it is ever the main clause of a real sentence, and whether its positive rate in the generated data matches its positive rate in the corpus.

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
# DRAFT: Look at appositive and coordinate. Neither is the main clause of a single real sentence, yet v17 sampled them as though they were alternatives to mechanism and denial, and made appositive sentences positive every time against a corpus rate near four in ten. These are not assertions at all but ways of packaging one, and that observation is the whole of the fix.

# %% [markdown]
# ## The fix: two axes
#
# DRAFT: v18 splits the inventory in two. Assertions bind a pair of drugs and carry a label, and a sentence draws one or two of them, the second usually negative. Modifiers carry no label and are sampled independently: headings, repeated mentions, dose details, appositives. This is the real v18 code, and it runs here without any API calls, because building and rendering a spec is free.

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
# DRAFT: The same measurement as before, now on v18's assertion sets and on the real corpus's constructions. The target is not zero, because real constructions do carry some information about the label, and a generator that drove this to zero would be writing denials that are as often positive as negative, which is its own artefact. The target is the corpus.

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
# DRAFT: v18 did not just come down from 1.0, it landed within about a hundredth of the corpus. In the logged runs this change took F1 from 0.391 to 0.486, and the whole of the gain was in recall, which went from 0.403 to 0.628 while precision barely moved. That is the signature of a shortcut being removed: the classifier stops relying on a cue that real text does not contain and starts predicting on sentences it previously had no handle on.

# %% [markdown]
# ## Extending the generator
#
# DRAFT: An assertion is a function that takes the entities chosen for a sentence and a random generator, and returns what the sentence must say, which pairs are positive and with what label, and which entities it concerns. The `kind` field is not free text: it has to be one of the keys below, because it selects which stylistic axes the renderer samples, and anything else raises a KeyError deep inside `make_specs`.

# %%
sorted(p17.AXES_FOR_KIND)

# %% [markdown]
# DRAFT: Here is a new assertion for protein-binding displacement, a real mechanism the inventory does not cover. It is registered, specs are built, and the registry is restored afterwards so the rest of the notebook is unaffected.

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
# DRAFT: A positive assertion on its own ought to produce a positive sentence every time, and it comes out at roughly three in four instead. That is deliberate and it is inherited rather than written: v18's `unlisted` variant retargets about a quarter of positive assertions at a drug that is not in the sentence's list, which turns them NONE, so a new assertion cannot become a fresh label-pure shortcut however carelessly it is written. Everything in this section costs nothing, because it all happens before a single sentence is generated, and that is the practical lesson: check a design in spec space first, and only pay for generation once the numbers there look right.
#
# NOTE: adaptation to another task goes here. Which parts are DDI-specific (label set, guidelines conventions, vocabulary, the pair structure) and which carry over unchanged (spec, render, generate, the two-stage raw/instance split, the diagnostics above).

# %% [markdown]
# ## What the versions scored
#
# DRAFT: Every number here is recomputed from the committed run records and each carries the ids of the runs behind it, so none of it depends on anyone's memory of what a run produced.

# %%
R = lib.results()
pd.DataFrame(R["trajectory"]).T[["f1", "sd", "p", "r", "n_seeds"]]

# %% [markdown]
# DRAFT: Two caveats belong next to this table. The v13 figure is a single seed and was measured on a version of the v13 data padded with extra negatives, not on the raw generation sampled above. And the human baseline these should be read against is about 0.79, which is not comparable to the 0.85 to 0.90 reported in much of the DDI-2013 literature, because those papers usually delete trivially negative pairs before training and evaluation and this project does not.

# %% [markdown]
# ## The comparison that reframed it
#
# DRAFT: Eight weeks went into making generated text better. The comparison that should have come first keeps the real sentences and asks the LLM only for the labels, on the same sentences, with the same entity spans, so that the only thing varying between arms is where the text came from and where the labels came from.

# %%
arms = R["three_arm"]
pd.DataFrame({k: arms[k] for k in ["human", "llm_labels", "synthetic"]}).T[["f1", "sd", "p", "r", "n_seeds"]]

# %% [markdown]
# DRAFT: Swapping human labels for LLM labels on real text costs 0.128, and swapping real text for generated text costs a further 0.195, so the text matters more than the labels even though the generated labels are correct by construction and the LLM's own labels are noticeably wrong, over-predicting positives with precision well below human while recall is slightly above it.
#
# The exchange-rate runs complicate that in an interesting way.

# %%
xr = R["exchange_rate"]
pd.DataFrame({b: {"human": v["human"]["f1"], "llm labels": v["llm_labels"]["f1"]}
              for b, v in xr.items()}).T.rename_axis("real sentences")

# %% [markdown]
# DRAFT: At 250 sentences the LLM-labelled arm scores higher than the human-labelled one, and above roughly 400 the order reverses and the gap widens. With five seeds the difference at 250 is about one and a half standard errors, so it is a weak signal rather than a finding, but if it holds it says that LLM labels are worth most when real annotation is scarcest, which is exactly the situation synthetic data was supposed to rescue.

# %% [markdown]
# ## Optional: train on a sample
#
# NOTE: needs a GPU. Reproduces the shape of the composition result on 400-sentence samples, not the logged figures.

# %%
# dev = human_sentences("dev")
# for v in ["v13", "v14", "v18"]:
#     print(v, lib.train_eval(synth[v], dev, seeds=(0,)))

# %% [markdown]
# ## What we would do differently
#
# NOTE: head-to-head first; shortcuts arrive one at a time; measure constructions against the corpus before generating; check designs in spec space for free; watch which way filters are biased.
