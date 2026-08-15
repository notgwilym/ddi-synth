"""Corpus divergence, tiered by annotation cost.

WHY THIS REPLACES gates.py's FRAMING
------------------------------------
gates.py was written to catch v13's specific defect, so every check asks a question
about the synthetic dataset in isolation: does entity count predict the label, does
bag-of-words predict the label, is role position skewed. Each has a threshold I invented,
and two of them are still set below what the real corpus achieves, so they report FAIL on
data better than DDI-2013.

Corpus similarity is the more general question and it subsumes them, because a shortcut
is a distributional divergence that happens to correlate with the label. Every gate
becomes a pair: corpus value, synthetic value, gap. No invented thresholds, and the
reference is the thing you are actually evaluated against.

This also fixes an ordering error that cost several days. The largest divergence yet
measured -- role language adjacent to a NONE-marked entity, 0.240 in v14 against 0.005 in
the corpus, a fiftyfold gap -- was found by writing a one-off notebook cell, not by any
gate. A framework that asks "how far is this from the corpus" would have surfaced it
before a full generation-and-training cycle was spent on the wrong hypothesis.

THE ANNOTATION-COST AXIS
------------------------
Each measurement needs a different amount of labelled data before it can be estimated at
all. That axis is what makes this a contribution rather than a dashboard, and it follows
directly from the project's reframing.

The original question was "can synthetic data replace annotation": answered, no, 0.379
against 0.790. The useful question is "given N annotations, what can you do", because
nobody generating synthetic data has zero labels -- with zero labels they could not
evaluate. The mixing curve puts a number on the training-data half of that: synthetic
data is worth +0.47 F1 at 100 sentences and nothing at 2400.

This module is the measurement half. Someone with a new task reads down the table and
learns what they can check for free, what 200 annotations buys, and what they cannot
verify at all without thousands.

    schema          label definitions only, no corpus contact
    unlabelled      corpus text without annotations. Cheap and usually available.
    labelled        needs relation labels
    labelled+spans  needs entity offsets too

The minimum-n column is measured, not asserted: subsample human train, redraw, and
record where the estimator stabilises. The probe lesson in hard numbers -- sd 0.020 at
26k instances, sd 0.085 and a spread of 0.22 at 1.6k -- is exactly what that column is
for, and it is why every threshold in gates.py should have been derived this way.

WHAT THIS FEEDS
---------------
The 19 August talk: divergence table beside the mixing curve.
The write-up: what to measure before training, and what it costs to measure.
The MSci tool: this is its data model. Panels tagged by tier, tier lockable, downstream
panels greyed out when you are not entitled to the input. Gwilym caught me violating
exactly that when I printed 50 gold MECHANISM sentences to design content slots -- the
counts were a declared MEASURED input, the prose reading contaminated the PRIOR arm.
"""
import re
import random
import statistics
from collections import Counter, defaultdict

MARKER = re.compile(r"\[/?E[12]\]")

# The role vocabulary that v14's r1-r5 tokens realise into. Not a general-purpose list:
# it exists to measure one specific generator artefact against the corpus.
ROLE_LANG = re.compile(
    r"\b(previous(ly)?|earlier|discontinu\w+|stopped|withdraw\w+|"
    r"subsequent(ly)?|thereafter|later|after the other\w*|"
    r"comparator|separately|comparison|"
    r"alternative|unsuitable|first choice|instead|"
    r"not given|not permitted|excluded|withheld|not administered)\b", re.I)


def _strip(text):
    return MARKER.sub("", text)


def _near_marker(text, chars=60):
    """Text within `chars` either side of each marked span."""
    out = []
    for k in (1, 2):
        m = re.search(rf"\[E{k}\].*?\[/E{k}\]", text, re.S)
        if m:
            out.append(text[max(0, m.start() - chars):m.end() + chars])
    return " ".join(out)


# ---------------------------------------------------------------- measurements
# Each takes a list of instances and returns a float. Scalars only, so corpus and
# synthetic are directly comparable and the gap is meaningful.

def positive_rate(inst):
    return sum(1 for r in inst if r["label"] != "NONE") / max(len(inst), 1)


def pairs_per_sentence(inst):
    n_sent = len({r["sent_id"] for r in inst})
    return len(inst) / max(n_sent, 1)


def median_sentence_length(inst):
    seen, lens = set(), []
    for r in inst:
        if r["sent_id"] in seen:
            continue
        seen.add(r["sent_id"])
        lens.append(len(_strip(r["text"]).split()))
    return statistics.median(lens) if lens else 0.0


def distinct_4(inst):
    seen, grams = set(), Counter()
    for r in inst:
        if r["sent_id"] in seen:
            continue
        seen.add(r["sent_id"])
        w = _strip(r["text"]).lower().split()
        for i in range(len(w) - 3):
            grams[" ".join(w[i:i + 4])] += 1
    tot = sum(grams.values())
    return len(grams) / tot if tot else 0.0


def opening_entropy(inst):
    """Shannon entropy over opening trigrams, in bits. Low means one frame dominates:
    v14 opened 327 sentences with 'the finding that' out of 5540."""
    import math
    seen, opens = set(), Counter()
    for r in inst:
        if r["sent_id"] in seen:
            continue
        seen.add(r["sent_id"])
        w = _strip(r["text"]).lower().split()[:3]
        opens[" ".join(w)] += 1
    n = sum(opens.values())
    if not n:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in opens.values())


def hard_negative_rate(inst):
    """NONE pairs sitting in a sentence that also asserts an interaction. v13 scored
    0.001 here against a corpus prose figure near 0.50, which was the composition
    shortcut in one number."""
    has_pos = defaultdict(bool)
    for r in inst:
        if r["label"] != "NONE":
            has_pos[r["sent_id"]] = True
    none = [r for r in inst if r["label"] == "NONE"]
    if not none:
        return 0.0
    return sum(1 for r in none if has_pos[r["sent_id"]]) / len(none)


def count_rule_f1(inst):
    """Micro-F1 over positive classes of a rule predicting positive iff the sentence has
    exactly one candidate pair, reading none of the text."""
    per_sent = Counter(r["sent_id"] for r in inst)
    tp = fp = fn = 0
    for r in inst:
        pred, gold = per_sent[r["sent_id"]] == 1, r["label"] != "NONE"
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * rec / (p + rec) if p + rec else 0.0


def role_adjacency_none(inst):
    """Share of NONE pairs with role language within 60 chars of a marked span.
    Corpus 0.005, v14 0.240. The largest divergence measured so far."""
    none = [r for r in inst if r["label"] == "NONE"]
    if not none:
        return 0.0
    return sum(1 for r in none if ROLE_LANG.search(_near_marker(r["text"]))) / len(none)


def role_adjacency_gap(inst):
    """NONE adjacency minus POS adjacency. Corpus -0.009, v14 +0.235.

    A difference, not a ratio. The ratio version had sd of 3.5 million at n=50 because
    POS adjacency is ~2% in the corpus and a 50-sentence draw can contain none, putting
    a near-zero in the denominator. The difference is bounded in [-1, 1] and monotone in
    the same direction.

    Sign is the point: negative means role language sits nearer positives, which is what
    the corpus does; positive means it flags negatives, which is the v14 artefact.
    """
    pos = [r for r in inst if r["label"] != "NONE"]
    none = [r for r in inst if r["label"] == "NONE"]
    if not pos or not none:
        return 0.0
    p = sum(1 for r in pos if ROLE_LANG.search(_near_marker(r["text"]))) / len(pos)
    n = sum(1 for r in none if ROLE_LANG.search(_near_marker(r["text"]))) / len(none)
    return n - p


def probe_lift(inst, seed=0, max_features=5000):
    """Partial-input baseline. Strip the markers, fit bag-of-words logistic regression,
    report macro-F1 lift over the majority class.

    Marker-stripped text is identical for every pair in a sentence, so the model cannot
    distinguish them and must predict the sentence's majority label. Anything above the
    class prior means a sentence-level cue is doing work the pair should be doing.

    Standard practice from the dataset-artifact literature (hypothesis-only baselines in
    NLI), not a new idea. Scale-dependent: full human train scores 0.152 with sd 0.020,
    1.6k-instance subsamples range 0.00 to 0.17 with sd 0.085. Only compare at matched
    size.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    X = [_strip(r["text"]) for r in inst]
    y = ["POS" if r["label"] != "NONE" else "NONE" for r in inst]
    if len(set(y)) < 2:
        return 0.0

    sents = sorted({r["sent_id"] for r in inst})
    if len(sents) < 4:
        return 0.0
    tr_s = set(train_test_split(sents, test_size=0.3, random_state=seed)[0])
    itr = [i for i, r in enumerate(inst) if r["sent_id"] in tr_s]
    ite = [i for i, r in enumerate(inst) if r["sent_id"] not in tr_s]
    if not itr or not ite:
        return 0.0

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features)
    try:
        Xtr = vec.fit_transform([X[i] for i in itr])
    except ValueError:
        return 0.0
    ytr, yte = [y[i] for i in itr], [y[i] for i in ite]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return 0.0

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    got = f1_score(yte, clf.predict(vec.transform([X[i] for i in ite])),
                   average="macro")
    major = Counter(ytr).most_common(1)[0][0]
    base = f1_score(yte, [major] * len(yte), average="macro", zero_division=0)
    return got - base


# ---------------------------------------------------------------- registry
# tier: what you must have to compute it at all.
# lower_is_closer is not used -- the gap's sign is meaningful and the reader should see
# direction, not an abs().

MEASUREMENTS = [
    ("median sentence length", "unlabelled", median_sentence_length),
    ("distinct-4", "unlabelled", distinct_4),
    ("opening entropy (bits)", "unlabelled", opening_entropy),
    ("pairs per sentence", "labelled", pairs_per_sentence),
    ("positive rate", "labelled", positive_rate),
    ("hard negative rate", "labelled", hard_negative_rate),
    ("count-rule F1", "labelled", count_rule_f1),
    ("probe lift", "labelled", probe_lift),
    ("role adjacency, NONE", "labelled+spans", role_adjacency_none),
    ("role adjacency gap", "labelled+spans", role_adjacency_gap),
]


def compare(corpus, synth, measurements=MEASUREMENTS, match_size=True, seed=0):
    """Corpus value, synthetic value, gap.

    match_size subsamples the LARGER set by sentence to the smaller one's sentence
    count. Several measurements move with data volume -- the probe most sharply -- so an
    unmatched comparison flatters whichever side has less data.
    """
    import pandas as pd

    if match_size:
        c_s = {r["sent_id"] for r in corpus}
        s_s = {r["sent_id"] for r in synth}
        n = min(len(c_s), len(s_s))
        if len(c_s) > n:
            keep = set(random.Random(seed).sample(sorted(c_s), n))
            corpus = [r for r in corpus if r["sent_id"] in keep]
        if len(s_s) > n:
            keep = set(random.Random(seed).sample(sorted(s_s), n))
            synth = [r for r in synth if r["sent_id"] in keep]

    rows = []
    for name, tier, fn in measurements:
        c, s = fn(corpus), fn(synth)
        rows.append({"measurement": name, "tier": tier,
                     "corpus": round(c, 4), "synth": round(s, 4),
                     "gap": round(s - c, 4),
                     "ratio": round(s / c, 2) if c else float("nan")})
    return pd.DataFrame(rows)


def stability(corpus, measurement, sizes=(50, 100, 200, 500, 1000, 2000),
              draws=10, seed=0):
    """How many labelled sentences before this measurement is worth trusting.

    Subsample the corpus, redraw, record the spread. This is the minimum-n column, and
    it is measured rather than asserted. Every threshold in gates.py should have come
    from here: count_rule_f1 is set to 0.21 when the corpus scores 0.274, and
    shortcut_probe_lift to 0.15 when size-matched corpus is 0.152 with sd 0.020, so both
    report FAIL on data better than DDI-2013.
    """
    name, tier, fn = measurement
    full = fn(corpus)
    sents = sorted({r["sent_id"] for r in corpus})
    rows = []
    for n in sizes:
        if n > len(sents):
            continue
        vals = []
        for d in range(draws):
            keep = set(random.Random((seed, n, d).__hash__()).sample(sents, n))
            vals.append(fn([r for r in corpus if r["sent_id"] in keep]))
        rows.append({"measurement": name, "tier": tier, "n_sentences": n,
                     "median": round(statistics.median(vals), 4),
                     "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                     "bias": round(statistics.median(vals) - full, 4),
                     "full": round(full, 4)})
    return rows


def stability_table(corpus, measurements=None, **kw):
    import pandas as pd
    measurements = measurements or [m for m in MEASUREMENTS if m[1] != "unlabelled"]
    out = []
    for m in measurements:
        out.extend(stability(corpus, m, **kw))
    return pd.DataFrame(out)