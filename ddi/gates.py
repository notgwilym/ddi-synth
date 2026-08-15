"""Build gates. Run on every generated dataset before training.

The v13 composition shortcut was invisible in aggregate F1 and took four controlled
experiments to localise, two of which returned nulls that were then misread. Everything
here runs in seconds with no training run, which is the point: prompt iteration gates
here.

The central check is shortcut_probe. count_rule_f1 tests one specific shortcut, entity
count, which was v13's. A slot-based generator grows lexical ones instead: the first
v14 run at scale failed the probe because scene vocabulary ("regimen", "treatment",
"administered") only rendered on specs with non-participants, and so predicted the
absence of an assertion almost perfectly. The probe generalises: strip the markers and
ask whether any bag-of-words feature recovers the label. If it can, the markers are
decorative and the pair is not what the model learns.
"""
import re
from collections import Counter, defaultdict


# Derived, not invented. Corpus values measured on the filtered human train split
# (2421 sentences, 15225 instances, eight enumeration sentences removed). Margins are
# 3 * sd of the corpus estimator at n=500 sentences, which is roughly the annotation
# budget the mixing curve says is worth having.
#
#   measurement          corpus   sd@500   threshold
#   hard negative rate    0.502    0.038   >= 0.39
#   count-rule F1         0.281    0.022   <= 0.35
#   probe lift            0.135    0.030   <= 0.23
#   role adjacency gap   -0.009    0.007   <= 0.02
#
# The first three were previously 0.30, 0.21 and 0.15. The role gap is new: it is the
# largest divergence v14 exhibits (+0.235 against a corpus -0.009) and no gate measured
# it, which is the argument for framing these as corpus divergences rather than
# standalone checks.
THRESHOLDS = {
    "hard_negative_rate": 0.39,      # min
    "count_rule_f1": 0.35,           # max
    "shortcut_probe_lift": 0.23,     # max
    "role_adjacency_gap": 0.02,      # max
    "role_position_skew": 0.75,      # max, unchanged, no corpus analogue
    "drift_rate": 0.15,              # max, unchanged, verifier not on critical path
}

_MARKER = re.compile(r"\[/?E[12]\]")


def crosstab(instances):
    per_sent = Counter(r["sent_id"] for r in instances)
    has_pos = defaultdict(bool)
    for r in instances:
        if r["label"] != "NONE":
            has_pos[r["sent_id"]] = True

    cross = defaultdict(Counter)
    for r in instances:
        n = per_sent[r["sent_id"]]
        b = "1" if n == 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+"
        cross[b]["POS" if r["label"] != "NONE" else "NONE"] += 1

    none_mixed = sum(1 for r in instances
                     if r["label"] == "NONE" and has_pos[r["sent_id"]])
    none_tot = sum(1 for r in instances if r["label"] == "NONE")
    return cross, (none_mixed / none_tot if none_tot else 0.0)


def count_rule_f1(instances):
    """Micro-F1 over positive classes of a rule predicting positive iff the sentence has
    exactly one candidate pair, reading none of the text."""
    per_sent = Counter(r["sent_id"] for r in instances)
    tp = fp = fn = 0
    for r in instances:
        pred_pos = per_sent[r["sent_id"]] == 1
        gold_pos = r["label"] != "NONE"
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos:
            fp += 1
        elif gold_pos:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * rec / (p + rec) if p + rec else 0.0


def shortcut_probe(instances, seed=0):
    """Strip the markers, fit bag-of-words logistic regression, report macro-F1 lift over
    always-predicting-the-majority-class. Marker-stripped text is identical for every
    pair in a sentence, so anything above the baseline means a sentence-level cue is
    doing work the pair should be doing.

    Returns (lift, top_features). The top features name the culprit directly.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    import numpy as np

    X_txt = [_MARKER.sub("", r["text"]) for r in instances]
    y = ["POS" if r["label"] != "NONE" else "NONE" for r in instances]
    if len(set(y)) < 2:
        return 0.0, []

    # split on sent_id so the same sentence cannot appear on both sides
    sents = sorted({r["sent_id"] for r in instances})
    tr_s, _ = train_test_split(sents, test_size=0.3, random_state=seed)
    tr_s = set(tr_s)
    idx_tr = [i for i, r in enumerate(instances) if r["sent_id"] in tr_s]
    idx_te = [i for i, r in enumerate(instances) if r["sent_id"] not in tr_s]
    if not idx_tr or not idx_te:
        return 0.0, []

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr = vec.fit_transform([X_txt[i] for i in idx_tr])
    Xte = vec.transform([X_txt[i] for i in idx_te])
    ytr = [y[i] for i in idx_tr]
    yte = [y[i] for i in idx_te]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return 0.0, []

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    got = f1_score(yte, clf.predict(Xte), average="macro")

    major = Counter(ytr).most_common(1)[0][0]
    base = f1_score(yte, [major] * len(yte), average="macro", zero_division=0)

    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    top = [(names[j], round(float(coef[j]), 2))
           for j in np.argsort(-np.abs(coef))[:15]]
    return got - base, top


def role_position_skew(records):
    """Larger of the before/after shares for role-bearing drugs relative to the asserted
    pair. Near 1.0 means non-participants always sit in the same slot, so clause
    position substitutes for reading the markers."""
    before = after = 0
    for r in records:
        anchor = r.get("positive_span")
        if anchor is None:
            continue
        for s in r.get("role_spans", []):
            if s < anchor:
                before += 1
            else:
                after += 1
    tot = before + after
    return max(before, after) / tot if tot else 0.0


def diversity(records, n=4, top=12):
    """distinct-n, repeated n-gram mass, and the most common openings. Templating shows
    up here long before it shows up in downstream F1."""
    grams, opens = Counter(), Counter()
    for r in records:
        w = r["sentence"].lower().split()
        opens[" ".join(w[:3])] += 1
        for i in range(len(w) - n + 1):
            grams[" ".join(w[i:i + n])] += 1
    total = sum(grams.values())
    if not total:
        return {}
    return {"distinct_n": len(grams) / total,
            "repeated_ngram_mass": sum(c for c in grams.values() if c > 1) / total,
            "top_ngrams": grams.most_common(top),
            "top_openings": opens.most_common(top)}


def report(instances, records=None, drift_rate=None, strict=True, verbose=True):
    cross, hard_neg = crosstab(instances)
    rule = count_rule_f1(instances)
    lift, top_feats = shortcut_probe(instances)

    print(f"{'pairs/sent':<12} {'POS':>7} {'NONE':>7} {'pos rate':>9}")
    for b in ["1", "2-3", "4-9", "10+"]:
        c = cross[b]
        tot = c["POS"] + c["NONE"]
        if tot:
            print(f"{b:<12} {c['POS']:>7} {c['NONE']:>7} {c['POS']/tot:>9.3f}")

    checks = [
        ("hard negative rate", hard_neg, THRESHOLDS["hard_negative_rate"], "min"),
        ("count-only rule F1", rule, THRESHOLDS["count_rule_f1"], "max"),
        ("shortcut probe lift", lift, THRESHOLDS["shortcut_probe_lift"], "max"),
    ]
    from ddi.divergence import role_adjacency_gap
    checks.append(("role adjacency gap", role_adjacency_gap(instances),
                   THRESHOLDS["role_adjacency_gap"], "max"))
    if records:
        checks.append(("role position skew", role_position_skew(records),
                       THRESHOLDS["role_position_skew"], "max"))
    if drift_rate is not None:
        checks.append(("verifier drift rate", drift_rate,
                       THRESHOLDS["drift_rate"], "max"))

    print()
    failed = []
    for name, value, thresh, direction in checks:
        ok = value >= thresh if direction == "min" else value <= thresh
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {value:.3f} "
              f"({'>=' if direction == 'min' else '<='} {thresh})")
        if not ok:
            failed.append(name)

    if verbose and lift > THRESHOLDS["shortcut_probe_lift"]:
        print("\n  probe is using:")
        for f, c in top_feats:
            print(f"    {c:>7.2f}  {f}")

    if verbose and records:
        d = diversity(records)
        if d:
            print(f"\n  distinct-4 {d['distinct_n']:.3f}, "
                  f"repeated 4-gram mass {d['repeated_ngram_mass']:.3f}")
            print("  most repeated 4-grams:")
            for g, c in d["top_ngrams"][:8]:
                print(f"    {c:>4}  {g}")
            print("  most common openings:")
            for g, c in d["top_openings"][:6]:
                print(f"    {c:>4}  {g}")

    out = {"hard_negative_rate": hard_neg, "count_rule_f1": rule,
           "shortcut_probe_lift": lift, "failed": failed}
    if failed and strict:
        raise AssertionError(f"build gates failed: {', '.join(failed)}")
    return out