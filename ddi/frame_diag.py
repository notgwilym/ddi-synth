"""Frame-level diagnostics.

WHY
---
v17's shortcut probe reads frame identity rather than relation language. Twelve of its
sixteen top-weighted features name a construction: "unchanged" and "no" are `denial`,
"mixed" and "before administration" are `incompatible`, "inhibitor of" is `enzyme_only`,
"gap" is `sequential`, "withdrawn" is `contradictory`, "contraindicated" is `advise`.

That is possible because `FRAMES` fixes a `positives` list per frame, so the label is a
deterministic function of the frame, and each frame writes its construction in a
distinctive vocabulary. The classifier can recover the frame from the surface and then
read the label off a lookup table without ever attending to the marked pair. It is the
v13 composition shortcut with construction in place of pair count.

This module measures the two halves separately and then in composition:

  A  frame recoverability   can a bag of words name the frame from the sentence
  B  label purity           does the frame determine the pair label
  C  composed               label accuracy from predicted frame alone

If C lands near the direct shortcut probe lift, frame identity accounts for the whole
shortcut and the fix is structural rather than lexical.

Nothing here requires the generator to record `frame` downstream. sent_id is
`synth:{gen_id}:{spec_index}` (synth.py:307), so the raw jsonl joins onto instances and
old datasets are diagnosable retroactively.
"""
import json
import re
from collections import Counter, defaultdict

from .synth import RAW

_MARKER = re.compile(r"\[/?E[12]\]")
_E1 = re.compile(r"\[E1\].*?\[/E1\]", re.S)
_E2 = re.compile(r"\[E2\].*?\[/E2\]", re.S)
_SUFFIX = re.compile(r":s\d+$")


def frame_index(gen_id):
    """sent_id -> spec metadata, read from the raw generation file."""
    out = {}
    path = RAW / f"{gen_id}.jsonl"
    for line in path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error") or not rec.get("sample"):
            continue
        spec = rec.get("spec") or {}
        idx = rec.get("spec_index", spec.get("spec_index"))
        if idx is None:
            continue
        out[f"synth:{gen_id}:{idx}"] = {
            "frame": spec.get("frame"),
            "kind": spec.get("kind"),
            "register": spec.get("register"),
            "n_entities": len(spec.get("entities") or []),
            "n_asserts": len(spec.get("asserts") or []),
            "n_roled": len(spec.get("roles") or {}),
        }
    return out


def attach_frames(instances, gen_id):
    """Returns (instances with frame/kind/n_roled attached, n dropped)."""
    idx = frame_index(gen_id)
    out, missing = [], 0
    for r in instances:
        meta = idx.get(_SUFFIX.sub("", r["sent_id"]))
        if meta is None:
            missing += 1
            continue
        out.append({**r, **meta})
    return out, missing


def _binary(label):
    return "NONE" if label == "NONE" else "POS"


def frame_label_table(instances):
    """Per frame: how many sentences and pairs it produced, and what share of those
    pairs are positive. A frame at 0.000 or 1.000 hands the label to any classifier
    that can name the frame."""
    sents = defaultdict(set)
    pairs = defaultdict(Counter)
    for r in instances:
        f = r.get("frame")
        sents[f].add(r["sent_id"])
        pairs[f][_binary(r["label"])] += 1

    rows = []
    for f, c in pairs.items():
        n = c["POS"] + c["NONE"]
        rows.append({"frame": f, "n_sent": len(sents[f]), "n_pairs": n,
                     "pos": c["POS"], "none": c["NONE"],
                     "pos_rate": c["POS"] / n if n else 0.0})
    rows.sort(key=lambda r: -r["n_pairs"])
    return rows


def _entropy(counter):
    from math import log2
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((v / total) * log2(v / total) for v in counter.values() if v)


def label_purity(instances):
    """Normalised mutual information between frame and binary pair label.

    1.0 means the frame fixes the label outright. The corpus figure computed the same
    way over LLM-assigned frames is the target, not zero: real constructions do carry
    some label signal, and driving this to zero would mean generating denials that are
    as often positive as negative, which is its own artefact.
    """
    pair_joint = Counter((r.get("frame"), _binary(r["label"])) for r in instances)
    pair_frame = Counter(f for f, _ in pair_joint.elements())
    pair_label = Counter(l for _, l in pair_joint.elements())

    sent_label, sent_frame = {}, {}
    for r in instances:
        sid = r["sent_id"]
        sent_frame[sid] = r.get("frame")
        if _binary(r["label"]) == "POS":
            sent_label[sid] = "POS"
        else:
            sent_label.setdefault(sid, "NONE")
    s_joint = Counter((sent_frame[s], sent_label[s]) for s in sent_frame)

    def nmi(joint):
        from math import log2
        total = sum(joint.values())
        if not total:
            return 0.0, 0.0
        px = Counter()
        py = Counter()
        for (x, y), v in joint.items():
            px[x] += v
            py[y] += v
        mi = 0.0
        for (x, y), v in joint.items():
            p = v / total
            mi += p * log2(p / ((px[x] / total) * (py[y] / total)))
        h = _entropy(py)
        return mi, (mi / h if h else 0.0)

    mi_p, n_p = nmi(pair_joint)
    mi_s, n_s = nmi(s_joint)

    rows = frame_label_table(instances)
    pure = [r["frame"] for r in rows if r["pos_rate"] in (0.0, 1.0)]
    near = [r["frame"] for r in rows if 0.0 < r["pos_rate"] < 0.03
            or 0.97 < r["pos_rate"] < 1.0]

    return {"nmi_pair": n_p, "mi_pair": mi_p, "h_label_pair": _entropy(pair_label),
            "nmi_sentence": n_s, "mi_sentence": mi_s,
            "n_frames": len(pair_frame),
            "pure_frames": pure, "near_pure_frames": near,
            "pure_share_of_pairs": sum(r["n_pairs"] for r in rows
                                       if r["frame"] in pure)
                                   / max(1, sum(r["n_pairs"] for r in rows))}


def _sentence_rows(instances, mask=True):
    """One row per sentence. Marker-stripped text is identical across a sentence's
    pairs, so anything learnable from it is sentence-level by construction."""
    seen, out = {}, []
    for r in instances:
        sid = r["sent_id"]
        if sid in seen:
            if _binary(r["label"]) == "POS":
                seen[sid]["sent_label"] = "POS"
            continue
        text = r["text"]
        text = (_E2.sub("drug2", _E1.sub("drug1", text)) if mask
                else _MARKER.sub("", text))
        row = {"sent_id": sid, "text": text, "frame": r.get("frame"),
               "sent_label": _binary(r["label"])}
        seen[sid] = row
        out.append(row)
    return out


def frame_recoverability(instances, seed=0, mask=True, min_count=6):
    """A: can a bag of words name the frame from the sentence alone.

    Entity spans are masked by default so the answer cannot come from the vocabulary
    sampler. Accuracy well above the majority baseline means every frame is writing in
    a signature the classifier can see.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    rows = [r for r in _sentence_rows(instances, mask=mask) if r["frame"]]

    # A frame seen once cannot be stratified, and a frame seen a handful of times
    # cannot be measured. At smoke-run sizes the rare assertions (incompatible and
    # contradictory are weighted 0.005) fall below this; dropping them is right, but
    # the accuracy that comes back is then over the frames that survived, not all of
    # them, so it is reported alongside the count.
    counts = Counter(r["frame"] for r in rows)
    keep = {f for f, n in counts.items() if n >= min_count}
    dropped = sorted(set(counts) - keep)
    rows = [r for r in rows if r["frame"] in keep]
    if len(keep) < 2 or len(rows) < 20:
        return {"error": "too few sentences per frame to measure",
                "n_frames_dropped": len(dropped), "dropped": dropped}

    tr, te = train_test_split(rows, test_size=0.3, random_state=seed,
                              stratify=[r["frame"] for r in rows])

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr = vec.fit_transform([r["text"] for r in tr])
    Xte = vec.transform([r["text"] for r in te])
    ytr = [r["frame"] for r in tr]
    yte = [r["frame"] for r in te]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    major = Counter(ytr).most_common(1)[0][0]
    return {"accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "majority_baseline": accuracy_score(yte, [major] * len(yte)),
            "n_frames": len(set(ytr)), "n_train": len(tr), "n_test": len(te),
            "n_frames_dropped": len(dropped), "dropped": dropped,
            "per_frame": _per_frame_recall(yte, pred)}


def _per_frame_recall(y, pred):
    hit, tot = Counter(), Counter()
    for a, b in zip(y, pred):
        tot[a] += 1
        hit[a] += int(a == b)
    return {f: hit[f] / tot[f] for f in sorted(tot, key=lambda k: -tot[k])}


def _best_macro_f1(y, score):
    """Highest macro-F1 over all thresholds. A hard majority-label rule degenerates here:
    every frame's majority pair label is NONE, because even a positive frame's
    non-participant pairs outnumber its focus pair. The frame carries the label as a
    rate, not as a verdict, so it has to be scored rather than voted."""
    from sklearn.metrics import f1_score
    import numpy as np
    order = sorted(set(score))
    best, best_t = 0.0, None
    for t in order:
        pred = ["POS" if s >= t else "NONE" for s in score]
        f = f1_score(y, pred, average="macro", zero_division=0)
        if f > best:
            best, best_t = f, t
    return best, best_t


def shortcut_via_frame(instances, seed=0, mask=True, min_count=6):
    """C: score every pair by the positive rate of its frame, using no pair information
    at any point, and see how well that alone separates POS from NONE.

    Two versions. `true_frame` uses the generator's own frame and is the ceiling: it is
    what a classifier would get if it could name the frame perfectly. `predicted_frame`
    predicts the frame from masked sentence text first, so it is what a classifier can
    actually reach from the surface.

    Compare `ap_predicted_frame` and `best_macro_f1_predicted_frame` against
    gates.shortcut_probe on the same dataset. If they are close, frame identity accounts
    for the shortcut and no amount of lexical variation inside a frame will remove it.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    import numpy as np

    rows = [r for r in instances if r.get("frame")]
    if not rows:
        return {}
    counts = Counter(r["frame"] for r in rows)
    rows = [r for r in rows if counts[r["frame"]] >= min_count]
    if len({r["frame"] for r in rows}) < 2:
        return {"error": "too few sentences per frame to measure"}
    sents = sorted({r["sent_id"] for r in rows})
    tr_s, te_s = train_test_split(sents, test_size=0.3, random_state=seed)
    tr_s, te_s = set(tr_s), set(te_s)

    sent_rows = {r["sent_id"]: r for r in _sentence_rows(rows, mask=mask)}
    tr = [sent_rows[s] for s in tr_s if s in sent_rows]
    te = [sent_rows[s] for s in te_s if s in sent_rows]
    if not tr or not te:
        return {}

    counts = defaultdict(Counter)
    for r in rows:
        if r["sent_id"] in tr_s:
            counts[r["frame"]][_binary(r["label"])] += 1
    prior = Counter()
    for c in counts.values():
        prior += c
    base_rate = prior["POS"] / max(1, sum(prior.values()))
    rate = {f: c["POS"] / max(1, c["POS"] + c["NONE"]) for f, c in counts.items()}

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr = vec.fit_transform([r["text"] for r in tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, [r["frame"] for r in tr])
    classes = list(clf.classes_)
    proba = clf.predict_proba(vec.transform([r["text"] for r in te]))
    expected = {r["sent_id"]: float(sum(p * rate.get(f, base_rate)
                                        for p, f in zip(row, classes)))
                for r, row in zip(te, proba)}

    y, s_pred, s_true = [], [], []
    for r in rows:
        if r["sent_id"] not in expected:
            continue
        y.append(_binary(r["label"]))
        s_pred.append(expected[r["sent_id"]])
        s_true.append(rate.get(r["frame"], base_rate))

    yb = [int(v == "POS") for v in y]
    f1_pred, t_pred = _best_macro_f1(y, s_pred)
    f1_true, _ = _best_macro_f1(y, s_true)

    from sklearn.metrics import f1_score
    major = "POS" if base_rate > 0.5 else "NONE"
    f1_base = f1_score(y, [major] * len(y), average="macro", zero_division=0)

    return {"ap_predicted_frame": average_precision_score(yb, s_pred),
            "ap_true_frame": average_precision_score(yb, s_true),
            "ap_baseline": float(np.mean(yb)),
            "best_macro_f1_predicted_frame": f1_pred,
            "best_macro_f1_true_frame": f1_true,
            "macro_f1_baseline": f1_base,
            "lift_predicted_frame": f1_pred - f1_base,
            "lift_true_frame": f1_true - f1_base,
            "threshold": t_pred,
            "n_pairs": len(y)}


def feature_frames(instances, features, min_count=20):
    """Which frame each probe feature belongs to.

    `features` is the (name, coef) list gates.shortcut_probe returns. For each feature,
    the frame whose sentences contain it most disproportionately, and the lift over that
    frame's base rate. Lift near 1 means the feature is spread across frames and is
    genuinely relation language; lift in the tens means the feature is a frame name.
    """
    rows = _sentence_rows(instances, mask=False)
    base = Counter(r["frame"] for r in rows)
    total = sum(base.values())

    out = []
    for name, coef in features:
        pat = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
        hits = Counter(r["frame"] for r in rows if pat.search(r["text"]))
        n = sum(hits.values())
        if n < min_count:
            out.append({"feature": name, "coef": coef, "n": n,
                        "frame": None, "share": None, "lift": None})
            continue
        f, c = hits.most_common(1)[0]
        share = c / n
        lift = share / (base[f] / total) if base[f] else float("inf")
        out.append({"feature": name, "coef": coef, "n": n, "frame": f,
                    "share": share, "lift": lift})
    return out


def role_gloss_report(instances):
    """The seven ROLES strings are rendered verbatim into every spec that carries a
    role, so they arrive in the output as fixed surface forms. Counts how many
    sentences contain each, and what share of the corpus-facing vocabulary they are.

    The reason they are fixed is that gates.role_position_skew and the role adjacency
    measurements locate roled entities by matching these strings. The instrument is
    holding the artefact in place. Recording the roled entity keys in the instance
    record would decouple the two and let the glosses vary.
    """
    from .prompt import ROLES
    rows = _sentence_rows(instances, mask=False)
    out = []
    for key, gloss in ROLES.items():
        stem = " ".join(gloss.split()[:4]).lower()
        n = sum(1 for r in rows if stem in r["text"].lower())
        out.append({"role": key, "gloss": gloss, "stem": stem,
                    "n_sentences": n, "share": n / max(1, len(rows))})
    out.sort(key=lambda r: -r["n_sentences"])
    return out