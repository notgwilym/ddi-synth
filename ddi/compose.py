"""Is v15's regression composition or content?

v15 scored 0.333 against v14's 0.379, despite being closer to the corpus on almost every
divergence measurement. The label-noise hypothesis is dead: non-participant drift into
the asserted clause is 0.036 in v14 and 0.035 in v15.

Two confounds remain, and both are composition rather than prose:

  instance count     v14 18,482 -> v15 36,000 from the same 6,000 specs. Pairs grow
                     quadratically with entity count, and the entity cap went 5 -> 8.
                     The extra 17,500 are overwhelmingly NONE pairs from large sentences.
  positive rate      v14 0.186 -> v15 0.127, against a corpus 0.163
  hard negative rate v14 0.509 -> v15 0.684, against a corpus 0.502

62% of v15's instances come from sentences with 10+ pairs, where the positive rate is
0.077. So the model sees a great many negatives from interaction-bearing prose, which
should teach caution, but may simply be diluting the positive signal.

Four arms, all subsamples of the SAME v15 dataset, so nothing is regenerated:

  v15-full        as built, the 0.333 baseline
  v15-sized       subsampled by sentence to v14's instance count
  v15-composed    sentences dropped preferentially at high k until pairs/sentence,
                  positive rate and hard negative rate approach v14's
  v15-negcap      NONE pairs downsampled to v14's positive rate, sentences intact

If a matched arm recovers to 0.379 or above, the regression is composition and the
content changes were neutral or good. If everything stays near 0.333, the content
changes hurt and the divergence table is actively misleading about what matters.
"""
import random
from collections import Counter, defaultdict


def stats(inst, name=""):
    n_sent = len({r["sent_id"] for r in inst})
    pos = sum(1 for r in inst if r["label"] != "NONE")
    has_pos = defaultdict(bool)
    for r in inst:
        if r["label"] != "NONE":
            has_pos[r["sent_id"]] = True
    none = [r for r in inst if r["label"] == "NONE"]
    hard = sum(1 for r in none if has_pos[r["sent_id"]]) / max(len(none), 1)
    d = {"n": len(inst), "sents": n_sent,
         "pairs_per_sent": round(len(inst) / max(n_sent, 1), 2),
         "pos_rate": round(pos / max(len(inst), 1), 4),
         "hard_neg": round(hard, 4)}
    if name:
        print(f"{name:<14} {d['n']:>6} inst {d['sents']:>5} sents  "
              f"pairs/sent {d['pairs_per_sent']:>5}  pos {d['pos_rate']:.3f}  "
              f"hard_neg {d['hard_neg']:.3f}")
    return d


def subsample_to_instances(inst, n_target, seed=0):
    """Whole sentences until the instance budget is met. Sentence-level so the
    composition inside each generated sentence stays intact."""
    by_sent = defaultdict(list)
    for r in inst:
        by_sent[r["sent_id"]].append(r)
    sents = sorted(by_sent)
    random.Random(seed).shuffle(sents)
    keep, n = set(), 0
    for s in sents:
        if n >= n_target:
            break
        keep.add(s)
        n += len(by_sent[s])
    return [r for r in inst if r["sent_id"] in keep]


def match_composition(inst, target_pairs_per_sent, n_target, seed=0):
    """Drop sentences preferentially at high pair count until pairs/sentence approaches
    the target, then trim to the instance budget.

    v15's large sentences are where the composition diverges: at 10+ pairs the positive
    rate is 0.077 against 0.364 at one pair. Weighting the draw toward small sentences
    pulls pairs/sentence, positive rate and hard negative rate together.
    """
    by_sent = defaultdict(list)
    for r in inst:
        by_sent[r["sent_id"]].append(r)
    rng = random.Random(seed)

    # weight inversely to size, so small sentences are kept preferentially
    sents = sorted(by_sent)
    weights = [1.0 / max(len(by_sent[s]), 1) for s in sents]

    keep, n, tries = set(), 0, 0
    while n < n_target and tries < len(sents) * 20:
        tries += 1
        s = rng.choices(sents, weights=weights, k=1)[0]
        if s in keep:
            continue
        cur = n / max(len(keep), 1) if keep else 0
        # accept if it moves pairs/sentence toward the target, or if we are short
        if len(keep) < 50 or abs((n + len(by_sent[s])) / (len(keep) + 1)
                                 - target_pairs_per_sent) <= abs(cur - target_pairs_per_sent) + 0.5:
            keep.add(s)
            n += len(by_sent[s])
    return [r for r in inst if r["sent_id"] in keep]


def downsample_negatives(inst, target_pos_rate, seed=0):
    """Drop NONE instances at random until the positive rate matches. Breaks the
    within-sentence pair set, which is a real cost: the whole project is about pairs
    from one sentence differing only in their markers. Included as a contrast to
    match_composition, which keeps sentences whole."""
    pos = [r for r in inst if r["label"] != "NONE"]
    neg = [r for r in inst if r["label"] == "NONE"]
    want = int(len(pos) * (1 - target_pos_rate) / max(target_pos_rate, 1e-9))
    if want >= len(neg):
        return list(inst)
    return pos + random.Random(seed).sample(neg, want)