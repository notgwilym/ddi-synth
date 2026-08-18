"""Prune synthetic instances the verifier disagreed with, and test whether it helps.

WHAT PRUNING IS ACTUALLY DOING HERE
-----------------------------------
The verifier flags 7.4% of v14's NONE pairs and 5.8% of v15's. Its own false-positive
rate on human dev is 8.0% (NONE recall 0.920), so both flag rates sit at or below the
noise floor and the corrected drift estimate is zero.

That means pruning removes mostly correct labels. At a 3% true drift rate roughly 80% of
flags are the verifier being wrong. So this is not cleaning, and the random-pruned
control is what makes the result interpretable: if random removal does the same thing,
the verifier contributed nothing and the effect is data quantity.

Three reasons to run it anyway. The calibration was on real corpus text and synthetic
sentences are simpler, so the verifier may be more accurate on them than 0.920. If
pruning helps despite being mostly wrong, that says the flagged pairs are harder or more
ambiguous regardless of label correctness. And it is the arm anyone would ask about.

WHAT WAS ALREADY LOST BEFORE PRUNING
------------------------------------
Three separate losses stack, and conflating them with "verifier said this is wrong"
would make the arms incomparable:

  stage 2 rejects     147 for v15. Never entered the dataset at all.
  MAX_PAIRS cap       enumeration sentences excluded from the verifier batches, so their
                      pairs are in the dataset but unjudged.
  errored batches     141 v14, 193 v15, dropped whole by load_verdicts. Every pair in
                      those sentences is unjudged.

So the arms are built on JUDGED instances only. An unjudged instance is not evidence of
cleanliness, and treating it as such biases toward finding no effect.
"""
import random
from collections import defaultdict

from .verify_binary import _marked_spans, _sentence_spans, group_by_sentence


def instance_keys(instances):
    """(sent_id, m1, m2) for every instance, using the same mention numbering
    build_batches used, so verdicts can be mapped back.

    Returns (keys, n_unmappable). Unmappable means nested or overlapping spans, the same
    cases build_batches skipped.
    """
    keys, bad = [], 0
    for sent_id, rows in group_by_sentence(instances).items():
        spans = _sentence_spans(rows)
        idx = {s: i for i, s in enumerate(spans, start=1)}
        for r in rows:
            s = _marked_spans(r["text"])
            if len(s) != 2 or any(x not in idx for x in s):
                keys.append(None)
                bad += 1
            else:
                keys.append((sent_id, idx[s[0]], idx[s[1]]))
    return keys, bad


def align(instances, verdicts):
    """Attach verdicts to instances. Returns (judged, unjudged, flagged).

    judged   instances the verifier gave a verdict on
    flagged  judged AND gold NONE AND verifier said an interaction is annotated
    """
    by_sent = group_by_sentence(instances)
    ordered = [r for sent_id in by_sent for r in by_sent[sent_id]]
    keys, bad = instance_keys(instances)

    seen = {(v.sent_id, v.m1, v.m2): bool(v.flagged)
            for v in verdicts.itertuples()}

    judged, unjudged, flagged = [], [], []
    for r, k in zip(ordered, keys):
        if k is None or k not in seen:
            unjudged.append(r)
            continue
        judged.append(r)
        if r["label"] == "NONE" and seen[k]:
            flagged.append(r)

    print(f"  {len(instances)} instances: {len(judged)} judged, "
          f"{len(unjudged)} unjudged ({bad} unmappable spans)")
    print(f"  {len(flagged)} flagged NONE "
          f"({len(flagged) / max(sum(1 for r in judged if r['label'] == 'NONE'), 1):.4f} "
          f"of judged NONE)")
    return judged, unjudged, flagged


def prune(judged, flagged):
    drop = {id(r) for r in flagged}
    return [r for r in judged if id(r) not in drop]


def prune_random(judged, n_drop, seed=0):
    """Control: remove the same number of NONE instances at random. If this performs
    like the verifier-pruned arm, the verifier contributed nothing."""
    none_idx = [i for i, r in enumerate(judged) if r["label"] == "NONE"]
    drop = set(random.Random(seed).sample(none_idx, min(n_drop, len(none_idx))))
    return [r for i, r in enumerate(judged) if i not in drop]


def build_arms(instances, verdicts, seed=0):
    judged, unjudged, flagged = align(instances, verdicts)
    return {
        "judged": judged,
        "judged-pruned": prune(judged, flagged),
        "judged-randprune": prune_random(judged, len(flagged), seed),
    }