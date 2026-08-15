"""Mixing curve: does synthetic data help, and up to what human annotation budget.

The original question was "can synthetic data replace human annotation". Measured, the
answer is no: v14 scores 0.379 against human's 0.800. Reporting that alone is a negative
result about one generator.

The question a reader can act on is "given N human annotations, does adding synthetic
data help". Nobody generating synthetic data has zero labels, because with zero labels
they could not evaluate. So the useful axis is annotation budget, not annotation
presence, and the deliverable is the N at which the synthetic contribution crosses zero.

Two curves, deliberately kept apart:

  A. budget as TRAINING data. N human instances train the model, synthetic is fixed.
     This module. Answers "does synthetic augment a small human set".

  B. budget as GENERATOR input. N human instances estimate priors and supply exemplars,
     generation runs against those, and the model trains on the result. Not here: it
     needs a generation run per N. Curve A's shape decides whether B is worth building.

Subsampling is by SENTENCE, not instance. A sentence's pairs are not independent, and
splitting them would inflate the effective sample size and break the composition that
the whole project is about.
"""
import random
import statistics
from collections import Counter

BUDGETS = [100, 250, 500, 1000, 2500, None]     # None = all human sentences
SEEDS_SMALL = [0, 1, 2, 3, 4]                   # below 500 sentences, variance is large
SEEDS = [0, 1, 2]


def subsample_by_sentence(instances, n_sentences, seed):
    """Keep every instance from n randomly chosen sentences."""
    sents = sorted({r["sent_id"] for r in instances})
    if n_sentences is None or n_sentences >= len(sents):
        return list(instances)
    keep = set(random.Random(seed).sample(sents, n_sentences))
    return [r for r in instances if r["sent_id"] in keep]


def match_negative_ratio(instances, target_ratio, seed=0):
    """Downsample NONE instances so the positive rate matches target_ratio.

    Mixing changes class balance: v14 is ~19% positive, human 9.6%. Run the curve
    uncontrolled first, because that is what a practitioner would actually do, then
    repeat the crossover region with this applied. If the two disagree, the balance is
    doing the work rather than the data, which is itself the finding.
    """
    pos = [r for r in instances if r["label"] != "NONE"]
    neg = [r for r in instances if r["label"] == "NONE"]
    want_neg = int(len(pos) * (1 - target_ratio) / max(target_ratio, 1e-9))
    if want_neg >= len(neg):
        return list(instances)
    return pos + random.Random(seed).sample(neg, want_neg)


def positive_rate(instances):
    n = len(instances)
    return sum(1 for r in instances if r["label"] != "NONE") / max(n, 1)


def run_curve(human, synth, dev, train_and_eval, base_cfg,
              budgets=BUDGETS, match_balance=False, log=None):
    """Train human-only and human+synthetic at each budget. Returns a list of row dicts.

    train_and_eval : ddi.train.train_and_eval
    base_cfg       : dict without seed/dataset
    log            : ddi.experiment.log_run, or None to skip run records
    """
    target = positive_rate(human) if match_balance else None
    rows = []

    for n_sent in budgets:
        seeds = SEEDS_SMALL if (n_sent is not None and n_sent < 500) else SEEDS
        label = "full" if n_sent is None else str(n_sent)

        for seed in seeds:
            sub = subsample_by_sentence(human, n_sent, seed)

            arms = {"human": sub, "human+synth": sub + list(synth)}
            for arm, data in arms.items():
                if match_balance and arm != "human":
                    data = match_negative_ratio(data, target, seed)
                cfg = {**base_cfg, "seed": seed, "dataset": arm,
                       "budget_sentences": n_sent, "match_balance": match_balance}
                m = train_and_eval(cfg, data, dev)
                if log:
                    log(cfg, m, notes=f"mixing curve, budget={label}, arm={arm}")
                rows.append({"budget": label, "n_sentences": n_sent or len(
                                 {r['sent_id'] for r in human}),
                             "arm": arm, "seed": seed,
                             "n_instances": len(data),
                             "pos_rate": round(positive_rate(data), 4),
                             "f1": m["micro_f1_pos"], "p": m["micro_p_pos"],
                             "r": m["micro_r_pos"]})
                print(f"  {label:>5} {arm:<12} seed={seed} "
                      f"n={len(data):>6} f1={m['micro_f1_pos']:.3f}")
    return rows


def summarise(rows):
    """Delta between human+synth and human at each budget, with a pooled sd.

    Read the sign and where it crosses zero. Report P and R separately: the v13-to-v14
    transition showed F1 hiding a precision-recall trade.
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    piv = df.groupby(["budget", "n_sentences", "arm"])[["f1", "p", "r"]].agg(
        ["mean", "std"]).reset_index()

    out = []
    for (budget, n_sent), g in df.groupby(["budget", "n_sentences"]):
        h = g[g.arm == "human"]
        s = g[g.arm == "human+synth"]
        if h.empty or s.empty:
            continue
        d_f1 = s.f1.mean() - h.f1.mean()
        sd = statistics.sqrt(h.f1.std(ddof=1) ** 2 + s.f1.std(ddof=1) ** 2) \
            if len(h) > 1 and len(s) > 1 else float("nan")
        out.append({"budget": budget, "n_sentences": n_sent,
                    "human_f1": round(h.f1.mean(), 4),
                    "mixed_f1": round(s.f1.mean(), 4),
                    "delta_f1": round(d_f1, 4),
                    "pooled_sd": round(sd, 4),
                    "delta_p": round(s.p.mean() - h.p.mean(), 4),
                    "delta_r": round(s.r.mean() - h.r.mean(), 4)})
    return pd.DataFrame(out).sort_values("n_sentences"), piv