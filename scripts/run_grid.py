"""The mixing grid: what should be added to a small amount of human-labelled data?

Run from the repo root, in a pod with a GPU:

    python scripts/run_grid.py --dry-run     # the plan, real pool sizes, time estimate; trains nothing
    python scripts/run_grid.py               # train, resumable, one row per seed to reports/grid.jsonl
    python scripts/run_grid.py --summary     # read the results back

THE QUESTION

Given B sentences with human labels, is it better to add nothing, to have the LLM label
the rest of the available real sentences, to add generated sentences, or to do both? The
exchange-rate runs compared human and LLM labels as substitutes. This compares them as
complements, which is the situation a practitioner is actually in.

Every arm is evaluated on val. Test is not touched.

TWO BUGS IN THE PREVIOUS VERSION, BOTH FIXED HERE

The human pool and the LLM-labelled pool are the same sentences with different labels.
The old grid added the whole LLM pool to the human subsample, so every human-labelled
sentence also appeared a second time with the LLM's label. Here the LLM labels only the
sentences the human budget did not cover, so every sentence appears exactly once, and an
`llm` arm always has exactly as many instances as the LLM pool. The dry run checks that.

mixing.match_negative_ratio can only drop negatives, so it can only raise the positive
rate. LLM labels run near 0.24 positive against a corpus target of 0.162, so it returned
the data unchanged, and the old balanced control arms were identical to the uncontrolled
ones. match_positive_rate below moves the rate in either direction. Lowering it means
dropping positives, which costs positive examples; that is the price of the control and
worth stating when reporting it.

TRAINING LENGTH

Every arm trains for three epochs or MIN_STEPS optimisation steps, whichever is more.
With a fixed number of epochs, arm size sets the number of steps, and the smallest
human-only arms collapse to predicting NONE for every pair.

ORDER

Arms run in order of how much they matter, so a pod that dies or a run that is stopped
early has still answered the main question: human alone, then human plus LLM labels,
then human plus generated text, then the balance control, then both together.

Rows go to reports/grid.jsonl rather than final_arms.jsonl, because another pod may be
appending to that file on the same NFS mount.

PART B: FILTERED LLM LABELS

Once 2c has finished, the same budgets are run with an `llmf` addition: the LLM labels
with every positive the verifier rejected demoted to NONE. By default the script runs
part A, then waits for 2c to finish, then runs part B, so it can be left overnight.

2c ran the verifier over the annotator's positive rows only, and build_batches numbers
mentions over the rows it is given, so the verifier's mention numbers do not match the
numbering over a whole sentence. The old cell 4 looked verdicts up with instance_keys
over the full pool, which put each rejection on the wrong pair, usually one already NONE,
so the filtered arm came out almost identical to the unfiltered one. Here each verdict is
turned back into character spans using the numbering 2c actually used, and matched on
those. The dry run checks that every rejection lands on exactly one positive instance.
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ddi import label_real
from ddi.data import build_human
from ddi.manifest import load_dataset
from ddi.mixing import subsample_by_sentence, positive_rate
from ddi.synth import RAW
from ddi.verify_binary import (load_verdicts, build_batches, group_by_sentence,
                               _sentence_spans, _marked_spans)

V18_ID = "20260825-095703-67c1fa"            # v18 low effort, 5,887 sentences
LABEL_VERDICTS = RAW / "label-high-c8.jsonl"  # the original annotator, not 2b or 2c
OUT = REPO / "reports" / "grid.jsonl"
TARGET_POS = 0.162                            # corpus pair positive rate
BUDGETS = [0, 100, 250, 500, 1000]
BALANCED_AT = [100, 500]
VERIFY_GEN = "label-pos-verify"            # 2c
POLL_MIN = 5

BASE = {"model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "epochs": 3, "lr": 2e-5, "batch_size": 32, "max_length": 256,
        "neg_ratio": None, "render_mode": "markers"}

# Measured on the v18 pruning runs: 12.6 it/s at batch 32, plus about 30 s of loading and
# evaluation per run. An estimate for planning, nothing more.
SEC_PER_STEP = 1 / 12.6
SEC_OVERHEAD = 30

# Fixed epochs make the size of an arm set how many optimisation steps it gets, so a
# 100-sentence arm trained for three epochs sees about 60 steps and collapses to NONE while
# a 14,758-instance arm sees about 1,380. Every arm therefore trains for three epochs or
# MIN_STEPS steps, whichever is more. Arms of 1,000 human sentences and above already
# exceed the floor and are unchanged. The value is a judgement: b1000-none reached 0.764
# at about 585 steps.
MIN_STEPS = 500


def epochs_for(n):
    per_epoch = max(1, -(-n // BASE["batch_size"]))
    return max(BASE["epochs"], -(-MIN_STEPS // per_epoch))


def run_secs(n):
    e = epochs_for(n)
    return SEC_OVERHEAD + e * -(-n // BASE["batch_size"]) * SEC_PER_STEP


def seeds_for(budget):
    """Five seeds where human data is scarce and variance is high, three elsewhere."""
    return (0, 1, 2, 3, 4) if 0 < budget < 500 else (0, 1, 2)


def match_positive_rate(instances, target, seed):
    """Resample so the positive rate equals target, dropping whichever class is in excess."""
    pos = [r for r in instances if r["label"] != "NONE"]
    neg = [r for r in instances if r["label"] == "NONE"]
    rng = random.Random(seed)
    if len(pos) / max(len(instances), 1) > target:
        return rng.sample(pos, int(len(neg) * target / (1 - target))) + neg
    return pos + rng.sample(neg, min(len(neg), int(len(pos) * (1 - target) / target)))


def load_pools():
    train, _, val = build_human()
    label_inst, dropped = label_real.to_instances(train, LABEL_VERDICTS)
    H = [{**r, "label": r["gold_label"]} for r in label_inst]
    L = label_inst
    S = load_dataset(V18_ID)[0]
    print(f"pools: human {len(H)} / LLM {len(L)} instances over "
          f"{len({r['sent_id'] for r in H})} sentences ({dropped} dropped), "
          f"v18 {len(S)} over {len({r['sent_id'] for r in S})}, val {len(val)}")
    return H, L, S, val


def build(budget, addition, seed, H, L, S, balanced=False, LF=None):
    sub = subsample_by_sentence(H, budget, seed) if budget else []
    used = {r["sent_id"] for r in sub}
    pool = LF if addition == "llmf" else L
    rest = [r for r in pool if r["sent_id"] not in used]
    add = {"none": [], "llm": rest, "llmf": rest, "synth": S, "both": S + rest}[addition]
    data = sub + add
    return match_positive_rate(data, TARGET_POS, seed) if balanced else data


def verify_status(L):
    """How far 2c has got: specs expected, attempted at least once, succeeded."""
    expected = {b["sent_id"] for b in build_batches([r for r in L if r["label"] != "NONE"])}
    path = RAW / f"{VERIFY_GEN}.jsonl"
    attempted, ok = set(), set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            sid = (rec.get("spec") or {}).get("sent_id")
            attempted.add(sid)
            if rec.get("sample") is not None and not rec.get("error"):
                ok.add(sid)
    return expected, attempted & expected, ok & expected


def filtered_pool(L):
    """L with every positive the verifier rejected demoted to NONE.

    Verdicts are mapped back by character span, using the mention numbering build_batches
    used when 2c ran, which was over each sentence's positive rows only."""
    numbering = {sid: _sentence_spans(rows) for sid, rows in
                 group_by_sentence([r for r in L if r["label"] != "NONE"]).items()}
    rejected = set()
    for v in load_verdicts(VERIFY_GEN).itertuples():
        spans = numbering.get(v.sent_id)
        if spans and not v.flagged:
            rejected.add((v.sent_id, spans[int(v.m1) - 1], spans[int(v.m2) - 1]))
    out, hit = [], set()
    for r in L:
        s = _marked_spans(r["text"])
        k = (r["sent_id"], s[0], s[1]) if len(s) == 2 else None
        if r["label"] != "NONE" and k in rejected:
            out.append({**r, "label": "NONE"})
            hit.add(k)
        else:
            out.append(r)
    return out, len(rejected), len(hit)


def wait_for_2c(L, hours):
    """Poll until every 2c spec has been attempted, or give up after `hours`."""
    deadline = time.time() + hours * 3600
    while True:
        expected, attempted, ok = verify_status(L)
        if expected and attempted == expected:
            print(f"2c finished: {len(ok)} of {len(expected)} sentences verified "
                  f"({len(expected) - len(ok)} errored, their positives left unfiltered)")
            return True
        if time.time() > deadline:
            print(f"gave up waiting: 2c at {len(attempted)} of {len(expected)}. "
                  f"Run `python scripts/run_grid.py --part b` once it has finished.")
            return False
        print(f"waiting for 2c: {len(attempted)} of {len(expected)} attempted", flush=True)
        time.sleep(POLL_MIN * 60)


def plan_b():
    return [(f"grid-b{b}-llmf", b, "llmf", False, s) for b in BUDGETS for s in seeds_for(b)]


def plan():
    order = []
    for addition, balanced in [("none", False), ("llm", False), ("synth", False),
                               ("llm", True), ("both", False)]:
        for b in BUDGETS:
            if b == 0 and addition == "none":
                continue
            if balanced and b not in BALANCED_AT:
                continue
            name = f"grid-b{b}-{addition}" + ("-bal" if balanced else "")
            for s in seeds_for(b):
                order.append((name, b, addition, balanced, s))
    return order


def done_runs():
    if not OUT.exists():
        return set()
    return {(r["arm"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())
            if r}


def dry_run(H, L, S):
    runs, done = plan(), done_runs()
    total, left = 0.0, 0.0
    print(f"\n{'arm':<22}{'seeds':>7}{'instances':>11}{'pos rate':>10}"
          f"{'epochs':>8}{'steps':>7}{'min/run':>9}")
    seen = set()
    for name, b, add, bal, s in runs:
        if name in seen:
            continue
        seen.add(name)
        data = build(b, add, 0, H, L, S, bal)
        if add in ("llm", "both") and not bal:
            # Every real sentence exactly once, human-labelled or LLM-labelled but never
            # both, so the real part of the arm is exactly the size of the LLM pool. The
            # old grid would fail this by the size of the human subsample.
            real = sum(1 for r in data if not r["sent_id"].startswith("synth:"))
            assert real == len(L), f"{name}: {real} real instances, expected {len(L)}"
        k = [x for x in runs if x[0] == name]
        mins = run_secs(len(data)) / 60
        e = epochs_for(len(data))
        steps = e * -(-len(data) // BASE["batch_size"])
        total += mins * len(k)
        left += mins * sum(1 for x in k if (x[0], x[4]) not in done)
        floor = "  raised to floor" if e > BASE["epochs"] else ""
        print(f"{name:<22}{len(k):>7}{len(data):>11}{positive_rate(data):>10.3f}"
              f"{e:>8}{steps:>7}{mins:>9.1f}{floor}")
    print(f"\n{len(runs)} runs, about {total / 60:.1f} h in total; "
          f"{len(runs) - len(done & {(x[0], x[4]) for x in runs})} left, about {left / 60:.1f} h")
    print("no human sentence appears twice in any llm or both arm")

    expected, attempted, ok = verify_status(L)
    print(f"\npart B: 2c has attempted {len(attempted)} of {len(expected)} sentences, "
          f"{len(ok)} succeeded")
    if expected and attempted == expected:
        LF, n_rej, n_hit = filtered_pool(L)
        assert n_hit == n_rej, (f"{n_rej} rejections but only {n_hit} landed on a positive "
                                f"instance; the verdict mapping is wrong, do not run part B")
        n_pos = sum(1 for r in L if r["label"] != "NONE")
        print(f"verifier rejected {n_rej} of {n_pos} LLM positives ({n_rej / n_pos:.1%}); "
              f"every rejection maps to exactly one positive instance")
        print(f"positive rate {positive_rate(L):.3f} -> {positive_rate(LF):.3f} after filtering")
        for b in BUDGETS:
            d = build(b, "llmf", 0, H, L, S, LF=LF)
            print(f"  grid-b{b}-llmf  {len(seeds_for(b))} seeds  {len(d)} instances  "
                  f"pos rate {positive_rate(d):.3f}")
    else:
        print("part B will run automatically once 2c has finished")


def train_all(H, L, S, val, runs, LF=None, label="part A"):
    import torch
    from ddi.train import train_and_eval
    from ddi.experiment import log_run

    if not torch.cuda.is_available():
        sys.exit("no GPU in this pod")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_runs()
    todo = [x for x in runs if (x[0], x[4]) not in done]
    print(f"{label}: {len(runs) - len(todo)} of {len(runs)} already on disk, {len(todo)} to go")

    for i, (name, b, add, bal, s) in enumerate(todo, 1):
        data = build(b, add, s, H, L, S, bal, LF)
        cfg = {**BASE, "seed": s, "dataset": name, "budget_sentences": b,
               "addition": add, "match_balance": bal, "epochs": epochs_for(len(data)),
               "min_steps": MIN_STEPS}
        t = time.time()
        m = train_and_eval(cfg, data, val)
        log_run(cfg, m, notes="grid v2, val")
        row = {"arm": name, "seed": s, "budget": b, "addition": add, "balanced": bal,
               "n": len(data), "pos_rate": round(positive_rate(data), 4),
               "f1": m["micro_f1_pos"], "p": m["micro_p_pos"], "r": m["micro_r_pos"],
               "epochs": cfg["epochs"], "secs": round(time.time() - t)}
        with open(OUT, "a") as fp:
            fp.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(todo)}] {name:<22} seed={s} f1={row['f1']:.3f} "
              f"p={row['p']:.3f} r={row['r']:.3f}  {row['epochs']}ep {row['secs']}s", flush=True)
        if row["p"] == 0 and row["r"] == 0:
            print(f"    WARNING: {name} seed {s} collapsed to all-NONE", flush=True)


def summary():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l]
    by, collapsed = defaultdict(list), defaultdict(int)
    for r in rows:
        by[r["arm"]].append(r["f1"])
        collapsed[r["arm"]] += int(r["p"] == 0 and r["r"] == 0)

    def cell(b, add, bal=False):
        f = by.get(f"grid-b{b}-{add}" + ("-bal" if bal else ""))
        if not f:
            return None
        return st.mean(f), (st.stdev(f) if len(f) > 1 else 0.0), len(f)

    adds = ["none", "llm", "llmf", "synth", "both"]
    print(f"{'budget':>7}" + "".join(f"{a:>16}" for a in adds) + f"{'llm-bal':>16}")
    for b in BUDGETS:
        cols = [cell(b, a) for a in adds] + [cell(b, "llm", True)]
        print(f"{b:>7}" + "".join(f"{'-':>16}" if c is None else
                                  f"{c[0]:>9.3f} ±{c[1]:.3f}" for c in cols))

    def gap(x, y):
        if not (x and y):
            return "-"
        se = (x[1] ** 2 / x[2] + y[1] ** 2 / y[2]) ** 0.5
        return f"{x[0] - y[0]:+.3f} ({(x[0] - y[0]) / se:.1f} se)" if se else f"{x[0] - y[0]:+.3f}"

    bad = {a: c for a, c in collapsed.items() if c}
    if bad:
        print("\ncollapsed to all-NONE (seeds): " + ", ".join(f"{a} {c}" for a, c in sorted(bad.items())))
        print("these cells measure under-training, not the value of the data")
    print(f"\n{'budget':>7}{'llm - none':>22}{'llmf - llm':>22}{'synth - none':>22}"
          f"{'both - llm':>22}{'llm - llm-bal':>22}")
    for b in BUDGETS:
        print(f"{b:>7}{gap(cell(b, 'llm'), cell(b, 'none')):>22}"
              f"{gap(cell(b, 'llmf'), cell(b, 'llm')):>22}"
              f"{gap(cell(b, 'synth'), cell(b, 'none')):>22}"
              f"{gap(cell(b, 'both'), cell(b, 'llm')):>22}"
              f"{gap(cell(b, 'llm'), cell(b, 'llm', True)):>22}")
    print("""
llm - none      does having the LLM label the remaining real sentences help
llmf - llm      does filtering the LLM's positives through the verifier help (2c)
synth - none    does generated text help (the old mixing curve used v14, not v18)
both - llm      does generated text add anything once LLM-labelled real text exists
llm - llm-bal   near zero means class balance is not what the llm arm is winning on
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--wait-hours", type=float, default=10,
                    help="how long part B waits for 2c to finish before giving up")
    a = ap.parse_args()
    if a.summary:
        summary()
        sys.exit()
    H, L, S, val = load_pools()
    if a.dry_run:
        dry_run(H, L, S)
        sys.exit()
    if a.part in ("a", "both"):
        train_all(H, L, S, val, plan())
    if a.part in ("b", "both") and wait_for_2c(L, a.wait_hours):
        LF, n_rej, n_hit = filtered_pool(L)
        if n_hit != n_rej:
            sys.exit(f"{n_rej} rejections, {n_hit} landed on a positive; not running part B")
        print(f"part B: {n_rej} LLM positives demoted, pos rate "
              f"{positive_rate(L):.3f} -> {positive_rate(LF):.3f}")
        train_all(H, L, S, val, plan_b(), LF=LF, label="part B")
