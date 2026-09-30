"""One evaluation on the official DDI-2013 test set, for the configurations validation chose.

Run from the repo root, in a GPU pod, after 2c has finished:

    python scripts/run_test.py --dry-run     # checks, the frozen selection, sizes, time; no training
    python scripts/run_test.py               # train and score on test, resumable
    python scripts/run_test.py --summary     # read the results back

WHY ONCE

No experiment in the project has touched the test set. Everything was chosen on dev or
val, so a test number is the only score not shaped by the choices it is reporting. That
holds only if the choice is fixed before the first test number exists. The arms below
are written to reports/test_selection.json with a timestamp on the first run, and the
script refuses to run if the file on disk and this file ever disagree. If a different
configuration would have scored higher on test, the report still uses these.

THE ARMS, BY HOW MUCH HUMAN ANNOTATION THEY NEED

    no human annotation    verified LLM labels on all real sentences, the best on val at
                           a budget of 0; unverified LLM labels and v18 alone for comparison
    1,000 human sentences  verified LLM labels on the remaining sentences, the best on val
                           at 1,000; human labels alone for comparison
    all human annotation   every sentence in the pool human-labelled

Five seeds each. Every arm trains for three epochs or 500 steps, whichever is more, as in
the grid.

THE SPLIT CHECK

data.load_brat_docs lists files with os.listdir and does not sort them, so the document
order fed to the seeded split depends on the filesystem. The split is stable on the pod's
NFS mount and different elsewhere. Every logged result used the split with 4,243 dev and
6,276 val pairs, so this script refuses to run unless build_human reproduces those sizes,
and it writes the document ids of each split to reports/split.json so the split can be
rebuilt on any machine. Do not fix the sort in data.py without that file: sorting would
change the split and break continuity with every logged number.
"""
import argparse
import json
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_grid as rg
from ddi.mixing import positive_rate

OUT = REPO / "reports" / "test.jsonl"
SELECTION = REPO / "reports" / "test_selection.json"
SPLIT = REPO / "reports" / "split.json"
SEEDS = (0, 1, 2, 3, 4)
EXPECTED = {"dev": 4243, "val": 6276}
FULL = "full"   # stands for every sentence in the pool

ARMS = [
    {"arm": "test-b0-llmf", "stage": "no human annotation", "budget": 0, "addition": "llmf",
     "why": "best on val at a budget of 0 (0.756)"},
    {"arm": "test-b0-llm", "stage": "no human annotation", "budget": 0, "addition": "llm",
     "why": "comparison: the same labels without verification (val 0.677)"},
    {"arm": "test-b0-synth", "stage": "no human annotation", "budget": 0, "addition": "synth",
     "why": "comparison: generated v18 text alone (val 0.465)"},
    {"arm": "test-b1000-llmf", "stage": "1,000 human sentences", "budget": 1000, "addition": "llmf",
     "why": "best on val at 1,000 (0.796)"},
    {"arm": "test-b1000-none", "stage": "1,000 human sentences", "budget": 1000, "addition": "none",
     "why": "comparison: human labels alone (val 0.764)"},
    {"arm": "test-bfull-none", "stage": "all human annotation", "budget": FULL, "addition": "none",
     "why": "ceiling: every sentence in the pool human-labelled"},
]


def build_test():
    import spacy
    from ddi.data import load_brat_docs, make_sentence_level, make_pair_instances
    nlp = spacy.load("en_core_web_sm")
    return [r for d in load_brat_docs(split="Test")
            for s in make_sentence_level(d, nlp) for r in make_pair_instances(s)]


def check_split():
    from ddi.data import build_human
    train, dev, val = build_human()
    sizes = {"dev": len(dev), "val": len(val)}
    if sizes != EXPECTED:
        sys.exit(f"split mismatch: build_human gives {sizes}, every logged result used {EXPECTED}. "
                 f"This machine lists the corpus files in a different order. Do not run.")
    if not SPLIT.exists():
        docs = lambda rows: sorted({":".join(r["sent_id"].split(":")[1:3]) for r in rows})
        SPLIT.write_text(json.dumps({"train": docs(train), "dev": docs(dev), "val": docs(val),
                                     "note": "document ids as register:doc, from the pod's split"}, indent=1))
        print(f"wrote {SPLIT}")
    print(f"split matches the ledger: dev {sizes['dev']}, val {sizes['val']}")


def freeze_selection():
    if SELECTION.exists():
        on_disk = json.loads(SELECTION.read_text())
        if [a["arm"] for a in on_disk["arms"]] != [a["arm"] for a in ARMS]:
            sys.exit(f"{SELECTION} lists different arms from this script. The selection is frozen "
                     f"once written; do not change it after test results exist.")
        print(f"selection frozen at {on_disk['frozen_at']}")
        return
    SELECTION.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "chosen_on": "validation, reports/grid.jsonl", "seeds": list(SEEDS),
                                     "arms": ARMS}, indent=1))
    print(f"selection frozen -> {SELECTION}. Commit it before reading any test number.")


def arm_data(a, seed, H, L, S, LF, n_pool):
    b = n_pool if a["budget"] == FULL else a["budget"]
    return rg.build(b, a["addition"], seed, H, L, S, LF=LF)


def done_runs():
    if not OUT.exists():
        return set()
    return {(r["arm"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines()) if r}


def dry_run(H, L, S, LF, n_pool, test):
    c = Counter(r["label"] for r in test)
    print(f"\ntest: {len(test)} pairs, {sum(v for k, v in c.items() if k != 'NONE')} positive")
    done, total = done_runs(), 0.0
    print(f"\n{'arm':<18}{'stage':<24}{'human sents':>12}{'instances':>11}{'pos rate':>10}{'epochs':>8}{'min/run':>9}")
    for a in ARMS:
        d = arm_data(a, 0, H, L, S, LF, n_pool)
        if a["addition"] in ("llm", "llmf"):
            real = sum(1 for r in d if not r["sent_id"].startswith("synth:"))
            assert real == len(L), f"{a['arm']}: {real} real instances, expected {len(L)}"
        mins = rg.run_secs(len(d)) / 60
        total += mins * sum(1 for s in SEEDS if (a["arm"], s) not in done)
        hs = n_pool if a["budget"] == FULL else a["budget"]
        print(f"{a['arm']:<18}{a['stage']:<24}{hs:>12,}{len(d):>11,}{positive_rate(d):>10.3f}"
              f"{rg.epochs_for(len(d)):>8}{mins:>9.1f}")
    print(f"\n{len(ARMS) * len(SEEDS)} runs, {len(done)} done, about {total / 60:.1f} h left")


def train(H, L, S, LF, n_pool, test):
    import torch
    from ddi.train import train_and_eval
    from ddi.experiment import log_run
    if not torch.cuda.is_available():
        sys.exit("no GPU in this pod")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_runs()
    todo = [(a, s) for a in ARMS for s in SEEDS if (a["arm"], s) not in done]
    print(f"{len(done)} rows on disk, {len(todo)} runs to go")
    for i, (a, s) in enumerate(todo, 1):
        data = arm_data(a, s, H, L, S, LF, n_pool)
        cfg = {**rg.BASE, "seed": s, "dataset": a["arm"], "epochs": rg.epochs_for(len(data)),
               "min_steps": rg.MIN_STEPS, "budget_sentences": a["budget"], "addition": a["addition"],
               "eval_split": "test"}
        t = time.time()
        m = train_and_eval(cfg, data, test)
        log_run(cfg, m, notes="TEST, val-selected")
        row = {"arm": a["arm"], "stage": a["stage"], "seed": s, "n": len(data),
               "human_sentences": n_pool if a["budget"] == FULL else a["budget"],
               "f1": m["micro_f1_pos"], "p": m["micro_p_pos"], "r": m["micro_r_pos"],
               "secs": round(time.time() - t)}
        with open(OUT, "a") as fp:
            fp.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(todo)}] {a['arm']:<18} seed={s} f1={row['f1']:.3f} "
              f"p={row['p']:.3f} r={row['r']:.3f}", flush=True)


def summary():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    S = {}
    for a in ARMS:
        rs = by.get(a["arm"], [])
        if not rs:
            continue
        f = [r["f1"] for r in rs]
        S[a["arm"]] = dict(f=st.mean(f), sd=st.stdev(f) if len(f) > 1 else 0.0, n=len(f),
                           p=st.mean(r["p"] for r in rs), r=st.mean(r["r"] for r in rs),
                           h=rs[0]["human_sentences"])
    full = S.get("test-bfull-none")
    print(f"\n{'arm':<18}{'human sents':>12}{'seeds':>7}{'F1':>8}{'sd':>7}{'P':>7}{'R':>7}{'of full human':>15}")
    for a in ARMS:
        x = S.get(a["arm"])
        if not x:
            continue
        rel = f"{x['f'] / full['f']:.1%}" if full else "-"
        print(f"{a['arm']:<18}{x['h']:>12,}{x['n']:>7}{x['f']:>8.3f}{x['sd']:>7.3f}{x['p']:>7.3f}{x['r']:>7.3f}{rel:>15}")

    def gap(a, b):
        x, y = S.get(a), S.get(b)
        if not (x and y):
            return "-"
        se = (x["sd"] ** 2 / x["n"] + y["sd"] ** 2 / y["n"]) ** 0.5
        return f"{x['f'] - y['f']:+.3f} ({(x['f'] - y['f']) / se:.1f} se)" if se else f"{x['f'] - y['f']:+.3f}"
    print(f"\nverified, no human labels  vs  1,000 human sentences alone : {gap('test-b0-llmf', 'test-b1000-none')}")
    print(f"verified vs unverified, no human labels                   : {gap('test-b0-llmf', 'test-b0-llm')}")
    print(f"verified + 1,000 human  vs  1,000 human alone              : {gap('test-b1000-llmf', 'test-b1000-none')}")
    print(f"verified, no human labels  vs  all human annotation        : {gap('test-b0-llmf', 'test-bfull-none')}")
    print(f"verified + 1,000 human  vs  all human annotation           : {gap('test-b1000-llmf', 'test-bfull-none')}")
    print(f"generated v18 alone  vs  all human annotation              : {gap('test-b0-synth', 'test-bfull-none')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
        sys.exit()
    check_split()
    freeze_selection()
    H, L, S, _val = rg.load_pools()
    LF, n_rej, n_hit = rg.filtered_pool(L)
    if n_rej != n_hit:
        sys.exit(f"{n_rej} verifier rejections but {n_hit} landed on a positive; not running")
    n_pool = len({r["sent_id"] for r in H})
    test = build_test()
    if a.dry_run:
        dry_run(H, L, S, LF, n_pool, test)
    else:
        train(H, L, S, LF, n_pool, test)
