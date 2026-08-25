"""Frame diagnosis for v17.

Two halves. The structural half needs no LLM and runs in seconds:

    python scripts/diagnose_frames.py structural --gen-id <gen_id> --dataset <dataset_id>

The distributional half classifies real corpus sentences and needs the API:

    python scripts/diagnose_frames.py gold --n 300
    python scripts/diagnose_frames.py compare --gen-id <gen_id> --dataset <dataset_id>

`gold` is resumable and costs one call per sentence. `compare` reads whatever `gold`
has written so far, so it can be run against a partial file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddi import frame_diag, gold_frames, gates
from ddi.data import build_human
from ddi.manifest import load_dataset
from ddi.prompt import FRAMES

OUT = Path("runs/frames")


def _pct(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def _table(rows, cols):
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def frame_diag_probe(instances, seed):
    return gates.shortcut_probe(instances, seed=seed)


def structural(args):
    instances, _ = load_dataset(args.dataset)
    instances, missing = frame_diag.attach_frames(instances, args.gen_id)
    print(f"{len(instances)} instances joined to a frame, {missing} unjoined\n")

    print("== frame -> label ==")
    rows = frame_diag.frame_label_table(instances)
    _table([{**r, "pos_rate": _pct(r["pos_rate"])} for r in rows],
           ["frame", "n_sent", "n_pairs", "pos", "none", "pos_rate"])

    pur = frame_diag.label_purity(instances)
    print(f"\nnormalised MI, frame vs pair label   {pur['nmi_pair']:.3f}")
    print(f"normalised MI, frame vs sent label   {pur['nmi_sentence']:.3f}")
    print(f"label-pure frames                    {len(pur['pure_frames'])}"
          f" of {pur['n_frames']}, covering "
          f"{pur['pure_share_of_pairs']:.3f} of pairs")
    print(f"  {', '.join(pur['pure_frames'])}")
    if pur["near_pure_frames"]:
        print(f"near-pure  {', '.join(pur['near_pure_frames'])}")

    print("\n== A: frame recoverability from masked text ==")
    rec = frame_diag.frame_recoverability(instances, seed=args.seed)
    print(f"accuracy {rec['accuracy']:.3f}  macro-F1 {rec['macro_f1']:.3f}  "
          f"majority {rec['majority_baseline']:.3f}  over {rec['n_frames']} frames")
    worst = sorted(rec["per_frame"].items(), key=lambda kv: kv[1])[:5]
    best = sorted(rec["per_frame"].items(), key=lambda kv: -kv[1])[:5]
    print("  most recoverable  " + ", ".join(f"{k} {v:.2f}" for k, v in best))
    print("  least recoverable " + ", ".join(f"{k} {v:.2f}" for k, v in worst))

    print("\n== C: label scored from frame alone, no pair information ==")
    sc = frame_diag.shortcut_via_frame(instances, seed=args.seed)
    print(f"average precision, predicted frame  {sc['ap_predicted_frame']:.3f}")
    print(f"average precision, true frame       {sc['ap_true_frame']:.3f}")
    print(f"average precision, base rate        {sc['ap_baseline']:.3f}")
    print(f"best macro-F1, predicted frame      "
          f"{sc['best_macro_f1_predicted_frame']:.3f} "
          f"(lift {sc['lift_predicted_frame']:.3f})")
    lift, top = frame_diag_probe(instances, args.seed)
    print(f"direct shortcut probe lift          {lift:.3f}")
    print("\nIf the frame lift and the probe lift are close, frame identity accounts "
          "for the shortcut\nand varying the wording inside a frame will not remove it.")

    print("\n== probe features attributed to frames ==")
    feats = frame_diag.feature_frames(instances, top)
    _table([{**f, "share": _pct(f["share"], 2), "lift": _pct(f["lift"], 1)}
            for f in feats],
           ["feature", "coef", "n", "frame", "share", "lift"])

    print("\n== role glosses ==")
    _table([{**r, "share": _pct(r["share"], 3)} for r in
            frame_diag.role_gloss_report(instances)],
           ["role", "n_sentences", "share", "gloss"])


def gold(args):
    from openai import OpenAI

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gold_frames_n{args.n}_seed{args.seed}.jsonl"

    train, dev, val = build_human(seed=args.split_seed)
    pool = {"train": train, "dev": dev, "val": val}[args.split]
    items = gold_frames.sample_sentences(pool, n=args.n, seed=args.seed)
    print(f"{len(items)} sentences sampled from {args.split}")

    client = OpenAI(base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1",
                    api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=60.0)
    classify = gold_frames.make_frame_classifier(client, model=args.model,
                                                 api=args.api)
    n, errors = gold_frames.run(items, classify, path, workers=args.workers)
    print(f"{n} classified, {errors} errors -> {path}")


def compare(args):
    path = args.verdicts or sorted(OUT.glob("gold_frames_*.jsonl"))[-1]
    verdicts = gold_frames.load(path)
    print(f"{len(verdicts)} corpus sentences classified, from {path}\n")

    train, dev, val = build_human(seed=args.split_seed)
    pool = {"train": train, "dev": dev, "val": val}[args.split]

    corpus_primary, counts = gold_frames.corpus_distribution(verdicts)
    corpus_any, _ = gold_frames.corpus_distribution(verdicts, include_secondary=True)

    synth_share = {}
    if args.dataset and args.gen_id:
        raw, _meta = load_dataset(args.dataset)
        instances, _ = frame_diag.attach_frames(raw, args.gen_id)
        rows = frame_diag.frame_label_table(instances)
        total = sum(r["n_sent"] for r in rows)
        synth_share = {r["frame"]: r["n_sent"] / total for r in rows}
        synth_purity = {r["frame"]: r["pos_rate"] for r in rows}
    else:
        synth_purity = {}

    print("== frame distribution ==")
    names = sorted(set(FRAMES) | set(corpus_primary) | set(corpus_any) | {"other"},
                   key=lambda k: -corpus_any.get(k, 0))
    _table([{"frame": k,
             "weight": _pct(FRAMES.get(k, {}).get("w"), 2),
             "synth": _pct(synth_share.get(k), 3),
             "corpus_primary": _pct(corpus_primary.get(k, 0.0)),
             "corpus_any": _pct(corpus_any.get(k, 0.0)),
             "n": counts.get(k, 0)} for k in names],
           ["frame", "weight", "synth", "corpus_primary", "corpus_any", "n"])
    print("\nweight is the sampler's target, synth is what it realised, corpus_primary "
          "is the main clause, corpus_any counts secondary constructions too.")

    print("\n== purity: positive rate of pairs in sentences carrying each construction ==")
    cp = gold_frames.corpus_purity(verdicts, pool)
    _table([{"frame": r["frame"], "n_pairs": r["n_pairs"],
             "corpus_pos_rate": _pct(r["pos_rate"]),
             "synth_pos_rate": _pct(synth_purity.get(r["frame"]))} for r in cp],
           ["frame", "n_pairs", "corpus_pos_rate", "synth_pos_rate"])
    print("\nEvery synth figure is 0.000 or 1.000. The corpus column is the target.")

    print("\n== how many things a real sentence does at once ==")
    mc = gold_frames.multi_construction_rate(verdicts)
    for k, v in mc.items():
        print(f"{k:55s} {v if isinstance(v, int) else round(v, 3)}")
    print("v17 sentences carry exactly one construction by construction.")

    others = [v for v in verdicts if v.get("primary") == "other"]
    print(f"\n== {len(others)} sentences matched no frame "
          f"({len(others) / max(1, len(verdicts)):.3f}) ==")
    for v in others[:args.show_other]:
        print(f"\n  {v['other_description']}")
        print(f"    {v['text'][:200]}")

    if args.propose and others:
        from openai import OpenAI
        client = OpenAI(base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1",
                    api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=60.0)
        print("\n== candidate frames ==")
        for g in gold_frames.propose_frames(client, verdicts, model=args.model):
            print(f"\n  {g['name']}  (n={g['n']})")
            print(f"    {g['description']}")
            print(f"    e.g. {g['example']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("structural")
    s.add_argument("--gen-id", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=structural)

    g = sub.add_parser("gold")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--split", default="train")
    g.add_argument("--split-seed", type=int, default=42)
    g.add_argument("--model", default="gpt-oss-120b")
    g.add_argument("--api", default="responses")
    g.add_argument("--workers", type=int, default=8)
    g.set_defaults(fn=gold)

    c = sub.add_parser("compare")
    c.add_argument("--verdicts")
    c.add_argument("--gen-id")
    c.add_argument("--dataset")
    c.add_argument("--split", default="train")
    c.add_argument("--split-seed", type=int, default=42)
    c.add_argument("--model", default="gpt-oss-120b")
    c.add_argument("--show-other", type=int, default=25)
    c.add_argument("--propose", action="store_true")
    c.set_defaults(fn=compare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()