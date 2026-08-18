"""Wait for gpt-oss-120b, then calibrate the binary verifier on human dev.

The cluster autoscaler is the constraint, not the code. From the scheduler log: jobs sit
in PENDING for ~305s and get killed when no GPU is free, and a worker that IS up gets
reclaimed for idleness ("Backing off, underloaded, draining 0 requests") after a few
hours. So the strategy is to poll until a worker answers and start immediately, rather
than checking by hand and losing the window.

Once the run starts, the steady request stream keeps the worker alive, which is why this
does not stop to inspect anything mid-run.

    tmux new -s verify
    cd /mnt/primary/synth_data_creation
    mkdir -p logs
    python scripts/verify_calibrate.py 2>&1 | tee logs/verify-calibrate.log
    # ctrl-b then d to detach

    tmux attach -t verify                      # come back
    tail -f logs/verify-calibrate.log          # or watch from anywhere
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddi.data import build_human
from ddi.synth import generate_raw, RAW
from ddi.verify_binary import (build_batches, make_binary_verifier, load_verdicts,
                               calibration_report)


def wait_for_worker(client, model, effort, every=120, give_up_after=None):
    """Poll until the worker answers. give_up_after in seconds, None to wait forever."""
    t0 = time.time()
    n = 0
    while True:
        try:
            client.responses.create(model=model, input="ping", max_output_tokens=16,
                                    reasoning={"effort": effort} if effort else {})
            print(f"[{time.strftime('%H:%M:%S')}] {model} up "
                  f"after {(time.time() - t0) / 60:.0f} min, {n} attempts", flush=True)
            return True
        except Exception as e:
            n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
            if give_up_after and time.time() - t0 > give_up_after:
                print("giving up", flush=True)
                return False
            time.sleep(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-id", default="verify-bin-humandev")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--effort", default="high",
                    help="high is not optional: at low the five-way verifier's NONE "
                         "recall collapsed to 0.45-0.48 and prompt changes did not fix it")
    ap.add_argument("--api", default="responses", choices=["responses", "chat"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-pairs", type=int, default=30,
                    help="skip the enumeration sentences; the largest dev batch is 561 "
                         "pairs and those are the flattened tables, not what this measures")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--give-up-after", type=int, default=None,
                    help="seconds; omit to wait indefinitely")
    ap.add_argument("--base-rate", type=float, default=0.03,
                    help="assumed true drift rate, for the flag-precision projection")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("IDA_LLM_BASE_URL",
                                "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
        api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=180.0)

    train, dev, val = build_human()
    batches = [b for b in build_batches(dev) if len(b["pairs"]) <= args.max_pairs]
    dropped = len(build_batches(dev)) - len(batches)
    n_pairs = sum(len(b["pairs"]) for b in batches)
    print(f"{len(batches)} batches, {n_pairs} pairs "
          f"({dropped} sentences dropped over {args.max_pairs} pairs)", flush=True)

    if not wait_for_worker(client, args.model, args.effort, args.poll,
                           args.give_up_after):
        sys.exit(1)

    verify_fn = make_binary_verifier(client, model=args.model,
                                     reasoning_effort=args.effort, api=args.api)

    t0 = time.time()
    generate_raw(batches, verify_fn, gen_id=args.gen_id, max_workers=args.workers)
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min", flush=True)

    d = load_verdicts(args.gen_id)
    if d.empty:
        print("no usable verdicts", flush=True)
        sys.exit(1)

    cal = calibration_report(d, base_rate=args.base_rate)

    print("\nfive-way verifier for comparison: 0.82 overall, NONE recall 0.81")
    if cal["none_recall"] >= 0.95:
        print("\nPASS. Binary reform worked. Next: run it on v14 and v15 NONE pairs and "
              "compare corrected drift rates.")
    else:
        print("\nFAIL. NONE recall below 0.95, so the verifier's own false positives "
              "will outnumber the drifted pairs it is meant to find. The label-noise "
              "hypothesis cannot be tested this way and the v15 regression needs "
              "another explanation.")

    # same-entity pairs inflate the false-positive rate for a reason that has nothing to
    # do with the generator: DDI-2013 annotates every mention, so "corticosteroids" and
    # "corticosteroid" appear as a pair and gold is NONE
    same = d[d.e1.str.lower().str.rstrip("s") == d.e2.str.lower().str.rstrip("s")]
    if len(same):
        print(f"\n{len(same)} same-entity pairs ({len(same) / len(d):.3f}), "
              f"{same.asserted.mean():.3f} flagged")
        d2 = d.drop(same.index)
        print("excluding them:")
        calibration_report(d2, base_rate=args.base_rate)


if __name__ == "__main__":
    main()