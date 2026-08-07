"""Long-running v14 generation, safe to detach.

tmux survives SSH disconnection. It does not survive the pod dying. What survives that
is generate_raw's append-and-flush plus resume=True: the raw jsonl lives on NFS, every
result is flushed as it arrives, and re-running the same command skips the spec indices
that already succeeded. Errored specs are retried, so a transient API failure does not
silently shrink the dataset.

make_v14_specs is a deterministic prefix: make_v14_specs(10000, seed=0)[:6000] is
identical to make_v14_specs(6000, seed=0), because the rng is consumed in order. So the
run can be extended later by raising --n with the same --gen-id and --seed.

DO NOT edit ddi/prompt14.py while a run is in flight. Specs are rebuilt from the module
on resume, so an edit mixes two generators into one raw file and the prompt fingerprint
stops describing the data.

    tmux new -s gen
    cd /mnt/primary/synth_data_creation
    mkdir -p logs
    python scripts/generate_v14.py --n 6000 --gen-id v14-full-1 --workers 16 \
        2>&1 | tee logs/v14-full-1.log
    # ctrl-b then d to detach; tmux attach -t gen to come back
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddi.vocab import build_vocab
from ddi.prompt14 import make_v14_specs, make_v14_sample_fn, v14_fingerprint
from ddi.resolve import v14_sample_to_instances, generation_records
from ddi.synth import generate_raw, build_dataset_from_raw
from ddi import gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000,
                    help="specs to generate. ~3.4 instances per surviving sentence, "
                         "~93%% survive, so 6000 gives roughly 19k instances")
    ap.add_argument("--gen-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--composition", default="prior", choices=["prior", "measured"])
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--api", default="responses", choices=["responses", "chat"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-output-tokens", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--no-build", action="store_true",
                    help="stage 1 only; stage 2 is deterministic and free to re-run")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("IDA_LLM_BASE_URL",
                                "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
        api_key=os.environ["IDA_LLM_API_KEY"],
        max_retries=5, timeout=args.timeout)

    # fail in 5 seconds rather than 6000 identical errors
    t0 = time.time()
    if args.api == "responses":
        client.responses.create(model=args.model, input="ping",
                                max_output_tokens=16,
                                reasoning={"effort": "low"} if args.effort else {})
    else:
        client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": "ping"}],
            max_tokens=16)
    print(f"{args.model} responding on /{args.api} ({time.time() - t0:.1f}s)")

    vocab = build_vocab()
    print(f"vocab: {len(vocab.drugs)} drugs, {len(vocab.groups)} groups, "
          f"fingerprint {vocab.fingerprint()}")

    specs = make_v14_specs(args.n, vocab=vocab, seed=args.seed,
                           composition=args.composition)
    print(f"{len(specs)} specs, prompt {v14_fingerprint()}, "
          f"composition {args.composition}, effort {args.effort}")

    sample_fn = make_v14_sample_fn(
        client, model=args.model, temperature=args.temperature,
        reasoning_effort=args.effort, max_output_tokens=args.max_output_tokens,
        api=args.api)

    t0 = time.time()
    generate_raw(specs, sample_fn, gen_id=args.gen_id, max_workers=args.workers)
    dt = time.time() - t0
    print(f"\nstage 1 finished in {dt / 3600:.2f}h ({dt / max(len(specs), 1):.2f}s/spec)")

    if args.no_build:
        return

    dataset_id, stats = build_dataset_from_raw(
        args.gen_id, resolver=v14_sample_to_instances, mode="markers",
        generator={"gen_id": args.gen_id,
                   "prompt_sha": v14_fingerprint(),
                   "model": args.model,
                   "reasoning_effort": args.effort,
                   "temperature": args.temperature,
                   "composition": args.composition,
                   "n_positives_by_k": "hand-tuned to decorrelate positive rate from "
                                       "entity count; not a corpus measurement"},
        vocab_source=vocab.fingerprint(), seed=args.seed,
        notes=f"v14 full run, {args.n} specs")
    print(f"\ndataset {dataset_id}")
    print(stats["reject_reasons"])

    from ddi.manifest import load_dataset
    instances, _ = load_dataset(dataset_id)
    gates.report(instances, records=generation_records(args.gen_id), strict=False)


if __name__ == "__main__":
    main()