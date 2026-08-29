"""Two arms at high reasoning effort, both resumable, both meant for tmux.

    tmux new -s runs
    python scripts/run_labelled.py generate --gen-id v18-high --n 6000
    python scripts/run_labelled.py label    --gen-id label-high --n 6000
    # ctrl-b d to detach. tmux a -t runs to come back.

Both stages append to raw/<gen-id>.jsonl as results arrive and skip whatever is already
there on restart, so a dropped session costs only the in-flight requests. That is a
better guarantee than keeping a browser tab alive, because it also survives the pod
dying, which a keepalive does not.

Then, in the notebook or here:

    python scripts/run_labelled.py build --gen-id v18-high
    python scripts/run_labelled.py agree  --gen-id label-high
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from ddi import label_real, prompt_v18
from ddi.data import build_human
from ddi.manifest import load_dataset
from ddi.resolve import v14_sample_to_instances
from ddi.synth import RAW, build_dataset_from_raw, generate_raw
from ddi.vocab import build_vocab

MODEL = "gpt-oss-120b"


def client():
    return OpenAI(base_url=os.environ.get(
        "LLM_BASE_URL", "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
        api_key=os.environ.get("IDA_LLM_API_KEY", os.environ.get("LLM_API_KEY", "x")),
        max_retries=5, timeout=180.0)


def generate(args):
    """v18 at high reasoning effort. v18's 0.486 was generated at low, so effort is an
    untested variable on the current best generator, and it is the cheapest remaining
    ablation: same prompt, same seed, one setting changed.

    Timeout is 180s not 60s. At high effort a multi-assertion spec takes far longer, and
    a 60s timeout turns slow completions into errors that look like API failures.
    """
    vocab = build_vocab()
    specs = prompt_v18.make_specs(args.n, vocab=vocab, seed=args.seed)
    print(f"prompt sha {prompt_v18.fingerprint()}  {len(specs)} specs  "
          f"effort={args.effort}  max_output_tokens={args.max_tokens}")

    sample_fn = prompt_v18.make_sample_fn(
        client(), model=MODEL, reasoning_effort=args.effort,
        max_output_tokens=args.max_tokens, api=args.api)

    t0 = time.time()
    generate_raw(specs, sample_fn, gen_id=args.gen_id, max_workers=args.workers)
    print(f"generation done in {(time.time() - t0) / 60:.1f} min")


def build(args):
    """Stage two is deterministic and free, so it can be re-run whenever the resolver
    changes without touching the API."""
    did, stats = build_dataset_from_raw(
        args.gen_id, resolver=v14_sample_to_instances, mode="markers",
        generator={"prompt_sha": prompt_v18.fingerprint(), "model": MODEL,
                   "version": "v18", "reasoning_effort": args.effort},
        vocab_source=build_vocab().fingerprint(), seed=args.seed,
        notes=f"{args.gen_id}, effort={args.effort}")
    print(did, stats["reject_reasons"])
    inst, _ = load_dataset(did)
    print(f"{len(inst)} instances, "
          f"{len({r['sent_id'] for r in inst})} sentences, "
          f"pos rate {sum(1 for r in inst if r['label'] != 'NONE') / len(inst):.3f}")


def label(args):
    """The labelling arm: real held-out Train sentences, gold spans kept, gold labels
    discarded and replaced by the model's. Matched to the generation arm on sentences."""
    train, dev, val = build_human(seed=args.split_seed)
    specs = label_real.sample_sentences(train, n=args.n, seed=args.seed)
    n_pairs = sum(len(s["pairs"]) for s in specs)
    print(f"{len(specs)} sentences, {n_pairs} pairs, "
          f"effort={args.effort}  max_output_tokens={args.max_tokens}")

    fn = label_real.make_labeller(
        client(), model=MODEL, reasoning_effort=args.effort,
        max_output_tokens=args.max_tokens, api=args.api)

    t0 = time.time()
    generate_raw(specs, fn, gen_id=args.gen_id, max_workers=args.workers)
    print(f"labelling done in {(time.time() - t0) / 60:.1f} min")


def agree(args):
    label_real.agreement(RAW / f"{args.gen_id}.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["generate", "build", "label", "agree"])
    p.add_argument("--gen-id", required=True)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--api", default="responses")
    args = p.parse_args()
    {"generate": generate, "build": build, "label": label, "agree": agree}[args.cmd](args)


if __name__ == "__main__":
    main()