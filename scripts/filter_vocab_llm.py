"""LLM-filter the vocabulary down to genuine pharmacological substances.

terms.json is a chemistry synonym dump: after regex cleaning it is still mostly
industrial chemicals, dyes, solvents and reagents, which no pattern can separate
from real drug names ("Dinonyl adipate" looks exactly like "Clofilium").

So we ask the model. We do NOT filter all 262k terms -- generation only needs a
few thousand distinct entity names, so we sample, filter the sample, and keep the
survivors. ~100 calls.

    python scripts/filter_vocab_llm.py --sample 10000 --batch 100

Writes datasets/other/vocab.json with the filtered lexicon, and keeps the
unfiltered pool alongside it so "clean lexicon vs raw chemical pool" stays
available as a one-line ablation.
"""
import argparse, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddi.vocab import clean_chemicals, DRUG_GROUPS
from ddi.synth import generate_raw

OUT = Path("datasets/other/vocab.json")

SYSTEM = (
    "You are a pharmacology expert curating a drug lexicon. "
    "The user gives you a numbered list of candidate terms from a chemical database. "
    "Return the indices of terms that are PHARMACOLOGICAL SUBSTANCES: approved drugs "
    "(generic or chemical names), brand/trade names of medicines, drug classes or groups, "
    "and active substances used in humans or in pharmacological research.\n"
    "EXCLUDE: industrial chemicals, solvents, plasticisers, dyes and pigments, laboratory "
    "reagents, cosmetics and consumer products, foods, database identifiers or codes, "
    "molecular formulae, and malformed or corrupted strings.\n"
    "Return only indices, no explanation. Be strict: when unsure, exclude."
)


def make_sample_fn(client, model, temperature):
    from pydantic import BaseModel

    class Kept(BaseModel):
        indices: list[int]

    def sample_fn(spec):
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(spec["terms"]))
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": listing}],
            text_format=Kept,
            temperature=temperature,
        )
        return {"indices": resp.output_parsed.indices}

    return sample_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default="datasets/other/terms.json")
    ap.add_argument("--sample", type=int, default=10000, help="how many terms to consider")
    ap.add_argument("--batch", type=int, default=100, help="terms per LLM call")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.0)  # deterministic-ish: this is a judgement, not creative
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--gen-id", default="vocabfilter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. regex clean (cheap, removes ~30% before we spend any calls)
    raw = json.loads(Path(args.terms).read_text())
    chemicals = raw["chemical"] if isinstance(raw, dict) else raw
    if isinstance(chemicals, dict):
        chemicals = list(chemicals.keys())
    simple, systematic = clean_chemicals(chemicals)

    # 2. sample -- we only need a few thousand survivors
    rng = random.Random(args.seed)
    pool = rng.sample(simple, min(args.sample, len(simple)))
    batches = [{"terms": pool[i:i + args.batch]} for i in range(0, len(pool), args.batch)]
    print(f"\nfiltering {len(pool)} terms in {len(batches)} batches")

    # 3. LLM filter (reuses the generation harness: concurrent, checkpointed, resumable)
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("IDA_LLM_BASE_URL",
                                            "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
                    api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=120.0)
    sample_fn = make_sample_fn(client, args.model, args.temperature)
    raw_path = generate_raw(batches, sample_fn, gen_id=args.gen_id, max_workers=args.workers)

    # 4. collect survivors
    kept = []
    for line in Path(raw_path).read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        terms = rec["spec"]["terms"]
        for idx in rec["sample"]["indices"]:
            if 0 <= idx < len(terms):          # model can hallucinate an index
                kept.append(terms[idx])
    kept = sorted(set(kept))

    print(f"\nLLM kept {len(kept)} / {len(pool)} ({100 * len(kept) / max(len(pool), 1):.1f}%)")
    print("sample of kept:", kept[:25])

    if len(kept) < 0.01 * len(pool):
        n_err = sum(1 for line in Path(raw_path).read_text().splitlines()
                    if line and json.loads(line).get("error"))
        raise SystemExit(
            f"\nABORT: kept only {len(kept)} terms -- refusing to overwrite {OUT}.\n"
            f"  {n_err} of the calls errored. Check the model is actually running:\n"
            f"    head -c 500 {raw_path}\n"
            f"  Then delete {raw_path} and re-run, or fall back to the regex-only\n"
            f"  vocabulary with:  python scripts/build_vocab.py"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "simple": kept,
        "systematic": systematic,
        "groups": DRUG_GROUPS,
        "context": _context(raw),
        "unfiltered_simple": pool,     # keep the raw pool for the ablation
        "filter": {"method": "llm", "model": args.model, "n_considered": len(pool),
                   "n_kept": len(kept), "seed": args.seed},
    }, indent=0))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


def _context(raw):
    if not isinstance(raw, dict):
        return []
    out = []
    for k in ("gene", "disease"):
        vals = raw.get(k, [])
        out.extend(list(vals.keys()) if isinstance(vals, dict) else vals)
    return out


if __name__ == "__main__":
    main()