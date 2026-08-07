"""LLM-filter the DrugBank + WHO-ATC vocabulary down to entities DDI-2013 would annotate.

Replaces the terms.json version, which sampled because that dump held 262k terms. The
current lexicon is ~14.9k drugs and ~425 groups, so at 100 per batch the whole thing is
~154 calls. Sampling is no longer needed and would only leave unfiltered names in the
pool.

Why this matters more than any prompt fix: rendered v14 specs draw names like Fusarium
oxysporum, Cosmetics, Black walnut, Cellotetraose, Coproporphyrin I and HUCMSCs. Those
are not drugs in the DDI-2013 sense, and asking the model to write pharmacology about
substances that have none is a distribution mismatch no prompt can repair.

Drugs and groups get separate criteria. The old script never filtered groups at all,
which is why Fibrinogen and Cosmetics appear as drug classes.

    python scripts/filter_vocab_llm.py
    python scripts/filter_vocab_llm.py --limit 500 --gen-id vocabfilter-smoke

Writes datasets/other/vocab_filtered.json with both the kept and rejected lists. The
rejected list is the diagnostic: read it before trusting the filter, since an
over-strict pass silently removes real drugs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddi.vocab import build_vocab, OTHER
from ddi.synth import generate_raw

OUT = OTHER / "vocab_filtered.json"

DRUG_SYSTEM = """You curate a lexicon of drug entities for a biomedical relation
extraction corpus. The user gives a numbered list of candidate names. Return the indices
of names that could plausibly be annotated as a drug entity in a drug interaction study.

KEEP:
- approved medicines, by generic or chemical name
- brand and trade names of medicines
- investigational and research compounds intended for human pharmacological use
- biologics: monoclonal antibodies, vaccines, enzymes, cell and gene therapies
- substances routinely discussed in interaction literature even when not prescription
  medicines, such as alcohol, grapefruit juice, tobacco, and vitamins

EXCLUDE:
- foods, spices, culinary plants, and plant parts or extracts with no established
  medicinal use
- minerals, elements, and inorganic compounds unless given therapeutically
- cosmetics, toiletries, and consumer products
- laboratory reagents, stains, buffers, solvents, and industrial chemicals
- endogenous biochemical intermediates, metabolites, and sugars that are not themselves
  administered as medicines
- micro-organisms, fungi, and allergenic extracts
- database identifiers, codes, molecular formulae, and malformed strings

Return only indices. When genuinely unsure whether something is administered to humans
for a therapeutic purpose, exclude it."""

GROUP_SYSTEM = """You curate a lexicon of drug CLASS entities for a biomedical relation
extraction corpus. The user gives a numbered list of candidate class names. Return the
indices of names that are pharmacological or therapeutic classes of drugs, of the kind
that would be annotated as a group entity in a drug interaction study.

KEEP:
- pharmacological classes, such as beta blocking agents, quinolones, azole antifungals
- therapeutic classes, such as antidepressants, anticoagulants, antiretrovirals
- mechanism-based classes, such as CYP3A4 inhibitors, proton pump inhibitors

EXCLUDE:
- anatomical or organ-system categories that are not classes of drug
- individual substances rather than classes
- administrative or packaging categories, such as combinations, other products,
  medicated dressings, diagnostic agents
- non-pharmacological groupings, such as cosmetics, foods, devices, reagents
- malformed strings and codes

Return only indices. When unsure whether a name denotes a class of drugs rather than a
single substance or a non-drug category, exclude it."""


def make_filter_fn(client, model, temperature, reasoning_effort, max_output_tokens=None,
                   api="responses"):
    from pydantic import BaseModel

    class Kept(BaseModel):
        indices: list[int]

    def _responses(spec, listing):
        kw = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
        if max_output_tokens is not None:
            kw["max_output_tokens"] = max_output_tokens
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": spec["system"]},
                   {"role": "user", "content": listing}],
            text_format=Kept, temperature=temperature, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return resp.output_parsed.indices


    def sample_fn(spec):
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(spec["terms"]))
        return {"indices": _responses(spec, listing)}

    return sample_fn


def _batches(terms, system, kind, size):
    return [{"terms": terms[i:i + size], "system": system, "kind": kind}
            for i in range(0, len(terms), size)]


def _collect(raw_path):
    """Returns {kind: (kept, rejected)}. A batch that errored contributes nothing to
    either list, so its terms are dropped rather than silently kept or lost."""
    kept, rejected, n_err = {}, {}, 0
    for line in Path(raw_path).read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        kind = rec["spec"]["kind"]
        kept.setdefault(kind, [])
        rejected.setdefault(kind, [])
        if rec.get("error"):
            n_err += 1
            continue
        terms = rec["spec"]["terms"]
        idx = {i for i in rec["sample"]["indices"] if 0 <= i < len(terms)}
        for i, t in enumerate(terms):
            (kept if i in idx else rejected)[kind].append(t)
    return kept, rejected, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap terms per kind, for a smoke run")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default="high",
                    help="this is a judgement task, not a writing one")
    ap.add_argument("--max-output-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gen-id", default="vocabfilter-v2")
    args = ap.parse_args()

    vocab = build_vocab()
    drugs = sorted(set(vocab.drugs))
    groups = sorted(set(vocab.groups))
    if args.limit:
        drugs, groups = drugs[:args.limit], groups[:args.limit]
    print(f"filtering {len(drugs)} drugs and {len(groups)} groups")

    batches = (_batches(drugs, DRUG_SYSTEM, "drug", args.batch)
               + _batches(groups, GROUP_SYSTEM, "group", args.batch))
    print(f"{len(batches)} batches")

    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("IDA_LLM_BASE_URL",
                                            "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
                    api_key=os.environ["IDA_LLM_API_KEY"], max_retries=5, timeout=180.0)
    fn = make_filter_fn(client, args.model, args.temperature,
                        args.reasoning_effort, args.max_output_tokens)
    raw_path = generate_raw(batches, fn, gen_id=args.gen_id, max_workers=args.workers)

    kept, rejected, n_err = _collect(raw_path)
    kd, kg = kept.get("drug", []), kept.get("group", [])
    rd, rg = rejected.get("drug", []), rejected.get("group", [])

    print(f"\ndrugs  kept {len(kd)}/{len(kd) + len(rd)}"
          f"  ({100 * len(kd) / max(len(kd) + len(rd), 1):.1f}%)")
    print(f"groups kept {len(kg)}/{len(kg) + len(rg)}"
          f"  ({100 * len(kg) / max(len(kg) + len(rg), 1):.1f}%)")
    if n_err:
        print(f"{n_err} batches errored and were dropped entirely")

    print("\nkept drugs (sample):", kd[::max(len(kd) // 20, 1)][:20])
    print("cut  drugs (sample):", rd[::max(len(rd) // 20, 1)][:20])
    print("\nkept groups (sample):", kg[::max(len(kg) // 15, 1)][:15])
    print("cut  groups (sample):", rg[::max(len(rg) // 15, 1)][:15])

    if len(kd) < 0.2 * (len(kd) + len(rd)):
        raise SystemExit(
            f"\nABORT: kept only {len(kd)} drugs. Refusing to overwrite {OUT}.\n"
            f"  Inspect {raw_path}, then delete it and re-run.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "drugs": sorted(kd),
        "groups": sorted(kg),
        "rejected_drugs": sorted(rd),
        "rejected_groups": sorted(rg),
        "filter": {"method": "llm", "model": args.model,
                   "reasoning_effort": args.reasoning_effort,
                   "temperature": args.temperature,
                   "n_drugs_in": len(kd) + len(rd), "n_drugs_kept": len(kd),
                   "n_groups_in": len(kg) + len(rg), "n_groups_kept": len(kg),
                   "n_batch_errors": n_err, "gen_id": args.gen_id},
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()