"""Paraphrase sentences with entities pinned, then prune on label agreement.

WHY
---
Two uses, and they are different questions that happen to share machinery.

As augmentation: the labelling arm is bounded by how much real text exists (2,402
eligible sentences, which is why --n 6000 returned 2,417). Paraphrase is the only way to
grow it. The generation arm is not bounded, so for that arm paraphrase has to beat simply
generating more, which is the size-matched control below.

As a diagnostic: if a label does not survive rewording, the label was carried by surface
form rather than by content. That is the shortcut this project has spent eight weeks on,
measured per pair instead of per corpus. Synthetic data should be more paraphrase-fragile
than real data, and the gap is a cleaner statement of the problem than probe lift.

ALIGNMENT
---------
Entities are pinned by inline index, not by string. 26% of real corpus sentences repeat a
drug name and v18 fires repeated_mention at 0.25, so a rewrite that reorders mentions will
silently misalign pairs if they are matched by surface. The model is required to carry
`name[n]` tokens through verbatim; spans come back by regex and labels map by index.

CALIBRATION
-----------
Run the human-labelled pool through the same pipeline. If gold labels disagree across
paraphrases at a similar rate, this is measuring paraphrase noise and not label fragility,
and none of the pruning numbers mean anything. That arm is not optional.
"""
import json
import re
from collections import Counter, defaultdict

from .verify_binary import _number_inline, _sentence_spans, _strip, group_by_sentence

NUMBERED = re.compile(r"([A-Za-z0-9][^\[\]]*?)\[(\d+)\]")

SYSTEM = """You rewrite biomedical sentences without changing what they claim.

Every drug mention is tagged inline with a number, like aspirin[3]. These tags are
immutable. Carry every one through into your rewrite exactly as it appears, same surface
form, same number, same bracket. Do not add tags, drop tags, renumber them, or change the
spelling of a tagged name.

Everything else should change. Rework the clause order, the voice, the connectives and the
wording. Aim for a sentence a different author would have written to report the same facts.

What must not change: which drugs are said to interact with which, in what direction, and
of what kind. If the original says one drug raises another's plasma levels, the rewrite
says that too. If the original denies an interaction, the rewrite denies it. If the
original merely lists drugs without claiming anything, the rewrite lists them without
claiming anything.

Return only the rewritten sentence."""


def build_paraphrase_specs(instances, k=3, skip_pairs_above=190):
    """One spec per (sentence, paraphrase index). Numbering is done once and reused so
    every paraphrase of a sentence is pinned to the same index set."""
    out = []
    for sent_id, rows in group_by_sentence(instances).items():
        spans = _sentence_spans(rows)
        if len(rows) >= skip_pairs_above or len(spans) < 2:
            continue
        numbered = _number_inline(_strip(rows[0]["text"]), spans)
        for j in range(k):
            out.append({"sent_id": sent_id, "para": j, "numbered": numbered,
                        "n_mentions": len(spans),
                        "expect": sorted(int(n) for _, n in NUMBERED.findall(numbered))})
    return out


def make_paraphraser(client, model="gpt-oss-120b", temperature=1.0,
                     reasoning_effort="low", max_output_tokens=2000, api="responses"):
    """temperature=1.0 deliberately. The point is variety between the k rewrites of one
    sentence; at temperature 0 they collapse to near-duplicates and the agreement signal
    measures nothing."""
    from pydantic import BaseModel

    class Rewritten(BaseModel):
        sentence: str

    def _responses(user):
        kw = {"max_output_tokens": max_output_tokens}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        r = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Rewritten, temperature=temperature, **kw)
        if r.output_parsed is None:
            raise ValueError("no parsed output")
        return r.output_parsed.sentence

    def _chat(user):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "rewritten", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["sentence"],
                           "properties": {"sentence": {"type": "string"}}}}})
        return Rewritten.model_validate_json(r.choices[0].message.content).sentence

    def paraphrase(spec):
        got = (_responses if api == "responses" else _chat)(spec["numbered"])
        found = sorted(int(n) for _, n in NUMBERED.findall(got))
        if found != spec["expect"]:
            raise ValueError(f"tag set changed: expected {spec['expect']}, got {found}")
        return {"sent_id": spec["sent_id"], "para": spec["para"], "numbered": got}

    return paraphrase


def paraphrase_report(raw_path):
    """How many rewrites survived with their tags intact, and how different they are.
    A distinct-4 near the original's means the model reworded nothing and the whole
    exercise is inert; check this before spending a verifier pass on the output."""
    ok, bad = [], Counter()
    for line in open(raw_path).read().splitlines():
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("sample"):
            bad[str(r.get("error"))[:60]] += 1
        else:
            ok.append(r["sample"])
    print(f"{len(ok)} rewrites kept, {sum(bad.values())} failed")
    for k, n in bad.most_common(5):
        print(f"  {n:5d}  {k}")
    per = Counter(o["sent_id"] for o in ok)
    print(f"{len(per)} sentences with at least one rewrite; "
          f"{sum(1 for v in per.values() if v >= 3)} with all three")
    return ok


def agreement_prune(originals, verdict_path, mode="unanimous"):
    """Keep pairs whose label survived rewording.

    `originals` maps (sent_id, pair_index) -> label as it stood before paraphrasing:
    gold-by-construction for synthetic, the annotator's label for the labelling arm,
    gold for the human calibration arm.

    mode="unanimous" keeps a pair only if every paraphrase agrees with the original.
    mode="majority" keeps it if more than half do. Unanimous is the stricter diagnostic
    and the one to report; majority loses less data and is the one more likely to help
    downstream. Run both, they cost nothing once the verdicts exist.
    """
    votes = defaultdict(list)
    for line in open(verdict_path).read().splitlines():
        if not line:
            continue
        r = json.loads(line)
        payload = r.get("sample") if "sample" in r else r
        if r.get("error") or not payload:
            continue
        for i, lab in enumerate(payload["labels"]):
            votes[(payload["sent_id"], i)].append(lab)

    keep, drop, no_votes = set(), set(), 0
    for key, orig in originals.items():
        vs = votes.get(key)
        if not vs:
            no_votes += 1
            continue
        agree = sum(1 for v in vs if v == orig)
        ok = agree == len(vs) if mode == "unanimous" else agree * 2 > len(vs)
        (keep if ok else drop).add(key)

    print(f"{mode}: keep {len(keep)}, drop {len(drop)}, "
          f"{len(drop) / max(len(keep) + len(drop), 1):.3f} prune rate, "
          f"{no_votes} pairs with no paraphrase verdict")

    by_label = defaultdict(lambda: [0, 0])
    for key, orig in originals.items():
        if key in keep:
            by_label[orig][0] += 1
        elif key in drop:
            by_label[orig][1] += 1
    print(f"\n{'label':>10} {'kept':>7} {'dropped':>8} {'prune rate':>11}")
    for lab, (k, d) in sorted(by_label.items(), key=lambda kv: -sum(kv[1])):
        print(f"{lab:>10} {k:>7} {d:>8} {d / max(k + d, 1):>11.3f}")
    return keep


def apply_prune(instances, keep):
    """Instances whose (sent_id, pair_index) survived. Pair index is position within the
    sentence, matching build_batches, so this has to be derived the same way there and
    here or the wrong pairs get dropped."""
    out = []
    for sent_id, rows in group_by_sentence(instances).items():
        for i, r in enumerate(rows):
            if (sent_id, i) in keep:
                out.append(r)
    return out
