"""v18 spec construction: assertions and modifiers on separate axes.

WHY
---
v17's normalised MI between frame and sentence label is exactly 1.000, and a bag of
words recovers the frame from masked text at 0.961 accuracy over 19 classes. The label
is therefore a deterministic function of something visible on the surface, which is the
v13 composition shortcut with construction in place of pair count.

The 295 classified corpus sentences say the inventory is on the wrong axis. `appositive`
and `coordinate` are the main clause of zero sentences out of 295. `regimen` is the main
clause of 9 but appears in roughly 52. `title` is 10 primary against roughly 36 total.
These are packaging that wraps an assertion, not alternatives to one. v17 samples them
as mutually exclusive primaries, which is why `appositive` runs at synth 0.580 positive
against a corpus figure near 0.112, the largest single purity gap in the table.

So: one axis of assertions, which bind pairs and carry labels, and one axis of modifiers,
which carry no label and are sampled independently. A sentence draws one or two
assertions and zero or more modifiers. The sentence label stops being a function of
anything nameable, and the combinations are generated rather than enumerated.

WHAT ELSE CHANGED
-----------------
Four assertion types added from the `other` bucket and from constructions the corpus uses
that v17 misfiles. Each is described at its builder.

`out_of_scope` is deleted. Its nine corpus instances were three lab-test interferences,
two comparative statements, two non-drug targets, a heading and a bare measurement. Each
piece has an independent trigger, so it was a dustbin rather than a construction.

`mechanism_mixed` is deleted as a frame and becomes a mechanism assertion co-sampled with
an effect assertion, which is what it always was.

`enzyme_only` is widened to `drug_property` rather than split. The corpus puts enzyme
roles, class membership, approval history, own-kinetics statements and product
composition in one shape: a claim about what one drug is, with other drugs merely named.
Splitting it would produce four frames with the same lexical signature, which is the
problem this module exists to fix.

TARGETS
-------
The number to move is frame_diag.label_purity's nmi_sentence, from 1.000 towards the
corpus figure. Secondary: gold_frames.corpus_purity's sent_pos_rate per construction,
which no v18 construction should hit 0.000 or 1.000 on.
"""
import itertools
import random
import re

from .prompt import (ADVICE, AXIS_POOL, AXES_FOR_KIND, ENZYME, ENZYME_ROLE, HARM,
                     HARM_EXTENT, REGISTER_BAN, REGISTER_LINE, REGISTERS, SCENES,
                     P_SCENE, _alias, _pick, _sample_axes, _unusable)
from . import prompt as _v17


def _referenced(rows):
    out = set()
    for _, v in rows:
        out |= set(re.findall(r"\{(\w)\}", v))
    return out


def _keys(ents):
    return [e["key"] for e in ents]


def _f_no_evidence(ents, rng):
    """Not studied, not established, not evaluated.

    Seven of the 295 corpus sentences do this and they were split across `denial`,
    `study_only` and `advise_reason`, which is why none of those three is clean. It is
    not `denial`: denial asserts that no change occurs, this asserts that nobody looked.
    The distinction matters because the corpus attaches advice to it ("although in vivo
    studies have not been done, it is not recommended that they be coadministered"),
    so it co-occurs with a positive assertion and must not be label-pure.
    """
    a, b = rng.sample(_keys(ents), 2)
    return {"kind": "none_single",
            "says": [("subject", f"{{{a}}} with {{{b}}}"),
                     ("what is missing", rng.choice(
                         ["no study has been done",
                          "safety and effectiveness not established",
                          "the combination has not been evaluated",
                          "appropriate doses are not established",
                          "no data exist for this combination"]))],
            "avoid": ["asserting that no interaction occurs; the point is that nobody "
                      "looked, not that the answer is no"],
            "positives": [], "focus": [a, b]}


def _f_single_drug_effect(ents, rng):
    """One drug produces an outcome; the others are merely present.

    The most valuable new construction, because it is the hard negative v17 cannot make.
    Every v17 negative frame avoids effect vocabulary, so effect vocabulary became a POS
    cue and denial vocabulary a NONE cue: that is what the probe reads. A sentence full
    of caused, induced, prolonged, damage, with no second drug participating, breaks
    that association directly.
    """
    a = rng.choice(_keys(ents))
    return {"kind": "none_single",
            "says": [("subject", f"{{{a}}}"),
                     ("what it produced", _alias(HARM, rng)[1]),
                     ("detail", rng.choice(HARM_EXTENT))],
            "avoid": ["naming any other listed drug as causing, preventing or changing "
                      "this"],
            "positives": [], "focus": [a]}


def _f_comparative(ents, rng):
    """X is better than, differs from, or was measured alongside Y.

    Four corpus instances as `other` plus several misfiled under `denial` and
    `out_of_scope`. Full of contrastive and outcome language while asserting no
    interaction, so it works on the same association as single_drug_effect.
    """
    a, b = rng.sample(_keys(ents), 2)
    return {"kind": "none_pair",
            "shape": f"a comparison of {{{a}}} against {{{b}}}",
            "says": [("what is compared", rng.choice(
                          ["efficacy", "tolerability", "measured activity",
                           "reported response rate", "binding"])),
                     ("finding", rng.choice(
                          ["one appears superior", "the values differ",
                           "one was more effective in a subset",
                           "corresponding values were in a different range"]))],
            "avoid": ["stating that one drug affects the other; they are being measured "
                      "side by side, not combined"],
            "positives": [], "focus": [a, b]}


def _f_lab_interference(ents, rng):
    """A drug interferes with a laboratory test, not with another drug.

    Five corpus instances, currently split between `effect` and `out_of_scope`, all gold
    NONE, and the guidelines give it its own section. It reads exactly like an effect
    assertion apart from the target being an assay.
    """
    a = rng.choice(_keys(ents))
    return {"kind": "none_single",
            "says": [("subject", f"{{{a}}}"),
                     ("what it interferes with", rng.choice(
                         ["a urinary assay", "a coagulation assay",
                          "a bioassay for antibacterial levels",
                          "a colorimetric determination",
                          "a false-positive reaction in a urine test"]))],
            "avoid": ["naming any other listed drug as affected"],
            "positives": [], "focus": [a]}


def _f_drug_property(ents, rng):
    """What one drug is: enzyme role, class membership, own kinetics, composition.

    Widened from v17's `enzyme_only`. The corpus writes all four in the same shape and
    they carry the same signature, so four frames would be four names for one thing.
    """
    a = rng.choice(_keys(ents))
    mode = rng.choice(["enzyme", "class", "kinetics", "composition"])
    if mode == "enzyme":
        says = [("subject", f"{{{a}}}"), ("what it is", rng.choice(ENZYME_ROLE)),
                ("of what", rng.choice(ENZYME))]
    elif mode == "class":
        says = [("subject", f"{{{a}}}"),
                ("what it is", rng.choice(["a member of its therapeutic class",
                                           "the first of its class approved for use",
                                           "a substrate of a transporter"]))]
    elif mode == "kinetics":
        says = [("subject", f"{{{a}}}"),
                ("what is described", rng.choice(
                    ["nonlinear elimination kinetics",
                     "dependence on hepatic blood flow for elimination",
                     "its own clearance and half-life"]))]
    else:
        says = [("subject", f"{{{a}}}"),
                ("what it contains", rng.choice(
                    ["an active ingredient of a named class",
                     "a fixed-dose combination of its components"]))]
    return {"kind": "none_single", "says": says,
            "avoid": ["naming any other listed drug as affected by this"],
            "positives": [], "focus": [a]}


ASSERTIONS = {
    "effect":            {"n": 2, "w": 0.170, "f": _v17._f_effect},
    "effect_pd":         {"n": 2, "w": 0.140, "f": _v17._f_effect_pd},
    "mechanism":         {"n": 2, "w": 0.140, "f": _v17._f_mechanism},
    "denial":            {"n": 2, "w": 0.130, "f": _v17._f_denial},
    "study_only":        {"n": 2, "w": 0.080, "f": _v17._f_study_only},
    "advise_reason":     {"n": 2, "w": 0.080, "f": _v17._f_advise_reason},
    "advise":            {"n": 2, "w": 0.060, "f": _v17._f_advise},
    "drug_property":     {"n": 1, "w": 0.050, "f": _f_drug_property},
    "effect_protect":    {"n": 2, "w": 0.025, "f": _v17._f_effect_protective},
    "no_evidence":       {"n": 2, "w": 0.025, "f": _f_no_evidence},
    "single_drug_effect":{"n": 1, "w": 0.020, "f": _f_single_drug_effect},
    "comparative":       {"n": 2, "w": 0.020, "f": _f_comparative},
    "lab_interference":  {"n": 1, "w": 0.017, "f": _f_lab_interference},
    "int":               {"n": 2, "w": 0.015, "f": _v17._f_int},
    "effect_failure":    {"n": 2, "w": 0.010, "f": _v17._f_effect_failure},
    "incompatible":      {"n": 2, "w": 0.005, "f": _v17._f_incompatible},
    "contradictory":     {"n": 2, "w": 0.005, "f": _v17._f_contradictory},
}

P_SECOND_ASSERTION = 0.40

NEGATIVE_ASSERTIONS = {"denial", "study_only", "drug_property", "no_evidence",
                       "single_drug_effect", "comparative", "lab_interference",
                       "incompatible"}

P_SECOND_IS_NEGATIVE = 0.75

P_UNLISTED_TARGET = 0.26

P_LIST_TARGET = 0.55

UNLISTED_TARGETS = ["another drug that is not named in this sentence",
                    "drugs of a class that is not named in this sentence",
                    "a laboratory measurement",
                    "food or drink",
                    "an endogenous substance"]


def _to_list_target(built, ents, rng):
    """Expand the thing being acted on into a list, and give every member the label.

    The corpus's positive sentences carry far more than one positive pair, because the
    object of the assertion is routinely a list: "drugs that diminish anticoagulant
    response include A; B; C". v17 could only do this through `appositive`, which is why
    its positive rate had to be tuned elsewhere to compensate.
    """
    if not built["positives"] or len(built["focus"]) < 2:
        return built
    a, b, lab = built["positives"][0]
    spare = [e["key"] for e in ents if e["key"] not in built["focus"]]
    if not spare:
        return built
    rng.shuffle(spare)
    members = spare[:rng.choice([1, 2, 2, 3])]
    pos = list(built["positives"])
    for m in members:
        pos.append((a, m, lab))
    says = list(built["says"]) + [
        ("and also acts on", ", ".join(f"{{{m}}}" for m in members))]
    return {**built, "says": says, "positives": pos, "variant": "list_target",
            "focus": sorted(set(built["focus"]) | set(members))}


def _to_unlisted(built, rng):
    """Assert the relation against a referent that is not one of the listed entities.

    This is how a positive assertion type ends up in a sentence with no annotatable pair,
    which is the only way to stop positive assertions being label-pure. It is not a
    contrivance: the corpus does it constantly. "The concomitant use of alcohol or other
    central nervous system depressants may have an additive effect" is gold NONE because
    the second referent is not an annotated entity, and the OBJECTIVE line about
    clarithromycin and simvastatin is gold NONE for a related reason. Effect vocabulary
    with no annotatable pair is the hard negative v17 cannot produce at all.
    """
    if not built["positives"] or len(built["focus"]) < 2:
        return built
    keep = built["focus"][0]
    dropped = [k for k in built["focus"] if k != keep]
    target = rng.choice(UNLISTED_TARGETS)

    def swap(t):
        for k in dropped:
            t = t.replace("{" + k + "}", target)
        return t

    says = [(k, swap(v)) for k, v in built["says"]]
    shape = swap(built["shape"]) if built.get("shape") else None
    return {**built, "says": says, "shape": shape, "positives": [], "focus": [keep],
            "variant": "unlisted",
            "avoid": list(built.get("avoid", [])) + [
                "naming any of the other listed drugs as the thing affected here"]}


MODIFIERS = {
    "regimen_background": 0.18,
    "title":              0.12,
    "appositive":         0.07,
    "sequential":         0.065,
    "coordinate":         0.04,
    "repeated_mention":   0.25,
    "dose_detail":        {"DrugBank": 0.15, "MedLine": 0.35},
    "fragment":           0.03,
}

ROLE_CODES = ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]

ROLE_HINTS = {
    "r1": "started before the others and since stopped",
    "r2": "started only after the others finished",
    "r3": "a comparator",
    "r4": "a fallback if the first choice does not suit",
    "r5": "withheld",
    "r6": "for an unrelated condition",
    "r7": "background therapy",
}

P_ROLE = 0.55


def _apply_modifiers(built, ents, mods, rng):
    """Modifiers rewrite the arrangement and add non-labelled material. None of them
    touch `positives`, which is the whole point: nothing a modifier contributes can move
    the label."""
    keys = _keys(ents)
    focus = set(built["focus"])
    spare = [k for k in keys if k not in focus]
    extra = []

    if "title" in mods:
        head = rng.choice(keys)
        built["shape"] = (f"a heading naming {{{head}}}, a colon, then "
                          + (built.get("shape") or "the statement"))
    if "appositive" in mods and spare:
        host = rng.choice(sorted(focus)) if focus else rng.choice(keys)
        member = spare.pop(rng.randrange(len(spare)))
        marker = rng.choice(["including", "such as", "e.g."])
        extra.append(("expand", f"{{{host}}}, {marker} {{{member}}}"))
        built.setdefault("appositive_pairs", []).append((host, member))
    if "coordinate" in mods and spare:
        partner = spare.pop(rng.randrange(len(spare)))
        if focus:
            host = sorted(focus)[0]
            extra.append(("coordinate", f"{{{host}}} and {{{partner}}} together"))
            built.setdefault("coordinate_pairs", []).append((host, partner))
    if "sequential" in mods:
        extra.append(("timing", rng.choice(
            ["with a gap between the two", "one stopped before the other started",
             "one taken some hours after the other"])))
    if "regimen_background" in mods and spare:
        extra.append(("also present", ", ".join(f"{{{k}}}" for k in spare[:2])))
    if "dose_detail" in mods:
        extra.append(("include", "doses or concentrations in parentheses"))
    if "fragment" in mods:
        built["shape"] = "a table row or list item, not a full sentence"
        built["avoid"] = [x for x in built.get("avoid", []) if "listed drug" in x]

    built["extras"] = list(built.get("extras", [])) + extra
    return built


def _compose(a, b):
    """Two assertions in one sentence. `avoid` is dropped: v17's effect frame forbids
    mentioning concentrations while its mechanism frame requires them, so composing them
    with both avoid blocks intact would issue contradictory instructions."""
    keep_avoid = [x for src in (a, b) if src.get("variant") == "unlisted"
                  for x in src.get("avoid", []) if "listed drug" in x]
    return {"kind": a["kind"],
            "variants": [v for v in (a.get("variant"), b.get("variant")) if v],
            "shape": a.get("shape") or b.get("shape"),
            "says_groups": [(a["focus"], a["says"]), (b["focus"], b["says"])],
            "says": [], "avoid": sorted(set(keep_avoid)),
            "positives": list(a["positives"]) + list(b["positives"]),
            "focus": sorted(set(a["focus"]) | set(b["focus"]))}


N_ENTITIES = {2: 0.52, 3: 0.24, 4: 0.10, 5: 0.06, 6: 0.04, 7: 0.02, 8: 0.01, 10: 0.01}

# Three quantities cannot all be matched at once. Corpus mention counts, corpus modifier
# rates and corpus pair positive rate are jointly inconsistent under one-positive-pair
# assertions: at measured entity counts and measured appositive and coordinate rates the
# pair positive rate lands near 0.11 against the corpus's 0.162. v17 hit 0.162 exactly by
# tuning n_positives_by_k, while getting entity counts and construction rates wrong.
# P_LIST_TARGET is the honest lever, since list objects are what actually produce
# multi-pair positives in the corpus, but it is set from judgement rather than
# measurement. Measure it on the 295 verdicts before trusting the number.


def make_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                   reasoning_effort="low", max_output_tokens=1500, api="responses"):
    """v17's sampler with v18's render. prompt.make_sample_fn closes over prompt.render,
    so it cannot be reused directly. SYSTEM is unchanged: it states the task and the
    guideline conventions, neither of which this revision touches."""
    from pydantic import BaseModel

    class Written(BaseModel):
        sentence: str

    def _responses(user):
        kw = {}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        r = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": _v17.SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Written, temperature=temperature, **kw)
        if r.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(r, 'status', '?')})")
        return r.output_parsed.sentence

    def _chat(user):
        kw = {"max_tokens": max_output_tokens} if max_output_tokens else {}
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _v17.SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {
                "name": "written", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["sentence"],
                           "properties": {"sentence": {"type": "string"}}}}}, **kw)
        return Written.model_validate_json(r.choices[0].message.content).sentence

    def sample_fn(spec):
        u = render(spec)
        return {"sentence": (_responses if api == "responses" else _chat)(u)}

    return sample_fn


def fingerprint():
    import hashlib
    parts = [_v17.SYSTEM] + sorted(ASSERTIONS) + sorted(MODIFIERS)
    parts += sorted(ROLE_HINTS.values()) + sorted(UNLISTED_TARGETS)
    parts += [f"{k}={v}" for k, v in sorted(N_ENTITIES.items())]
    parts += [str(P_SECOND_ASSERTION), str(P_UNLISTED_TARGET), str(P_LIST_TARGET),
              str(P_ROLE)]
    parts += [f"{k}={v}" for k, v in sorted(MODIFIERS.items(), key=lambda x: x[0])]
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def make_specs(n, vocab, seed=0):
    """One spec per sentence. Assertions bind pairs and carry labels; modifiers do not.

    A repeated mention is expressed as a duplicate entity key sharing one surface form.
    The gold matrix marks only the pair the assertion binds, so the second mention's
    pairs are NONE, which is rule P1. `v14_sample_to_instances` already resolves an
    asserted pair to the closest matching mentions by character offset, which is the
    right P1 proxy, so no resolver change is needed for this.
    """
    group_set = set(vocab.groups)
    rng = random.Random(seed)
    aw = {k: v["w"] for k, v in ASSERTIONS.items()}
    specs = []

    for i in range(n):
        register = _pick(REGISTERS, rng)

        a_name = _pick(aw, rng)
        second = None
        if rng.random() < P_SECOND_ASSERTION:
            if rng.random() < P_SECOND_IS_NEGATIVE:
                pool = {k: v for k, v in aw.items()
                        if k != a_name and k in NEGATIVE_ASSERTIONS}
            else:
                pool = {k: v for k, v in aw.items() if k != a_name}
            if pool:
                second = _pick(pool, rng)

        need = ASSERTIONS[a_name]["n"] + (ASSERTIONS[second]["n"] if second else 0)
        k = max(need, _pick(N_ENTITIES, rng))

        surfaces, seen = [], set()
        for _ in range(400):
            if len(surfaces) == k:
                break
            cand = vocab.sample(1, rng)[0]
            if _unusable(cand):
                continue
            low = cand.lower()
            if low in seen or any(low in s or s in low for s in seen):
                continue
            seen.add(low)
            surfaces.append(cand)
        if len(surfaces) < k:
            raise RuntimeError("vocab exhausted while sampling distinct names")

        ents = [{"key": chr(65 + j), "surface": s,
                 "type": "GROUP" if s in group_set else "DRUG"}
                for j, s in enumerate(surfaces)]

        mods = set()
        for name, p in MODIFIERS.items():
            rate = p[register] if isinstance(p, dict) else p
            if rng.random() < rate:
                mods.add(name)

        built = ASSERTIONS[a_name]["f"](ents, rng)
        if rng.random() < P_UNLISTED_TARGET:
            built = _to_unlisted(built, rng)
        elif rng.random() < P_LIST_TARGET:
            built = _to_list_target(built, ents, rng)
        if second:
            sec = ASSERTIONS[second]["f"](ents, rng)
            if rng.random() < P_UNLISTED_TARGET:
                sec = _to_unlisted(sec, rng)
            built = _compose(built, sec)
        else:
            built = {**built, "says_groups": [(built["focus"], built["says"])],
                     "says": []}
        built = _apply_modifiers(built, ents, mods, rng)

        keys_now = _keys(ents)
        focus_now = set(built["focus"])
        named = set()
        for _, rows in built.get("says_groups", []):
            named |= _referenced(rows)
        named |= _referenced(built["says"]) | focus_now
        if built.get("shape"):
            named |= set(re.findall(r"\{(\w)\}", built["shape"]))

        if "repeated_mention" in mods:
            host = rng.choice(_keys(ents))
            dup = {"key": chr(65 + len(ents)), "surface": ents[
                [e["key"] for e in ents].index(host)]["surface"],
                "type": "DRUG", "duplicate_of": host}
            ents = ents + [dup]
            built["repeat"] = host

        keys = _keys(ents)
        focus = set(built["focus"])
        spare = [x for x in keys if x not in focus
                 and not any(e["key"] == x and e.get("duplicate_of")
                             for e in ents)]
        orphans = [x for x in spare if x not in named]
        rest = [x for x in spare if x in named]
        roled = orphans + [x for x in rest if rng.random() < P_ROLE]
        roled = roled[:max(2, len(orphans))]
        roles, used = {}, set()
        for x in roled:
            avail = [r for r in ROLE_CODES if r not in used]
            if not avail:
                break
            r = rng.choice(avail)
            used.add(r)
            roles[x] = r

        matrix = {f"{a}|{b}": "NONE" for a, b in itertools.combinations(keys, 2)}
        for a, b, lab in built["positives"]:
            matrix[f"{min(a, b)}|{max(a, b)}"] = lab
        for host, member in built.get("appositive_pairs", []):
            for a, b, lab in built["positives"]:
                for other in (a, b):
                    if other not in (host, member):
                        p = f"{min(member, other)}|{max(member, other)}"
                        matrix[p] = lab
        for host, partner in built.get("coordinate_pairs", []):
            for a, b, lab in built["positives"]:
                if host in (a, b):
                    other = b if a == host else a
                    p = f"{min(partner, other)}|{max(partner, other)}"
                    matrix[p] = lab

        specs.append({
            "spec_index": i,
            "assertions": [a_name] + ([second] if second else []),
            "variants": built.get("variants",
                                  [built["variant"]] if built.get("variant") else []),
            "modifiers": sorted(mods),
            "frame": a_name,
            "kind": built["kind"],
            "register": register,
            "register_line": rng.choice(REGISTER_LINE[register]),
            "entities": ents,
            "shape": built.get("shape"),
            "says": built.get("extras", []),
            "says_groups": built.get("says_groups", []),
            "avoid": built.get("avoid", []),
            "positives": [{"between": [a, b], "label": lab}
                          for a, b, lab in built["positives"]],
            "asserts": [{"between": [a, b], "label": lab}
                        for a, b, lab in built["positives"]],
            "roles": roles,
            "repeat": built.get("repeat"),
            "scene": rng.choice(SCENES[register]) if rng.random() < P_SCENE else None,
            "axes": _sample_axes(built["kind"], register, rng),
            "matrix": {k: v for k, v in matrix.items()},
        })
    return specs


def render(spec):
    """Two changes from v17's render.

    `says` is grouped by the entities it concerns. With two assertions in one sentence a
    flat list gives the model no way to tell which pair each row is about.

    Roles are codes with a hint, not fixed English. v17's seven ROLES strings are
    rendered verbatim and supply all eight of the most repeated 4-grams in its output.
    prompt.py:228 records that cutting P_ROLE to 0.2 cost 0.035 F1, so prevalence was
    the wrong lever; the fixed surface forms are the right one.

    Note for gates: role position and role adjacency are currently recovered by matching
    the ROLES strings, which is why they were frozen. The spec now records `roles` by
    entity key, so the gates should read that instead of the text.
    """
    by_key = {e["key"]: e["surface"] for e in spec["entities"]}

    def sub(t):
        for k, v in by_key.items():
            t = t.replace("{" + k + "}", v)
        return t

    lines = [spec["register_line"]]
    if spec["scene"]:
        lines.append(f"the drugs below are {spec['scene']}")
    lines.append("")

    surfaces = []
    for e in spec["entities"]:
        if e.get("duplicate_of"):
            continue
        surfaces.append(e["surface"])
    lines.append(f"drug names, use all {len(surfaces)}")
    lines += [f"  {s}" for s in surfaces]
    lines.append("")

    if spec["shape"]:
        lines += ["arrangement", f"  {sub(spec['shape'])}", ""]

    groups = []
    for keys, says in spec.get("says_groups", []):
        if not says:
            continue
        names = " and ".join(by_key[k] for k in keys if k in by_key)
        if groups and groups[-1][0] == names:
            groups[-1] = (names, groups[-1][1] + says)
        else:
            groups.append((names, says))

    for names, says in groups:
        lines.append(f"says about {names}" if len(groups) > 1 else "says")
        w = max(len(k) for k, _ in says)
        lines += [f"  {k.ljust(w)}  {sub(v)}" for k, v in says]
        lines.append("")

    if spec["says"]:
        lines.append("also")
        w = max(len(k) for k, _ in spec["says"])
        lines += [f"  {k.ljust(w)}  {sub(v)}" for k, v in spec["says"]]
        lines.append("")

    if spec["avoid"]:
        lines += ["avoid"] + [f"  {sub(s)}" for s in spec["avoid"]] + [""]

    if spec.get("repeat"):
        lines += ["name twice",
                  f"  {by_key[spec['repeat']]}, once more later in the sentence, in a "
                  f"clause that is not the one making the claim above", ""]

    if spec["roles"]:
        lines.append("the remaining drugs, put each in the sentence in your own words")
        lines += [f"  {by_key[k]}  {ROLE_HINTS[r]}" for k, r in spec["roles"].items()]
        lines.append("")

    if spec["axes"]:
        lines += ["how to write it"] + [f"  {v}" for v in spec["axes"].values()] + [""]

    return "\n".join(lines).rstrip()