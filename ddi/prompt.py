import random

LABELS = ["ADVISE", "EFFECT", "INT", "MECHANISM"]

# Drug count drives the negative ratio for free: with one positive relation,
# C(n,2) pairs means n=3 -> 67% NONE, n=4 -> 83%, n=5 -> 90%.

DEFAULT_N_DRUGS = {3: 0.25, 4: 0.45, 5: 0.2}
DEFAULT_REGISTERS = {"DrugBank": 0.5, "MedLine": 0.5}


def _pick(dist, rng):
    keys = list(dist)
    return rng.choices(keys, weights=[dist[k] for k in keys], k=1)[0]


def make_specs(n, vocab, seed=0, label_dist=None, n_drugs_dist=None, registers=None):
    """Build n generation specs.

    label_dist controls the class balance of the OUTPUT -- crank INT up to mint a
    surplus of the class that is thinnest in the real corpus.
    """
    label_dist = label_dist or {l: 1 / len(LABELS) for l in LABELS}
    n_drugs_dist = n_drugs_dist or DEFAULT_N_DRUGS
    registers = registers or DEFAULT_REGISTERS
    rng = random.Random(seed)

    specs = []
    for _ in range(n):
        k = _pick(n_drugs_dist, rng)
        drugs = vocab.sample(k, rng)
        i, j = rng.sample(range(k), 2)          # which pair actually interacts
        specs.append({
            "drugs": drugs,
            "relations": [{"arg1": i, "arg2": j, "label": _pick(label_dist, rng)}],
            "register": _pick(registers, rng),
        })
    return specs


# prompt
SYSTEM = """You write single sentences for a drug-drug interaction (DDI) corpus.

You are given a list of substances and ONE interaction to express between two of \
them. Write one sentence that mentions ALL the given substances, asserts the \
requested interaction between the specified pair, and asserts NO interaction \
between any other pair. The other substances are mentioned only as things the \
patient is also taking. Do not describe them as interacting, and do not advise \
anything about them."

Label definitions (DDI-2013):
- MECHANISM: a pharmacokinetic interaction -- a change in absorption, distribution, \
metabolism or excretion, or in plasma levels, concentration, clearance, AUC or half-life.
- EFFECT: a pharmacological effect, clinical finding, sign or symptom, a change in \
one drug's effect, increased toxicity, or a protective effect. Also pharmacodynamic \
mechanisms (additive, synergistic, antagonistic).
- ADVISE: a recommendation or advice about using the two together (caution, avoid, \
contraindicated, monitor, adjust dosage).
- INT: an interaction is stated to occur, with NO further detail about effect, \
mechanism or advice.

Rules:
1. Copy each substance name character-for-character from the list. Same spelling, \
same case, same punctuation. Do not pluralise, abbreviate, expand or correct them, \
even if a name looks like a code or an odd chemical.
2. State the interaction as a fact of the sentence. Do not negate it and do not \
describe it as merely studied: "X and Y do not interact" and "the interaction of X \
and Y was investigated" are both wrong.
3. Substances outside the requested pair appear in the sentence but take no part in \
any interaction -- as co-medications, list members, or background context.
4. List every substance mention in `entities`, in order of appearance. A name that \
appears twice is listed twice, with different ids.
5. `relations` contains exactly the one requested interaction.
6. Never write the words ADVISE, EFFECT, MECHANISM or INT in the sentence. Express \
the interaction through ordinary clinical language instead.
7. Plain text only: no markdown, no asterisks, no footnote marks. Use ordinary ASCII \
hyphens and spaces.
8. One sentence, under 60 words. Write natural biomedical prose even when the \
substances are unusual, and never remark on whether the pairing is realistic."""

REGISTER_HINT = {
    "DrugBank": ("Style: terse drug product-label prose, as in a package insert. "
                 "Often begins with a substance name followed by a colon."),
    "MedLine":  ("Style: a sentence from a biomedical research abstract. "
                 "Longer and more academic, may mention study context."),
}


def render_spec(spec):
    drugs = spec["drugs"]
    rel = spec["relations"][0]
    lines = ["Substances (use these exact strings):"]
    lines += [f"  - {d}" for d in drugs]
    lines.append("")
    lines.append(f"Assert a {rel['label']} interaction between "
                 f"{drugs[rel['arg1']]!r} and {drugs[rel['arg2']]!r}.")
    others = [d for i, d in enumerate(drugs) if i not in (rel["arg1"], rel["arg2"])]
    if others:
        lines.append("These must appear but must NOT interact with anything: "
                     + ", ".join(repr(o) for o in others))
    lines.append("")
    lines.append(REGISTER_HINT[spec["register"]])
    return "\n".join(lines)


def make_sample_fn(client, model="gpt-oss-120b", temperature=0.7,
                   reasoning_effort="low", max_output_tokens=3000):
    """Returns a spec -> dict function for synth.generate_raw().

    max_output_tokens : reasoning tokens count against this budget, so it must be
        well above the size of the JSON itself -- too tight and responses come back
        status='incomplete' with truncated JSON. The degeneracy check in
        synth.sample_to_instances is what actually catches repetition loops.
    """
    from pydantic import BaseModel
    from typing import Literal

    class Entity(BaseModel):
        id: str            # T1, T2, ... in order of appearance
        text: str          # exact substring of the sentence
        type: str = "drug"

    class Relation(BaseModel):
        arg1_id: str
        arg2_id: str
        label: Literal["ADVISE", "EFFECT", "INT", "MECHANISM"]

    class Generated(BaseModel):
        sentence: str
        entities: list[Entity]
        relations: list[Relation]

    def sample_fn(spec):
        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": render_spec(spec)}],
            text_format=Generated,
            temperature=temperature,
            max_output_tokens=max_output_tokens,   # hard stop on repetition loops
            **kwargs,
        )
        parsed = resp.output_parsed
        if parsed is None:                      # refusal, truncation, or parse failure
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return parsed.model_dump()

    return sample_fn

VERIFIER_SYSTEM = """You are an expert biomedical annotator applying the DDI-2013 annotation guidelines. You are shown one sentence with two drug entities marked [E1]...[/E1] and [E2]...[/E2]. Assign the type of drug-drug interaction (DDI) asserted in the sentence between the [E1] drug and the [E2] drug, considering ONLY what the sentence states.

A DDI is a change in the effects of one drug by the presence of another drug. Annotate the relationship between the two marked entities only. Interactions are annotated at sentence level: information in other sentences is irrelevant.

LABELS

MECHANISM — assigned when a PHARMACOKINETIC mechanism is described: a change in how a drug is absorbed, distributed, metabolized or excreted, or a change in its levels or concentration. This includes volume of distribution, bioavailability, peak level, AUC, clearance and half-life.
  - "Grepafloxacin, like other quinolones, may inhibit the metabolism of caffeine." -> MECHANISM
  - "probenecid increased the AUC by 25 percent and reduced the plasma and renal clearances." -> MECHANISM
  - "Elevated plasma levels of theophylline have been reported with concomitant quinolone use." -> MECHANISM

EFFECT — assigned when the sentence describes the EFFECT of the interaction: a pharmacological effect, a clinical finding, a sign or symptom, an increase in toxicity, a protective effect, therapeutic failure, or an unspecified modification of one drug's effect. ALSO assigned when the sentence describes a PHARMACODYNAMIC mechanism (synergistic/additive/potentiated, or antagonistic).
  - "The concomitant administration of ciprofloxacin with glyburide has resulted in severe hypoglycemia." -> EFFECT
  - "Quinolones may enhance the effects of the oral anticoagulant, warfarin." -> EFFECT
  - "Antagonism has been demonstrated between clindamycin and erythromycin in vitro." -> EFFECT (pharmacodynamic)
  - "Methionine may protect against the ototoxic effects of gentamicin." -> EFFECT (protective)

ADVISE — assigned when a recommendation or advice about the concomitant use of the two drugs is given.
  - "UROXATRAL should not be used in combination with other alpha-blockers." -> ADVISE
  - "DISULFIRAM should be used with caution in those patients receiving PHENYTOIN." -> ADVISE

INT — assigned when the sentence states that an interaction occurs but gives NO information about its effect, mechanism, or any advice, so none of the other three types can apply. Often appears in abstract titles.
  - "The interaction of omeprazole and ketoconazole has been established." -> INT
  - "linezolid has the potential for interaction with adrenergic and serotonergic agents." -> INT

NONE — assigned when the sentence asserts NO annotatable interaction between the two MARKED drugs. This is the default: if the sentence does not assert an interaction between [E1] and [E2] that fits one of the four types above, the answer is NONE. Assign NONE when:
  - The two marked drugs are merely co-mentioned, co-listed, or co-administered, with no interaction asserted between THEM (e.g. one is a background co-medication, or both appear in a list of a third drug's interactions).
  - The asserted interaction is between one marked drug and some OTHER drug/class in the sentence, not between [E1] and [E2].
  - The interaction is NEGATED — stated not to occur or not to alter pharmacokinetics (§4.5.1). "The pharmacokinetics of CANCIDAS are not altered by itraconazole." -> NONE
  - The sentence only states that an interaction was STUDIED/investigated, without confirming it occurs (§4.5.5). "The interaction of prostaglandin F2alpha and oxytocin was studied in vitro." -> NONE
  - The sentence describes interference with a LABORATORY TEST, not a drug-drug interaction.

RULES
- Annotate certainty-independently: a possible/suggested/in-vitro interaction IS annotated if it is asserted to occur (§4.5.2).
- Annotate beneficial and harmful interactions alike (§4.5.3).
- If an interaction is affirmed, annotate it even if a negation also appears for the same pair in the sentence (§4.5.4). "Although this interaction has not been reported with cinoxacin, caution should be exercised when cinoxacin is given with caffeine." -> ADVISE
- TIE-BREAK when more than one type could fit: a pharmacokinetic description is MECHANISM; a pharmacodynamic description is EFFECT. If advice AND an effect/mechanism both appear, prefer the type that the sentence most specifically asserts about the pair.

Return JSON: {label, justification} — justification one clause, quoting the span of text that determines the label.
"""

def make_verifier(client, model="gpt-oss-120b", temperature=0,
                   reasoning_effort=None, max_output_tokens=3000):
    from pydantic import BaseModel
    from typing import Literal

    class Verdict(BaseModel):
        label: Literal["NONE","ADVISE","EFFECT","INT","MECHANISM"]
        justification: str   # one clause, cite the span that decides it
        
    def sample_fn(spec):
        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": VERIFIER_SYSTEM},
                   {"role": "user", "content": spec["text"]}],
            text_format=Verdict,
            temperature=temperature,
            max_output_tokens=max_output_tokens,   # hard stop on repetition loops
            **kwargs,
        )
        parsed = resp.output_parsed
        if parsed is None:                      # refusal, truncation, or parse failure
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return parsed.model_dump()

    return sample_fn

def prompt_fingerprint():
    import hashlib
    blob = SYSTEM + "".join(f"{k}{v}" for k, v in sorted(REGISTER_HINT.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]