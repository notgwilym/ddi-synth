from pydantic import BaseModel
from typing import Literal
import random, hashlib

class Entity(BaseModel):
    text: str
    type: str = "drug"

class Relation(BaseModel):
    arg1: str                       # drug name, must appear in this sentence
    arg2: str
    label: Literal["ADVISE", "EFFECT", "INT", "MECHANISM"]

class Sentence(BaseModel):
    text: str
    relations: list[Relation] = []  # interactions asserted IN this sentence

class Generated(BaseModel):
    sentences: list[Sentence]       # the passage, pre-split by the model
    entities: list[Entity]          # every drug surface form used

VIGNETTE_SYSTEM = """You are writing short passages to serve as synthetic training data for a drug-interaction classifier. These are NOT real clinical claims; do not worry about whether an interaction is pharmacologically real, only that it is clearly written and correctly typed.

You are given a document framing, a pool of drugs, and target drug-drug interaction (DDI) types to assert. A DDI is a change in the effects of one drug by the presence of another. The types:

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
  
  
RULES
- Assert each target interaction once, in natural prose, between a plausible-sounding pair of drugs from the pool. Pick a pair and commit; do not deliberate over realism.
- Put each interaction ENTIRELY within one sentence; both named drugs must appear in that sentence. Record it in that sentence's relations (the two drug names exactly as written, and the type).
- Assert at most ONE interaction per drug pair. Never give one pair two types.
- Every pool drug must appear. Drugs not in an interaction are mentioned only in their given role, as bare context -- never comment on their safety. Do NOT write "no interaction", "without interaction", "no reported interaction", "without adverse outcomes", "well tolerated", "may be taken safely", or any similar phrase. A non-interacting drug is simply named in its role and left alone.
- Do not collect leftover drugs into a summary sentence ("the regimen also included...").
- Don't include the labels (MECHANISM, EFFECT, ADVISE, INT) in the text. Express the interaction in ordinary clinical language. 
- Only record a relation for a sentence that actually asserts an interaction between two named drugs from the pool. A sentence that merely mentions a drug in its role has no relations — leave its relation list empty. Every relation must name two different drugs.

Return the passage as a list of sentences. For each sentence, record any interactions asserted in it, giving the two interacting drugs by their exact surface form as written in that sentence, and the interaction type. List every drug surface form you use. Use plain ASCII prose; no markdown."""

FRAMINGS = {
    "DrugBank": [
        "the interactions subsection of a drug's prescribing information",
        "a drug-label precautions statement",
        "a summary of product characteristics, interactions section",
    ],
    "MedLine": [
        "the findings section of a biomedical journal abstract",
        "a short case report of a patient taking several medications",
        "a pharmacovigilance case series describing co-administration",
        "a review of concomitant drug therapy in a treated patient",
    ],
}

ROLES = ["a background therapy", "a comparator agent", "a prior medication",
         "a concurrent treatment for an unrelated condition",
         "part of the patient's regular regimen", "an incidental co-medication"]

def render_vignette(spec):
    lines = [f"Write {spec['framing']}.", ""]
    lines += ["Drugs to include (use every one, exactly as spelled):"]
    lines += [f"  - {d}" for d in spec["drug_pool"]]
    lines += ["", "Assert these interactions (you choose which drugs realise each):"]
    lines += [f"  - one {t['label']} interaction" for t in spec["targets"]]
    lines += ["", "Any drug not part of an interaction should appear woven into the "
              "narrative in a distinct role such as: " + ", ".join(spec["roles"]) + ". "
              "Do not collect them into a trailing summary sentence."]
    return "\n".join(lines)

LABELS = ["ADVISE", "EFFECT", "INT", "MECHANISM"]
DEFAULT_POOL_SIZE = {3: 0.25, 4: 0.45, 5: 0.20, 6: 0.10}

def _pick(dist, rng):
    ks = list(dist); return rng.choices(ks, weights=[dist[k] for k in ks], k=1)[0]

def make_vignette_specs(n, vocab, seed=0, label_dist=None, pool_size=None,
                        registers=None, n_targets=(1, 2)):
    label_dist = label_dist or {l: 1/len(LABELS) for l in LABELS}
    pool_size  = pool_size or DEFAULT_POOL_SIZE
    registers  = registers or {"DrugBank": 0.5, "MedLine": 0.5}
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        register = _pick(registers, rng)
        k = _pick(pool_size, rng)
        pool = vocab.sample(k, rng)
        n_t = min(rng.randint(*n_targets), max(1, k - 1))
        specs.append({
            "register": register,
            "framing": rng.choice(FRAMINGS[register]),
            "drug_pool": pool,
            "targets": [{"label": _pick(label_dist, rng)} for _ in range(n_t)],
            "roles": rng.sample(ROLES, min(len(ROLES), max(2, k - n_t))),
        })
    return specs

def make_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                   reasoning_effort="low", max_output_tokens=3000):
    def sample_fn(spec):
        kw = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": VIGNETTE_SYSTEM},
                   {"role": "user", "content": render_vignette(spec)}],
            text_format=Generated, temperature=temperature,
            max_output_tokens=max_output_tokens, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp,'status','?')})")
        return resp.output_parsed.model_dump()
    return sample_fn

def prompt_fingerprint():
    blob = VIGNETTE_SYSTEM
    return hashlib.sha256(blob.encode()).hexdigest()[:12]

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

CONEG_SYSTEM = """You are writing short clinical passages that list several medications a patient is taking concurrently, to serve as synthetic training data.

You are given a document framing and a pool of drugs. Write a 1-3 sentence passage that mentions ALL of these drugs together as part of a shared medication regimen. Group several drugs per sentence.

Example: "The patient's regimen included metformin, ramipril, and atorvastatin for diabetes, hypertension, and cardiovascular risk respectively, alongside amoxicillin for a concurrent infection."

RULES
- Every drug in the pool must appear in the text, grouped naturally with others.
- You MUST list every drug you write, in the `entities` field, copying each name exactly as it appears in the text. The entities field must never be empty.
- Describe the drugs only as co-administered. Do not describe any drug as affecting or interacting with another, and do not write "no interaction" or similar. Because no interactions are described, each sentence's `relations` field will be an empty list -- but `entities` must still contain every drug.

Plain ASCII prose, no markdown."""


def render_coneg(spec):
    lines = [f"Write {spec['framing']}.", "",
             "Mention all of these drugs together as concurrent, non-interacting medications:"]
    lines += [f"  - {d}" for d in spec["drug_pool"]]
    lines += ["", "Group several per sentence. Assert NO interactions between any of them."]
    return "\n".join(lines)

def make_coneg_specs(n, vocab, seed=1, pool_size=None, registers=None):
    pool_size = pool_size or {4: 0.3, 5: 0.4, 6: 0.3}   # bigger pools -> more NONE pairs
    registers = registers or {"DrugBank": 0.5, "MedLine": 0.5}
    rng = random.Random(seed)
    specs = []
    for _ in range(n):
        register = _pick(registers, rng)
        k = _pick(pool_size, rng)
        specs.append({"register": register,
                      "framing": rng.choice(FRAMINGS[register]),
                      "drug_pool": vocab.sample(k, rng)})
    return specs

def make_coneg_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                         reasoning_effort="low", max_output_tokens=4000):
    def sample_fn(spec):
        kw = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": CONEG_SYSTEM},
                   {"role": "user", "content": render_coneg(spec)}],
            text_format=Generated, temperature=temperature,
            max_output_tokens=max_output_tokens, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp,'status','?')})")
        return resp.output_parsed.model_dump()
    return sample_fn