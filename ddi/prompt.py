"""Generator v14.

The failure this replaces: v13 drew positives from the vignette generator (~1 pair per
sentence) and negatives from CONEG (6-15 pairs), so entity count predicted the label.
One-pair sentences were 75% positive, multi-pair ~100% negative, hard negative rate
0.001 against a corpus prose rate of ~0.50. The entity markers carried no signal at all
because pair identity was never a variable. Here every sentence comes from one sampler
and one prompt, and zero-assertion specs are drawn from the same distribution, so no
structural property can separate the classes.

Four things about the prompt that are easy to get wrong, and were:

The label NAME must not reach the model: v13 wrote "a clear MECHANISM interaction" into
sentences. The label CRITERIA must reach it, or a spec describing a change in exposure
can produce a sentence an annotator would read as a clinical effect, and the gold label
is then wrong.

Nothing in a spec may be a phrase the model can lift. Content slots are noun phrases
that legitimately belong in the output, but ROLES are predicates, and written as English
they were copied straight through. They are opaque tokens, glossed once in the system
prompt.

The scene must render on EVERY spec. When it only appeared where non-participants
existed, scene vocabulary ("regimen", "treatment pathway", "administered") correlated
almost perfectly with the absence of an assertion, and a bag-of-words probe recovered
the label from it. That is the v13 shortcut in lexical form.

Where non-participants sit must be sampled. Left free, they landed after the assertion
96% of the time, so the classifier could learn "second half of the sentence means NONE"
without reading the markers at all. That is the same shortcut in positional form.
"""
import hashlib
import itertools
import random

LABELS = ["ADVISE", "EFFECT", "INT", "MECHANISM"]

# PRIOR is hand-specified from the annotation guidelines, matching the existing
# p_group=0.3 policy and keeping the from-scratch claim clean. MEASURED comes from the
# training split with the eight monster sentences excluded. Run both as arms: if they
# perform alike, the corpus statistics were a convenience rather than a crutch.
PRIOR = {
    "n_entities": {2: 0.40, 3: 0.30, 4: 0.20, 5: 0.10},
    "labels": {l: 0.25 for l in LABELS},
    "registers": {"DrugBank": 0.5, "MedLine": 0.5},
}
MEASURED = {
    "n_entities": {2: 0.410, 3: 0.272, 4: 0.137, 5: 0.073, 6: 0.040},
    "labels": {"EFFECT": 0.440, "MECHANISM": 0.289, "ADVISE": 0.210, "INT": 0.061},
    "registers": {"DrugBank": 0.85, "MedLine": 0.15},
}
COMPOSITION = {"prior": PRIOR, "measured": MEASURED}

N_POSITIVES_BY_K = {
    2: {0: 0.65, 1: 0.35},
    3: {0: 0.40, 1: 0.45, 2: 0.15},
    4: {0: 0.30, 1: 0.45, 2: 0.25},
    5: {0: 0.25, 1: 0.40, 2: 0.35},
    6: {0: 0.20, 1: 0.40, 2: 0.40},
}

# Keys are opaque: "concurrent" and "subsequent" were being used as adjectives lifted
# straight from the field ("with concurrent Amedalin, later followed by subsequent
# Amidephrine"). group records whether the role implies concurrent use: "separated"
# roles make an interaction semantically unavailable, so the NONE label is safe by
# construction, while "concurrent" roles are the realistic hard-negative case. Using
# only the separated ones would trade one shortcut for another.
ROLES = {
    "r1": ("separated", "was given earlier and has stopped"),
    "r2": ("separated", "started after the others ended"),
    "r3": ("separated", "evaluated separately, for comparison"),
    "r4": ("separated", "used if the first choice is unsuitable"),
    "r5": ("separated", "not given, or not permitted"),
}
P_ROLE = 0.55
ROLE_POS = {"before": 0.40, "after": 0.60}
_ROLE_GLOSS = "\n".join(f"  {k}   {v[1]}" for k, v in ROLES.items())


SYSTEM = f"""You write single sentences of biomedical text for a drug interaction corpus.

Each request lists drug names, the content the sentence carries, and a style. Write
exactly one sentence.

WHAT THE SENTENCE ASSERTS

A request either asks you to assert something between two named drugs, or asks for no
assertion at all. When it does, the "scope" line says which kind of statement to write.
The four kinds are exclusive. Write the one asked for and none of the others: a sentence
that mixes them cannot be used.

1. Exposure and handling / mechanism. How much of one drug is present, or how the body
   takes it up, distributes it, breaks it down or removes it, changed by the presence of
   the other. Say what changes and in which direction. Do not say what happens to the
   patient as a result.

2. Clinical consequence / effect. What follows for the patient when the two are
   combined: an effect, a risk, a loss of efficacy. Say what the consequence is. Do not
   explain the handling or the levels that produce it. The consequence follows from the
   two being used together; do not compare one drug against the other.

3. Recommended action / advice. What a prescriber should do about the combination. Say
   what should be done. Do not state the consequence or the mechanism motivating it.

4. Bare interaction. That the two interact, and nothing further. No mechanism, no
   consequence, no advice, no figures.

Only the pair named in the statement block is being asserted about. Other drugs in the
list are present in the sentence but nothing is claimed between them, or between them
and the pair.

When two statements are given they share a drug. Write them as one coordinated
statement rather than repeating the same clause twice.

WHEN NO ASSERTION IS ASKED FOR

Some requests have no statement block. Write ordinary clinical or research prose that
places every listed drug in the situation described at the top. Nothing in the sentence
says that any drug affects any other. Write it as one connected scene, not as a list of
drugs each with its own clause.

OTHER DRUGS

Names under "other drugs" carry a code saying what part they play. The code is a
category, not a word to use: work the part into the sentence in your own words. A drug
marked r1 might appear as "previously treated with", "after discontinuing", or "who had
received".

{_ROLE_GLOSS}

RULES

- Use every drug name listed, and no others.
- Reproduce each name as given, including any punctuation or non-ASCII characters in it.
- Every drug you name must be doing something in the sentence.
- The drugs listed may not be combined in real practice. Write the sentence as specified.
- Write only prose. The field names and codes in the request are instructions to you,
  not words to put in the sentence.
- Vary how the sentence opens. Do not begin every sentence the same way.
- Keep it to one clause or two, about the length of a single sentence in a drug label or
  an abstract.
- Match verb agreement to the name; some names are plural.
- One sentence. No markdown."""


SCOPE = {
    "MECHANISM": "exposure and handling only, no clinical consequence",
    "EFFECT": "clinical consequence only, no explanation of handling or levels",
    "ADVISE": "recommended action only, no consequence and no mechanism",
    "INT": "bare interaction only, no mechanism, consequence, advice or figures",
}


# ---------------------------------------------------------------- content slots
SITES = {
    "hepatic_metabolism": ["hepatic metabolism", "CYP-mediated metabolism",
                           "oxidative metabolism", "liver enzyme activity",
                           "first-pass metabolism"],
    "absorption": ["gastrointestinal absorption", "uptake from the gut",
                   "oral absorption", "absorption from the gastrointestinal tract"],
    "protein_binding": ["plasma protein binding", "binding to serum albumin",
                        "the protein-bound fraction"],
    "renal_clearance": ["renal clearance", "excretion by the kidney",
                        "tubular secretion", "urinary elimination"],
    "transporter": ["P-glycoprotein transport", "efflux transport",
                    "carrier-mediated transport", "intestinal transport"],
    "gastric_ph": ["gastric pH", "stomach acidity", "gastric acid secretion"],
}
EXPOSURE = ["higher", "lower"]

# "reduced" is deliberately absent: pairing a toxicity with a reduction produces a
# sentence asserting that one drug protects against the other's harm.
HARMS = {
    "bleeding": ["bleeding", "haemorrhage", "bleeding risk"],
    "sedation": ["sedation", "drowsiness", "central nervous system depression"],
    "hypotension": ["hypotension", "a drop in blood pressure"],
    "arrhythmia": ["arrhythmia", "irregular heart rhythm", "cardiac conduction changes"],
    "nephrotoxicity": ["nephrotoxicity", "impaired kidney function", "renal injury"],
    "hepatotoxicity": ["hepatotoxicity", "raised liver enzymes", "liver injury"],
    "hypoglycaemia": ["hypoglycaemia", "low blood glucose"],
    "myopathy": ["myopathy", "muscle damage", "rhabdomyolysis"],
    "seizures": ["seizures", "convulsions", "a lowered seizure threshold"],
    "respiratory_depression": ["respiratory depression", "slowed breathing"],
    "qt": ["QT prolongation", "prolongation of the QT interval"],
}
HARM_EXTENT = ["greater", "new"]

EFFICACY = {"loss_of_efficacy": ["therapeutic effect", "clinical efficacy",
                                 "the intended response"]}
EFFICACY_EXTENT = ["reduced", "lost"]

ACTIONS = {
    "avoid": ["avoidance of the combination", "contraindication"],
    "dose_reduction": ["dose reduction", "a lower starting dose", "dose adjustment"],
    "monitoring": ["closer monitoring", "monitoring of blood levels",
                   "clinical observation"],
    "timing": ["separation of dosing times", "staggered administration"],
    "discontinuation": ["discontinuation before the other is started",
                        "withdrawal of one agent"],
}
P_EFFICACY = 0.2


# ---------------------------------------------------------------- realisation
AXES = {
    "subject": ["the agent drug", "the affected drug", "the combination",
                "plasma concentrations", "the patients", "the study"],
    "voice": ["active", "passive"],
    "certainty": ["established", "possible", "observed", "a single case"],
    "detail": ["no figures", "direction only", "one figure",
               "a pharmacokinetic figure"],
    "position": ["main clause", "subordinate clause", "relative clause"],
    # 112 of ~150 MedLine sentences opened with "in a/the cohort/study". The register
    # line was supplying a fixed frame, so the opening is now drawn explicitly.
    "opening": ["with a drug name", "with the finding", "with a condition or setting",
                "with a subordinate clause", "with the recommendation"],
}
N_AXES = {1: 0.45, 2: 0.55}

LENGTH = {"short": 0.35, "medium": 0.40, "long": 0.25}
LENGTH_TEXT = {
    "short": "a single clause",
    "medium": "one clause with a subordinate clause",
    "long": "two or three clauses, with clinical detail",
}

REGISTER_BANS = {
    "DrugBank": {"subject": {"the patients", "the study"},
                 "certainty": {"a single case"},
                 "detail": {"a pharmacokinetic figure"}},
    "MedLine": {},
}
LABEL_BANS = {
    "MECHANISM": {},
    "EFFECT": {"subject": {"plasma concentrations"},
               "detail": {"a pharmacokinetic figure"}},
    "ADVISE": {"subject": {"plasma concentrations"},
               "certainty": {"a single case"},
               "detail": {"direction only", "one figure", "a pharmacokinetic figure"}},
    "INT": {"subject": {"plasma concentrations", "the combination"},
            "detail": {"direction only", "one figure", "a pharmacokinetic figure"},
            "position": {"relative clause"}},
}

NO_ASSERT_AXES = ["voice", "certainty", "subject", "opening"]
NO_ASSERT_SUBJECTS = ["one of the drugs listed", "the treatment", "the regimen",
                      "the patients"]
NO_ASSERT_BANS = {"subject": {"the patients"}}

REGISTER_LINE = {
    "DrugBank": "product label prose: impersonal, present tense, no study participants,"
                " no sample sizes",
    "MedLine": "research abstract prose: past tense, participants or patients, a sample"
               " size or percentage where detail is called for, hedged conclusions",
}

# Rendered on EVERY spec. Confined to specs with non-participants, scene vocabulary was
# a near-perfect predictor of the absence of an assertion.
SCENES = {
    "DrugBank": ["used in the same area of treatment",
                 "options for the same indication",
                 "part of the same treatment pathway",
                 "listed together on the same label"],
    "MedLine": ["compared in the same study",
                "given at different points in the patient's care",
                "the treatments recorded for this patient",
                "the options available for the same indication"],
}


def _pick(dist, rng):
    ks = list(dist)
    return rng.choices(ks, weights=[dist[k] for k in ks], k=1)[0]


def _sample_style(register, label, rng):
    """Axes filtered against register, scope, and whether anything is asserted."""
    pool = list(AXES) if label else list(NO_ASSERT_AXES)
    chosen = rng.sample(pool, min(_pick(N_AXES, rng), len(pool)))
    if "opening" not in chosen:
        chosen.append("opening")
    style = {}
    length = _pick(LENGTH, rng)
    style["length"] = LENGTH_TEXT[length]
    for ax in chosen:
        base = AXES[ax] if (label or ax != "subject") else NO_ASSERT_SUBJECTS
        banned = set(REGISTER_BANS.get(register, {}).get(ax, set()))
        banned |= set((LABEL_BANS[label] if label else NO_ASSERT_BANS).get(ax, set()))
        options = [o for o in base if o not in banned]
        if options:
            style[ax] = rng.choice(options)
    return style


def _content(label, rng):
    if label == "MECHANISM":
        site = rng.choice(list(SITES))
        return {"site": rng.choice(SITES[site]), "exposure": rng.choice(EXPOSURE),
                "_slots": {"site": site}}
    if label == "EFFECT":
        if rng.random() < P_EFFICACY:
            slot = "loss_of_efficacy"
            return {"outcome": rng.choice(EFFICACY[slot]),
                    "extent": rng.choice(EFFICACY_EXTENT),
                    "relative to": "the affected drug given alone",
                    "_slots": {"outcome": slot}}
        slot = rng.choice(list(HARMS))
        return {"outcome": rng.choice(HARMS[slot]),
                "extent": rng.choice(HARM_EXTENT),
                "relative to": "either drug given alone",
                "_slots": {"outcome": slot}}
    if label == "ADVISE":
        action = rng.choice(list(ACTIONS))
        return {"action": rng.choice(ACTIONS[action]), "_slots": {"action": action}}
    return {"_slots": {}}


def _assign_roles(roled, rng):
    roles, used = {}, set()
    for key in roled:
        candidates = [r for r in ROLES if r not in used]
        if not candidates:
            break
        role = rng.choice(candidates)
        used.add(role)
        roles[key] = {"role": role, "group": ROLES[role][0]}
    return roles


def make_v14_specs(n, vocab, seed=0, composition="prior"):
    """One spec per sentence. The label is decided here and never named to the model.

    vocab.sample(k, rng) returns k names and chooses groups with probability p_group;
    there is no typed sampling and no BRAND or DRUG_N in the vocabulary, so 14% of real
    entity types cannot be matched. Recorded rather than faked.
    """
    cfg = COMPOSITION[composition]
    group_set = set(vocab.groups)
    rng = random.Random(seed)
    specs = []

    for i in range(n):
        register = _pick(cfg["registers"], rng)
        key = _pick(cfg["n_entities"], rng)

        surfaces, seen = [], set()
        for _ in range(80):
            if len(surfaces) == key:
                break
            cand = vocab.sample(1, rng)[0]
            low = cand.lower()
            # a name containing another name breaks span resolution downstream
            if low in seen or any(low in s or s in low for s in seen):
                continue
            seen.add(low)
            surfaces.append(cand)
        if len(surfaces) < key:
            raise RuntimeError("vocab exhausted while sampling distinct names")

        ents = [{"key": chr(65 + j), "surface": s,
                 "type": "GROUP" if s in group_set else "DRUG"}
                for j, s in enumerate(surfaces)]
        keys = [e["key"] for e in ents]
        by_key = {e["key"]: e for e in ents}
        pairs = list(itertools.combinations(keys, 2))

        n_pos = _pick(N_POSITIVES_BY_K[key], rng)

        positives = []
        if n_pos >= 1:
            positives.append(rng.choice(pairs))
        if n_pos >= 2:
            hub = rng.choice(positives[0])
            others = [p for p in pairs if hub in p and p not in positives]
            if others:
                positives.append(rng.choice(others))

        # two asserts share label and content: they come off one hub in one sentence, so
        # a single scope and style block is honest rather than a compromise
        label = _pick(cfg["labels"], rng) if positives else None
        content = _content(label, rng) if positives else None
        asserts = [{"between": [a, b], "label": label, "content": content}
                   for a, b in positives]

        style = _sample_style(register, label, rng)

        participating = {x for p in positives for x in p}
        non_participants = [e["key"] for e in ents if e["key"] not in participating]
        cap = 2 if positives else max(1, len(non_participants) // 3)
        candidates = rng.sample(non_participants, min(len(non_participants), cap))
        roled = [key for key in candidates if rng.random() < P_ROLE]
        roles = _assign_roles(roled, rng)
        role_pos = _pick(ROLE_POS, rng) if (roles and asserts) else None

        matrix = {f"{a}|{b}": "NONE" for a, b in pairs}
        for asrt in asserts:
            a, b = sorted(asrt["between"])
            matrix[f"{a}|{b}"] = asrt["label"]

        specs.append({
            "spec_index": i,
            "register": register,
            "entities": ents,
            "asserts": asserts,
            "style": style,
            "roles": roles,
            "role_pos": role_pos,
            "scene": rng.choice(SCENES[register]),
            "matrix": matrix,
        })
    return specs


def render_v14(spec):
    """One names block with a count: split drug/class blocks lost a class name in three
    of three samples, and the inline type marker was written into sentences. Style
    renders last, so the model knows what it is styling before it is told how."""
    by_key = {e["key"]: e for e in spec["entities"]}
    names = [(e["surface"], "") for e in spec["entities"]]
    blocks = [(f"names to use, all {len(names)}", names)]

    for n, asrt in enumerate(spec["asserts"]):
        a, b = asrt["between"]
        rows = [("scope", SCOPE[asrt["label"]]),
                ("agent", by_key[a]["surface"]),
                ("affected", by_key[b]["surface"])]
        rows += [(kk, vv) for kk, vv in asrt["content"].items() if kk != "_slots"]
        head = "statement" if len(spec["asserts"]) == 1 else f"statement {n + 1}"
        blocks.append((head, rows))

    if spec["roles"]:
        rows = [(by_key[k]["surface"], v["role"]) for k, v in spec["roles"].items()]
        if spec["role_pos"]:
            rows.append(("these come", f"{spec['role_pos']} the main statement"))
        blocks.append(("other drugs", rows))

    if spec["style"]:
        blocks.append(("style", list(spec["style"].items())))

    lines = [REGISTER_LINE[spec["register"]], f"these are {spec['scene']}", ""]
    for head, rows in blocks:
        lines.append(head)
        width = max((len(k) for k, _ in rows), default=0)
        lines += [f"  {k.ljust(width)}  {v}".rstrip() for k, v in rows]
        lines.append("")
    return "\n".join(lines).rstrip()


def make_v14_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                       reasoning_effort="low", max_output_tokens=1500,
                       api="responses"):
    """Returns {sentence}. Nothing else: the label lives in the spec, and asking for it
    back would put label vocabulary into the prompt.

    Low effort by default. At medium the model spent 300-570 reasoning tokens per call,
    much of it counting words by hand and restating the register rules; at low it spends
    ~100 and the sentences are no worse. max_output_tokens bounds a runaway rather than
    letting it burn the full context, which turns a stall into a fast incomplete.
    """
    from pydantic import BaseModel

    class Written(BaseModel):
        sentence: str

    def _responses(user):
        kw = {}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Written, temperature=temperature, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return resp.output_parsed.sentence

    def _chat(user):
        kw = {"max_tokens": max_output_tokens} if max_output_tokens else {}
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {
                "name": "written", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["sentence"],
                           "properties": {"sentence": {"type": "string"}}}}}, **kw)
        return Written.model_validate_json(resp.choices[0].message.content).sentence

    def sample_fn(spec):
        user = render_v14(spec)
        return {"sentence": (_responses if api == "responses" else _chat)(user)}

    return sample_fn


def v14_fingerprint():
    parts = [SYSTEM] + sorted(SCOPE.values())
    parts += sorted(a for v in SITES.values() for a in v)
    parts += sorted(a for v in HARMS.values() for a in v)
    parts += sorted(a for v in EFFICACY.values() for a in v)
    parts += sorted(a for v in ACTIONS.values() for a in v)
    parts += sorted(ROLES)
    parts += sorted(a for v in AXES.values() for a in v)
    parts += sorted(s for v in SCENES.values() for s in v)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]