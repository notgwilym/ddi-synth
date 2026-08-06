"""Generator v14.

The failure this replaces: v13 drew positives from the vignette generator (~1 pair per
sentence) and negatives from CONEG (6-15 pairs), so entity count predicted the label.
One-pair sentences were 75% positive, multi-pair ~100% negative, hard negative rate
0.001 against a corpus prose rate of ~0.50. The entity markers carried no signal at all
because pair identity was never a variable. Here every sentence comes from one sampler
and one prompt, and zero-assertion specs are drawn from the same distribution, so no
structural property can separate the classes.

Two things about the prompt that are easy to get wrong, and were:

The label NAME must not reach the model: v13 wrote "a clear MECHANISM interaction" into
sentences. The label CRITERIA must reach it, or a spec describing a change in exposure
can produce a sentence an annotator would read as a clinical effect, and the gold label
is then wrong. Those are separable, and the system prompt carries the criteria as kinds
of statement with worked examples on placeholder names.

Nothing in a spec is a phrase that can be lifted. Content is slot tokens with a rotated
alias pool per slot, so the model builds the syntax itself rather than rephrasing a
clause it was handed.

Every axis is filtered against register, scope, and whether the spec asserts anything at
all. An unfiltered axis produces specs the model cannot satisfy: a no-assertion spec
told to make "the affected drug" the subject, or a recommended-action spec told to make
plasma concentrations the subject. The model then violates one instruction or the other
and which one is unpredictable.
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
    "n_positives": {0: 0.45, 1: 0.40, 2: 0.15},
    "labels": {l: 0.25 for l in LABELS},
    "registers": {"DrugBank": 0.5, "MedLine": 0.5},
}
MEASURED = {
    "n_entities": {2: 0.410, 3: 0.272, 4: 0.137, 5: 0.073, 6: 0.040},
    "n_positives": {0: 0.448, 1: 0.375, 2: 0.086},
    "labels": {"EFFECT": 0.440, "MECHANISM": 0.289, "ADVISE": 0.210, "INT": 0.061},
    "registers": {"DrugBank": 0.85, "MedLine": 0.15},
}
COMPOSITION = {"prior": PRIOR, "measured": MEASURED}


SYSTEM = """You write single sentences of biomedical text for a fictional drug interaction dataset. 

Each request lists drug names, the content the sentence carries, and a style. Write
exactly one sentence.

WHAT THE SENTENCE ASSERTS

A request either asks you to assert something between two named drugs, or asks for no
assertion at all. When it does, the "scope" line says which kind of statement to write.
The four kinds are exclusive. Write the one asked for and none of the others: a sentence
that mixes them cannot be used.

1. Exposure and handling / mechanism. How much of one drug is present, or how the body takes it up,
   distributes it, breaks it down or removes it, changed by the presence of the other.
   Say what changes and in which direction. Do not say what happens to the patient as a
   result.

2. Clinical consequence / effect.  What follows for the patient when the two are combined: an
   effect, a risk, a loss of efficacy. Say what the consequence is. Do not explain the
   handling or the levels that produce it.

3. Recommended action / advice. What a prescriber should do about the combination. Say what
   should be done. Do not state the consequence or the mechanism motivating it.

4. Bare interaction. That the two interact, and nothing further. No mechanism, no
   consequence, no advice, no figures.

Only the pair named in the statement block is being asserted about. Other drugs in the
list are present in the sentence but nothing is claimed between them, or between them
and the pair.

WHEN NO ASSERTION IS ASKED FOR

Some requests have no statement block. Write ordinary clinical or research prose that
places every listed drug in the situation described at the top: a regimen, a comparison,
a sequence of treatments. Nothing in the sentence says that any drug affects any other.

RULES

- Use every drug name listed, and no others.
- Reproduce each name as given, including any punctuation or non-ASCII characters in it.
- Every drug you name must be doing something in the sentence.
- Do not worry about whether the drugs are actually used together in practice: the sentence is a fiction.
- The sentence should read as plausible clinical or research prose.
- Match verb agreement to the name; some names are plural.
- One sentence. No markdown."""


# The scope line is what stops a spec producing a sentence an annotator would read as a
# different class. It names the kind without naming the label.
SCOPE = {
    "MECHANISM": "exposure and handling only, no clinical consequence",
    "EFFECT": "clinical consequence only, no explanation of handling or levels",
    "ADVISE": "recommended action only, no consequence and no mechanism",
    "INT": "bare interaction only, no mechanism, consequence, advice or figures",
}


# ---------------------------------------------------------------- content slots
# Each slot is a token with a small alias pool; one alias is drawn per call, so the
# lexical surface rotates while the semantics hold. Aliases are noun phrases, never
# clauses: the model supplies every verb.

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

OUTCOMES = {
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
    "loss_of_efficacy": ["loss of therapeutic effect", "reduced efficacy"],
}
VALENCE = ["greater", "reduced", "new"]

ACTIONS = {
    "avoid": ["avoidance of the combination", "contraindication"],
    "dose_reduction": ["dose reduction", "a lower starting dose", "dose adjustment"],
    "monitoring": ["closer monitoring", "monitoring of blood levels",
                   "clinical observation"],
    "timing": ["separation of dosing times", "staggered administration"],
    "discontinuation": ["discontinuation before the other is started",
                        "withdrawal of one agent"],
}

# Non-participant roles. group records whether the role implies concurrent use.
# "separated" roles make an interaction semantically unavailable, so the label is safe
# by construction. "concurrent" roles are the realistic hard-negative case. Keeping only
# the separated ones would trade the composition shortcut for a lexical one.
ROLES = {
    "prior_therapy": ("separated", ["given earlier and since stopped",
                                    "the previous treatment"]),
    "subsequent": ("separated", ["started after the others were withdrawn",
                                 "the treatment that followed"]),
    "comparator": ("separated", ["evaluated in a separate arm",
                                 "the comparator agent"]),
    "alternative": ("separated", ["used instead when the first choice is unsuitable",
                                  "the substitute option"]),
    "not_enrolled": ("separated", ["excluded from the study population",
                                   "not permitted during the study"]),
    "concomitant": ("concurrent", ["part of the same regimen",
                                   "taken at the same time"]),
    "background": ("concurrent", ["treating an unrelated condition",
                                  "the patient's other ongoing medication"]),
}
ROLE_GROUP_DIST = {"separated": 0.45, "concurrent": 0.55}


# ---------------------------------------------------------------- realisation
# One or two axes per call. Six instructions is more than the model holds at once, and
# each is another chance to leave a visible seam.

AXES = {
    "subject": ["the agent drug", "the affected drug", "the combination",
                "plasma concentrations", "the patients", "the study"],
    "voice": ["active", "passive"],
    "certainty": ["established", "possible", "observed", "a single case"],
    "detail": ["no figures", "direction only", "one figure",
               "a pharmacokinetic figure"],
    "position": ["main clause", "subordinate clause", "relative clause"],
}
N_AXES = {1: 0.45, 2: 0.55}

# DrugBank product labels have no participants and no statistics; MedLine abstracts do.
REGISTER_BANS = {
    "DrugBank": {"subject": {"the patients", "the study"},
                 "certainty": {"a single case"},
                 "detail": {"a pharmacokinetic figure"}},
    "MedLine": {},
}

# Each scope forbids the axis values that contradict it. Without these the spec asks the
# model to write a recommendation whose subject is a plasma concentration, or a bare
# interaction with a stated direction.
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

# With no statement there is no agent, no affected, no clause to position and no detail
# to give. Only voice, certainty, and the subjects that exist without an assertion.
NO_ASSERT_AXES = ["voice", "certainty", "subject"]
NO_ASSERT_SUBJECTS = ["one of the drugs listed", "the treatment", "the regimen"]
NO_ASSERT_BANS = {"subject": {"the agent drug", "the affected drug",
                              "plasma concentrations", "the combination"}}

REGISTER_LINE = {
    "DrugBank": "product label prose: impersonal, present tense, no study participants,"
                " no sample sizes",
    "MedLine": "research abstract prose: past tense, participants or patients, a sample"
               " size or percentage where detail is called for, hedged conclusions",
}

SCENES = {
    "DrugBank": ["these are used in the same area of treatment",
                 "these are the options for the same indication",
                 "these appear in the same treatment pathway"],
    "MedLine": ["these were compared in the same study",
                "these were given at different points in the patient's care",
                "these were the treatments recorded for this patient",
                "these were the options available for the same indication"],
}

# other drug names falling between the interacting pair. Real positives have a median
# token gap of 10 and p90 of 57; separation is the controllable proxy, since in a
# multi-drug sentence you cannot set one pair's distance independently of the rest.
SEPARATION = {0: 0.40, 1: 0.35, 2: 0.25}


def _pick(dist, rng):
    ks = list(dist)
    return rng.choices(ks, weights=[dist[k] for k in ks], k=1)[0]


def _sample_style(register, label, rng):
    """Axes filtered against register, scope, and whether anything is asserted."""
    pool = list(AXES) if label else list(NO_ASSERT_AXES)
    chosen = rng.sample(pool, min(_pick(N_AXES, rng), len(pool)))
    style = {}
    for ax in chosen:
        banned = set(REGISTER_BANS.get(register, {}).get(ax, set()))
        banned |= set((LABEL_BANS[label] if label else NO_ASSERT_BANS).get(ax, set()))
        options = [o for o in (NO_ASSERT_SUBJECTS if (ax == "subject" and not label)
                               else AXES[ax]) if o not in banned]
        if options:
            style[ax] = rng.choice(options)
    return style


def _content(label, rng):
    """Symbolic content for one asserted interaction. Values are slot aliases, never
    clauses. INT carries only its scope, which the system prompt explains."""
    if label == "MECHANISM":
        site = rng.choice(list(SITES))
        return {"site": rng.choice(SITES[site]), "exposure": rng.choice(EXPOSURE),
                "_slots": {"site": site}}
    if label == "EFFECT":
        outcome = rng.choice(list(OUTCOMES))
        return {"outcome": rng.choice(OUTCOMES[outcome]),
                "extent": rng.choice(VALENCE), "_slots": {"outcome": outcome}}
    if label == "ADVISE":
        action = rng.choice(list(ACTIONS))
        return {"action": rng.choice(ACTIONS[action]), "_slots": {"action": action}}
    return {"_slots": {}}


def _assign_roles(roled, by_key, has_group, rng):
    """One role each, no repeated role text within a sentence: two drugs both described
    as 'treating an unrelated condition' reads as a template rather than a scene."""
    roles, used_role, used_text = {}, set(), set()
    for key in roled:
        grp = _pick(ROLE_GROUP_DIST, rng)
        candidates = [r for r, (g, _) in ROLES.items()
                      if g == grp and r not in used_role]
        # class membership only makes sense for a drug when a class is present, and
        # never for the class itself
        if not (has_group and by_key[key]["type"] == "DRUG"):
            candidates = [r for r in candidates if r != "class_member"]
        if not candidates:
            candidates = [r for r in ROLES if r not in used_role and r != "class_member"]
        if not candidates:
            break
        role = rng.choice(candidates)
        texts = [t for t in ROLES[role][1] if t not in used_text] or ROLES[role][1]
        text = rng.choice(texts)
        used_role.add(role)
        used_text.add(text)
        roles[key] = {"role": role, "group": ROLES[role][0], "text": text}
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
        k = _pick(cfg["n_entities"], rng)

        surfaces, seen = [], set()
        for _ in range(80):
            if len(surfaces) == k:
                break
            cand = vocab.sample(1, rng)[0]
            low = cand.lower()
            # a name containing another name breaks span resolution downstream
            if low in seen or any(low in s or s in low for s in seen):
                continue
            seen.add(low)
            surfaces.append(cand)
        if len(surfaces) < k:
            raise RuntimeError("vocab exhausted while sampling distinct names")

        ents = [{"key": chr(65 + j), "surface": s,
                 "type": "GROUP" if s in group_set else "DRUG"}
                for j, s in enumerate(surfaces)]
        keys = [e["key"] for e in ents]
        by_key = {e["key"]: e for e in ents}
        pairs = list(itertools.combinations(keys, 2))

        max_pos = min(len(pairs), 2)
        n_pos = _pick({p: w for p, w in cfg["n_positives"].items() if p <= max_pos}, rng)

        positives = []
        if n_pos >= 1:
            positives.append(rng.choice(pairs))
        if n_pos >= 2:
            hub = rng.choice(positives[0])
            others = [p for p in pairs if hub in p and p not in positives]
            if others:
                positives.append(rng.choice(others))

        # two asserts share a label and their content slot: they come off one hub in one
        # sentence, so a single scope and style block is honest rather than a compromise
        label = _pick(cfg["labels"], rng) if positives else None
        content = _content(label, rng) if positives else None

        asserts = []
        for a, b in positives:
            sep = None
            if k >= 3:
                sep = _pick({s: w for s, w in SEPARATION.items() if s <= k - 2}, rng)
            asserts.append({"between": [a, b], "label": label,
                            "content": content, "separation": sep})

        style = _sample_style(register, label, rng)

        participating = {x for p in positives for x in p}
        non_participants = [e["key"] for e in ents if e["key"] not in participating]
        roled = rng.sample(non_participants, min(len(non_participants), 2))
        has_group = any(e["type"] == "GROUP" for e in ents)
        roles = _assign_roles(roled, by_key, has_group, rng)

        scene = rng.choice(SCENES[register]) if non_participants else None

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
            "scene": scene,
            "matrix": matrix,
        })
    return specs


def render_v14(spec):
    """Style renders last: the model should know what it is styling before it is told
    how. Content, then the other drugs, then style."""
    by_key = {e["key"]: e for e in spec["entities"]}
    blocks = [("drugs", [(e["surface"],
                          "drug class" if e["type"] == "GROUP" else "drug")
                         for e in spec["entities"]])]

    for n, asrt in enumerate(spec["asserts"]):
        a, b = asrt["between"]
        rows = [("scope", SCOPE[asrt["label"]]),
                ("agent", by_key[a]["surface"]),
                ("affected", by_key[b]["surface"])]
        rows += [(kk, vv) for kk, vv in asrt["content"].items() if kk != "_slots"]
        if asrt["separation"] is not None:
            rows.append(("other drug names between them", str(asrt["separation"])))
        head = "statement" if len(spec["asserts"]) == 1 else f"statement {n + 1}"
        blocks.append((head, rows))

    if spec["roles"]:
        blocks.append(("other drugs",
                       [(by_key[k]["surface"], v["text"])
                        for k, v in spec["roles"].items()]))
    if spec["style"]:
        blocks.append(("style", list(spec["style"].items())))

    lines = [REGISTER_LINE[spec["register"]]]
    if spec["scene"]:
        lines.append(spec["scene"])
    lines.append("")
    for head, rows in blocks:
        lines.append(head)
        width = max((len(k) for k, _ in rows), default=0)
        lines += [f"  {k.ljust(width)}  {v}" for k, v in rows]
        lines.append("")
    return "\n".join(lines).rstrip()


def make_v14_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                       reasoning_effort="high", max_output_tokens=4000,
                       api="responses"):
    """Returns {sentence}. Nothing else: the label lives in the spec, and asking for it
    back would put label vocabulary into the prompt.

    api="chat" falls back to chat.completions, which every vLLM worker serves. The
    Responses route is worker-specific and 404s on some models.
    """
    from pydantic import BaseModel

    class Written(BaseModel):
        sentence: str

    def _responses(user):
        kw = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
        resp = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Written, temperature=temperature,
            max_output_tokens=max_output_tokens, **kw)
        if resp.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(resp, 'status', '?')})")
        return resp.output_parsed.sentence

    def _chat(user):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "written", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "required": ["sentence"],
                           "properties": {"sentence": {"type": "string"}}}}})
        return Written.model_validate_json(resp.choices[0].message.content).sentence

    def sample_fn(spec):
        user = render_v14(spec)
        return {"sentence": (_responses if api == "responses" else _chat)(user)}

    return sample_fn


def v14_fingerprint():
    parts = [SYSTEM] + sorted(SCOPE.values())
    parts += sorted(a for v in SITES.values() for a in v)
    parts += sorted(a for v in OUTCOMES.values() for a in v)
    parts += sorted(a for v in ACTIONS.values() for a in v)
    parts += sorted(a for _, v in ROLES.values() for a in v)
    parts += sorted(a for v in AXES.values() for a in v)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]