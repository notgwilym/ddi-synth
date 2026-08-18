"""Generator v17: frames, slots, and no liftable clauses.

WHAT WENT WRONG IN THE FIRST v17 DRAFT, and why it is the same mistake as v13
-----------------------------------------------------------------------------
The draft rendered `say` as a finished English clause: "that Antipsychotics changes
urinary elimination of Minzasolmin, making steady-state levels lower". The model strips
"that", fixes agreement, and is done. Every MECHANISM spec then has one realisation and
the only variation is which nouns fill it. That is the v13 content brief, removed in v14
for exactly this reason.

Also: `regimen` was 15% of specs with a single fixed say line and a 25-word fixed avoid
line, so ~900 identical instructions in a 6,000-spec run. And realisation axes were
sampled without regard to frame, producing "start from the drug that causes the change"
on specs where nothing causes anything, and up to seven simultaneous constraints.

Fixed here by three rules:

  1. A spec never contains a clause. Content is rows of nouns; the model composes the
     syntax. Where a construction genuinely needs describing, it is described as a
     shape ("open with a name, then a colon, then a statement"), not as prose.
  2. Frames that need no content carry none. The system prompt states the no-assertion
     rule once; the spec then just omits the block. Scene, roles and register do the
     varying, which is what they are for.
  3. Axes are declared per frame and capped at two. An axis that cannot apply to a
     frame is not in its pool.

WHAT THE EVIDENCE SUPPORTS, kept unchanged
------------------------------------------
One sampler, one prompt, both labels in one call. v13 drew positives and negatives from
separate generators, so entity count predicted the label (hard negative rate 0.001
against a corpus 0.50, count-only rule 0.86). Fixing it moved 0.280 -> 0.379.

Nothing in a spec may be liftable. Every rendered English phrase has come back verbatim:
role glosses, "for a different condition" 60+ times, label names, field annotations,
even "these come after the main statement".

A field's PRESENCE must be label-independent. When the scene rendered only on specs with
spare drugs, scene vocabulary predicted the absence of an assertion. Independence is the
rule, not omnipresence, which is what makes optional axes safe.

Length drawn independently of label: without it, zero-assertion sentences ran ~6 words
longer and a bag-of-words probe read clause count.

No vague joint-outcome negatives. A verifier flagged 1,056 of v14's NONE pairs; removing
exactly those gained 0.041 F1 over removing the same number at random, and the gain was
recall (0.525 -> 0.623) with precision flat.

Not acted on, because unattributed: v15 bundled three changes and lost 0.046. Per-drug
roles stay at v14's rate, since lowering them cost 0.035 while leaving the artefact
unchanged.

GUIDELINES
----------
Priority order §4.5.11 (mechanism > effect > advise) makes the scope constraint
asymmetric: a MECHANISM sentence may also state a consequence and stays MECHANISM, while
an EFFECT sentence must not mention a PK change. Only the latter needs an avoid line.

New constructions the corpus contains and no previous generator produced: negation
(§4.5.1), studies with no outcome (§4.5.5), incompatibility (§4.5.6), enzyme
descriptions (§3.5.4), out-of-scope substances (§3.5.5), titles whose entity is not the
subject (§4.5.9), pharmacodynamic vocabulary (§4.1.4), protective interactions (§4.5.3),
advice with its reason (Fig 116), affirmation alongside denial (§4.5.4), and appositive
cross-products (§4.5.10, four relations from one clause).
"""
import hashlib
import itertools
import random

LABELS = ["ADVISE", "EFFECT", "INT", "MECHANISM"]


# ============================================================ system prompt
# Carries what is true of every request. The no-assertion rule lives here rather than in
# 900 identical spec lines, and the four statement kinds are defined once so a frame can
# just name its kind.

SYSTEM = """You write one sentence of biomedical text for a drug interaction corpus.

Each request lists drug names and, usually, a short block of facts to build the sentence
from. The facts are given as separate items, not as a finished phrase. Compose the
sentence yourself: choose the verb, the clause order and the connectives.

- Use every name listed, spelled exactly as given, and no other drug names.
- Every name must be doing something in the sentence.
- When a request has no "says" block, the sentence places the drugs in a situation and
  claims nothing at all: not that one affects another, and not that the combination as a
  whole did anything, worked, was tolerated, or changed an outcome. Any outcome you
  mention belongs to one drug alone or to the patient's condition.
- Follow any "avoid" line strictly. A sentence that breaks it cannot be used.
- Where the request does not constrain something, vary it. Do not open every sentence
  the same way and do not reuse the same sentence shape.
- The drugs listed may not be sensible together in real practice. Write it as asked.
- The field names in the request are instructions to you, not words for the sentence.
- Match verb agreement to the name; some names are plural.
- One sentence. Plain prose, no markdown."""


# ============================================================ content pools
# Nouns and noun phrases. Nothing here contains a verb that could be lifted.

PK_SITE = {
    "hepatic": ["hepatic metabolism", "CYP-mediated metabolism", "oxidative metabolism",
                "hepatic clearance", "first-pass metabolism", "liver enzyme activity",
                "presystemic metabolism", "demethylation"],
    "absorption": ["gastrointestinal absorption", "absorption from the gut",
                   "oral absorption", "oral bioavailability", "uptake from the small bowel",
                   "the extent of absorption", "the rate of absorption"],
    "renal": ["renal clearance", "excretion by the kidney", "tubular secretion",
              "urinary elimination", "renal excretion", "glomerular filtration",
              "tubular reabsorption"],
    "binding": ["plasma protein binding", "binding to serum proteins",
                "the unbound fraction", "displacement from albumin",
                "the free drug fraction"],
    "transport": ["intestinal transport", "efflux transport",
                  "carrier-mediated transport", "active transport across the gut wall",
                  "biliary transport"],
    "gastric": ["gastric pH", "gastric acid secretion", "gastric emptying",
                "the acidity of the stomach"],
    "distribution": ["the volume of distribution", "tissue distribution",
                     "distribution into the central compartment"],
}
PK_PARAM = ["plasma concentrations", "serum levels", "the AUC", "peak concentration",
            "the elimination half-life", "steady-state levels", "systemic exposure",
            "trough concentrations", "the maximum plasma concentration",
            "blood levels", "circulating levels", "total exposure", "Cmax",
            "the area under the curve", "plasma levels"]
PK_DIRECTION = ["increased", "reduced", "raised", "lowered", "elevated", "diminished"]
PK_MAGNITUDE = ["marked", "modest", "slight", "substantial", "considerable",
                "appreciable", "pronounced", "small but consistent"]

HARM = {
    "bleeding": ["bleeding", "haemorrhage", "bleeding risk", "prolonged bleeding time",
                 "bruising"],
    "sedation": ["sedation", "drowsiness", "central nervous system depression",
                 "somnolence", "impaired alertness"],
    "hypotension": ["hypotension", "a fall in blood pressure", "postural hypotension",
                    "dizziness on standing"],
    "arrhythmia": ["arrhythmia", "irregular heart rhythm", "conduction disturbances",
                   "bradycardia", "atrioventricular block"],
    "renal": ["nephrotoxicity", "impaired renal function", "raised serum creatinine",
              "acute kidney injury", "reduced urine output"],
    "hepatic": ["hepatotoxicity", "raised liver enzymes", "liver injury",
                "elevated transaminases", "jaundice"],
    "glycaemic": ["hypoglycaemia", "low blood glucose", "impaired glucose control",
                  "hyperglycaemia"],
    "muscle": ["myopathy", "muscle damage", "rhabdomyolysis", "muscle pain",
               "raised creatine kinase"],
    "neuro": ["seizures", "convulsions", "a lowered seizure threshold", "tremor",
              "confusion", "extrapyramidal symptoms"],
    "respiratory": ["respiratory depression", "slowed breathing", "apnoea"],
    "cardiac": ["QT prolongation", "prolongation of the QT interval",
                "torsades de pointes", "ECG changes"],
    "derm": ["rash", "photosensitivity", "skin reactions", "urticaria",
             "Stevens-Johnson syndrome"],
    "haem": ["neutropenia", "thrombocytopenia", "marrow suppression", "anaemia",
             "agranulocytosis"],
    "gi": ["nausea", "gastrointestinal irritation", "ulceration", "diarrhoea",
           "gastric bleeding"],
    "electrolyte": ["hypokalaemia", "hyponatraemia", "electrolyte disturbance",
                    "hyperkalaemia"],
}
HARM_EXTENT = ["greater", "new", "more frequent", "more severe", "increased",
               "an added"]

PD_RELATION = {
    "synergism": ["synergism", "a synergistic effect", "synergistic activity"],
    "additive": ["an additive effect", "additive toxicity", "additive depression"],
    "potentiation": ["potentiation", "an enhanced response", "enhancement of the effect"],
    "antagonism": ["antagonism", "an opposing action", "a blunted response",
                   "reduced activity"],
}

PROTECTIVE = ["a protective effect", "reduced toxicity", "less severe injury",
              "a lower incidence of adverse effects", "partial protection",
              "attenuation of the damage", "a reduced rate of complications"]

EFFICACY_LOSS = ["therapeutic failure", "loss of therapeutic effect",
                 "reduced clinical efficacy", "an inadequate response",
                 "loss of control of the condition", "reduced effectiveness",
                 "diminished benefit"]

ADVICE = {
    "avoid": ["avoidance of the combination", "contraindication",
              "not using the two together"],
    "dose": ["dose reduction", "a lower starting dose", "dose adjustment",
             "halving the usual dose", "titration from a lower dose"],
    "monitor": ["closer monitoring", "monitoring of blood levels",
                "clinical observation", "checking renal function",
                "monitoring of the ECG", "regular blood counts"],
    "timing": ["separation of dosing times", "staggered administration",
               "an interval of several hours between doses"],
    "stop": ["withdrawal of one agent", "discontinuation before the other is started",
             "stopping treatment beforehand"],
}

DENIAL_OBJECT = ["the pharmacokinetics", "plasma concentrations", "clearance",
                 "the observed effect", "steady-state levels", "the response",
                 "the AUC", "protein binding", "the half-life", "absorption"]
DENIAL_STRENGTH = ["no significant change", "no detectable change", "no alteration",
                   "no measurable difference", "no clinically relevant change",
                   "no effect"]

STUDY_SETTING = ["in vitro", "in healthy volunteers", "in an open-label study",
                 "in a crossover study", "in isolated tissue", "in a small cohort",
                 "in a single-dose study", "in an animal model", "in vivo",
                 "in a pharmacokinetic study"]
STUDY_VERB_NOUN = ["investigation", "assessment", "examination", "evaluation",
                   "characterisation"]

INCOMPAT = ["precipitation in the infusion bag", "a visible change in the solution",
            "chemical instability when mixed", "clouding of the prepared solution",
            "a colour change on mixing", "formation of a precipitate",
            "loss of potency in the syringe"]

ENZYME = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2E1",
          "P-glycoprotein", "UGT1A1", "OATP1B1"]
ENZYME_ROLE = ["a potent inhibitor", "a substrate", "an inducer", "a weak inhibitor",
               "a moderate inhibitor", "a competitive inhibitor"]

OUT_OF_SCOPE = ["grapefruit juice", "St John's wort", "a high-fat meal", "green tea",
                "cranberry juice", "milk", "a protein-rich meal", "tobacco smoking",
                "liquorice", "ginkgo biloba"]

FIGURE = ["about 30%", "roughly twofold", "by 80%", "approximately 25%", "threefold",
          "by 45%", "about 60%", "from 0.9 to 1.4 mg/L", "by a factor of four",
          "by around 15%", "some 70%", "more than double"]

# Opaque codes. v14's values and rate: lowering P_ROLE to 0.2 cost 0.035 F1 while
# leaving the artefact ratio unchanged, so they earn their place.
ROLES = {
    "r1": "was given earlier and has stopped",
    "r2": "started after the others ended",
    "r3": "a comparator",
    "r4": "used if the first choice is unsuitable",
    "r5": "not given, or not permitted",
    "r6": "for a condition unrelated to the others",
    "r7": "part of the background regimen",
}
P_ROLE = 0.55

REGISTER_LINE = {
    "DrugBank": ["product label prose: impersonal, present tense",
                 "drug label prose: no patients as subjects, present tense",
                 "prescribing information: impersonal, generalising"],
    "MedLine": ["research abstract prose: past tense, patients or volunteers",
                "abstract findings prose: past tense, hedged where appropriate",
                "clinical study report prose: past tense"],
}
REGISTERS = {"DrugBank": 0.85, "MedLine": 0.15}

# Rendered on a fixed proportion regardless of frame, so presence carries no label
# information. Some specs get no scene at all: the definition of the sentence type is
# enough, and leaving the context out produces sentences the scene pool would never
# license.
P_SCENE = 0.6
SCENES = {
    "DrugBank": ["used in the same area of treatment",
                 "options for the same indication",
                 "part of the same treatment pathway",
                 "listed together on the same label",
                 "prescribed for different conditions in the same patient",
                 "available in the same section of the formulary",
                 "named in the same warnings section",
                 "alternatives within one therapeutic class",
                 "supplied together in a treatment pack"],
    "MedLine": ["recorded in the same patient's notes",
                "given at different points in the patient's care",
                "the treatments this patient had received",
                "the options available for the same indication",
                "listed in the study's inclusion criteria",
                "documented in the same case series",
                "the agents this cohort was taking at baseline",
                "reported in the same adverse event registry",
                "prescribed across the study population"],
}


# ============================================================ realisation axes
# Declared per frame. An axis a frame cannot honour is not in its pool, which is what
# produced "start from the drug that causes the change" on specs where nothing causes
# anything. Capped at two per spec: seven simultaneous constraints is more than a small
# model holds, and each one is another chance to leave a seam.

MAX_AXES = 2

AXIS_POOL = {
    "grammar": ["active voice", "passive voice",
                "begin from the act of giving them together",
                "put the main point in a subordinate clause",
                "put the main point in a relative clause",
                "phrase it as a condition, using when or if",
                "use a participle clause",
                "lead with a prepositional phrase"],
    "orientation": ["start from the drug that acts",
                    "start from the drug that is acted on",
                    "start from neither drug"],
    "separation": ["keep the two drug names close together",
                   "put the indication being treated between the two drug names",
                   "put the dose or the route between the two drug names",
                   "put the study population between the two drug names",
                   "put the timing of administration between the two drug names"],
    "length": ["a single clause", "one clause with a subordinate clause",
               "two or three clauses"],
    "opening": ["with a drug name", "with a condition or setting",
                "with a subordinate clause", "with a quantity or a proportion",
                "with a time or sequence word", "with the finding"],
    "detail": ["include a sample size", "include a percentage",
               "name the patient population", "name the route of administration",
               "give the duration of treatment"],
    "certainty": ["state it as something that may happen", "state it as reported",
                  "state it as a single observed case",
                  "state it as a theoretical concern", "state it as established",
                  "state it as suggested by the data"],
}

# §4.5.2: interactions are annotated regardless of certainty, so hedging must appear on
# positives. It must appear on negatives at the same rate too, or hedging becomes a
# negative-predicting cue. Hence "certainty" is in almost every frame's pool.
AXES_FOR_KIND = {
    "none_plain":   ["grammar", "length", "opening", "detail", "certainty"],
    "none_pair":    ["grammar", "orientation", "separation", "length", "opening",
                     "detail", "certainty"],
    "none_single":  ["grammar", "length", "opening", "detail", "certainty"],
    "positive":     ["grammar", "orientation", "separation", "length", "opening",
                     "detail", "certainty"],
    "structural":   ["length", "detail", "certainty"],
}

REGISTER_BAN = {
    "DrugBank": {"detail": {"include a sample size", "name the patient population"},
                 "certainty": {"state it as a single observed case"},
                 "opening": {"with the finding"}},
    "MedLine": {},
}


def _pick(dist, rng):
    ks = list(dist)
    return rng.choices(ks, weights=[dist[k] for k in ks], k=1)[0]


def _alias(pool, rng):
    k = rng.choice(list(pool))
    return k, rng.choice(pool[k])


# ============================================================ frames
# Each returns rows of nouns under `says`, never a clause. `shape` is used only where a
# construction genuinely needs describing, and is phrased as an arrangement rather than
# as prose, so it cannot be lifted into the sentence.

def _f_regimen(ents, rng):
    """N1. No says block at all: the system prompt covers it, and 900 copies of a fixed
    instruction was the bulk of the diversity problem in the first draft."""
    return {"kind": "none_plain", "says": [], "positives": [], "focus": []}


def _f_denial(ents, rng):
    """N2, §4.5.1."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "none_pair",
            "says": [("about", f"{{{a}}} and {{{b}}}"),
                     ("what is unchanged", rng.choice(DENIAL_OBJECT)),
                     ("how much", rng.choice(DENIAL_STRENGTH))],
            "avoid": ["hedging the denial, or suggesting an effect might still occur"],
            "positives": [], "focus": [a, b]}


def _f_study_only(ents, rng):
    """N3, §4.5.5. The avoid line is the whole point of the frame."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "none_pair",
            "says": [("what was looked at", f"the interaction of {{{a}}} and {{{b}}}"),
                     ("kind of work", rng.choice(STUDY_VERB_NOUN)),
                     ("setting", rng.choice(STUDY_SETTING))],
            "avoid": ["reporting what was found, in any form"],
            "positives": [], "focus": [a, b]}


def _f_incompatible(ents, rng):
    """N8, §4.5.6."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "none_pair",
            "says": [("what is mixed", f"{{{a}}} and {{{b}}}, before administration"),
                     ("what is seen", rng.choice(INCOMPAT))],
            "avoid": ["saying anything about what happens inside the body"],
            "positives": [], "focus": [a, b]}


def _f_enzyme_only(ents, rng):
    """N11, §3.5.4. Enzymes are not entities, so this yields no relation."""
    a = rng.choice([e["key"] for e in ents])
    return {"kind": "none_single",
            "says": [("subject", f"{{{a}}}"),
                     ("what it is", rng.choice(ENZYME_ROLE)),
                     ("of what", rng.choice(ENZYME))],
            "avoid": ["naming any other listed drug as affected by this"],
            "positives": [], "focus": [a]}


def _f_out_of_scope(ents, rng):
    """N10, §3.5.5. Food, drink and herbal preparations are not entities."""
    a = rng.choice([e["key"] for e in ents])
    site = _alias(PK_SITE, rng)[1]
    return {"kind": "none_single",
            "says": [("what acts", rng.choice(OUT_OF_SCOPE)),
                     ("what is affected", f"{{{a}}}"),
                     ("what changes", site)],
            "avoid": ["saying that any of the listed drugs affects another"],
            "positives": [], "focus": [a]}


def _f_sequential(ents, rng):
    """N7. Temporally separated, so co-administration never happened."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "none_pair",
            "says": [("stopped first", f"{{{a}}}"),
                     ("started after", f"{{{b}}}"),
                     ("gap", rng.choice(["several days", "two weeks", "a washout period",
                                         "the following course"]))],
            "avoid": ["saying that either changed the effect of the other"],
            "positives": [], "focus": [a, b]}


def _f_title(ents, rng):
    """N12 plus a positive, §4.5.9. The heading entity does not participate."""
    keys = [e["key"] for e in ents]
    head, rest = keys[0], keys[1:]
    a, b = rng.sample(rest, 2)
    site = _alias(PK_SITE, rng)[1]
    return {"kind": "structural",
            "shape": f"open with {{{head}}}, then a colon, then a complete statement "
                     f"that does not involve it",
            "says": [("acts", f"{{{a}}}"), ("acted on", f"{{{b}}}"),
                     ("what changes", site),
                     ("which way", rng.choice(PK_DIRECTION)),
                     ("measured as", rng.choice(PK_PARAM))],
            "positives": [(a, b, "MECHANISM")], "focus": [head, a, b]}


def _f_coordinate(ents, rng):
    """N5. Joint agents are related to their target, not to each other."""
    a, b, c = rng.sample([e["key"] for e in ents], 3)
    return {"kind": "structural",
            "shape": f"{{{a}}} and {{{b}}} together as one subject, acting on {{{c}}}",
            "says": [("what follows", _alias(HARM, rng)[1]),
                     ("extent", rng.choice(HARM_EXTENT))],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, c, "EFFECT"), (b, c, "EFFECT")], "focus": [a, b, c]}


def _f_mechanism(ents, rng):
    """P1/P2, §4.2. Priority order means a consequence would not change the label, so no
    avoid line."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    rows = [("acts", f"{{{a}}}"), ("acted on", f"{{{b}}}"),
            ("what changes", _alias(PK_SITE, rng)[1]),
            ("which way", rng.choice(PK_DIRECTION)),
            ("measured as", rng.choice(PK_PARAM))]
    if rng.random() < 0.35:
        rows.append(("size of change", rng.choice(PK_MAGNITUDE)))
    if rng.random() < 0.20:
        rows.append(("figure", rng.choice(FIGURE)))
    return {"kind": "positive", "says": rows,
            "positives": [(a, b, "MECHANISM")], "focus": [a, b]}


def _f_mechanism_mixed(ents, rng):
    """P3, §4.5.11. PK change plus a consequence; gold stays MECHANISM by priority.
    Never generated before v17, and MECHANISM has the worst precision of any class in
    both v14 (0.23) and v15 (0.20)."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("acts", f"{{{a}}}"), ("acted on", f"{{{b}}}"),
                     ("what changes", _alias(PK_SITE, rng)[1]),
                     ("which way", rng.choice(PK_DIRECTION)),
                     ("measured as", rng.choice(PK_PARAM)),
                     ("and then a risk of", _alias(HARM, rng)[1])],
            "positives": [(a, b, "MECHANISM")], "focus": [a, b]}


def _f_effect(ents, rng):
    """P4/P6, §4.1. Must not mention a PK change or priority order makes it mechanism."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("taken together", f"{{{a}}} and {{{b}}}"),
                     ("what follows", _alias(HARM, rng)[1]),
                     ("extent", rng.choice(HARM_EXTENT))],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, b, "EFFECT")], "focus": [a, b]}


def _f_effect_pd(ents, rng):
    """P7, §4.1.4. This vocabulary existed in no previous generator."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("between", f"{{{a}}} and {{{b}}}"),
                     ("relation", _alias(PD_RELATION, rng)[1])],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, b, "EFFECT")], "focus": [a, b]}


def _f_effect_protective(ents, rng):
    """P8, §4.5.3. Beneficial interactions are annotated. Never generated before."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("protects", f"{{{a}}}"), ("against harm from", f"{{{b}}}"),
                     ("the harm", _alias(HARM, rng)[1]),
                     ("degree", rng.choice(PROTECTIVE))],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, b, "EFFECT")], "focus": [a, b]}


def _f_effect_failure(ents, rng):
    """P9, §4.1."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("acts", f"{{{a}}}"), ("acted on", f"{{{b}}}"),
                     ("what follows", rng.choice(EFFICACY_LOSS))],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, b, "EFFECT")], "focus": [a, b]}


def _f_advise(ents, rng):
    """P10, §4.3."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("about", f"{{{a}}} with {{{b}}}"),
                     ("what to do", _alias(ADVICE, rng)[1])],
            "avoid": ["giving any reason, and mentioning levels, concentrations or any "
                      "consequence for the patient"],
            "positives": [(a, b, "ADVISE")], "focus": [a, b]}


def _f_advise_reason(ents, rng):
    """P11, Fig 116. Gold stays ADVISE: the consequence explains the advice rather than
    describing the interaction."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("about", f"{{{a}}} with {{{b}}}"),
                     ("what to do", _alias(ADVICE, rng)[1]),
                     ("reason given", f"the possibility of {_alias(HARM, rng)[1]}")],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, b, "ADVISE")], "focus": [a, b]}


def _f_int(ents, rng):
    """P12, §4.4."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "positive",
            "says": [("between", f"{{{a}}} and {{{b}}}"),
                     ("all that is known", "that they interact")],
            "avoid": ["saying anything about how, what follows from it, or what to do "
                      "about it"],
            "positives": [(a, b, "INT")], "focus": [a, b]}


def _f_contradictory(ents, rng):
    """P14, §4.5.4. The affirmation is annotated."""
    a, b = rng.sample([e["key"] for e in ents], 2)
    return {"kind": "structural",
            "shape": "a denial first, then an affirmation that overrides it",
            "says": [("not reported for", f"{{{a}}}"),
                     ("but still applies with", f"{{{b}}}"),
                     ("what to do", _alias(ADVICE, rng)[1])],
            "positives": [(a, b, "ADVISE")], "focus": [a, b]}


def _f_appositive(ents, rng):
    """P15, §4.5.10. Four relations from one clause: the single largest construction gap
    in every previous generator."""
    a, b, c, d = rng.sample([e["key"] for e in ents], 4)
    marker = rng.choice(["including", "such as"])
    return {"kind": "structural",
            "shape": f"{{{a}}}, {marker} {{{b}}}, acting on {{{c}}}, {marker} {{{d}}}",
            "says": [("what follows", _alias(HARM, rng)[1]),
                     ("extent", rng.choice(HARM_EXTENT))],
            "avoid": ["mentioning levels, concentrations, absorption, metabolism or "
                      "clearance"],
            "positives": [(a, c, "EFFECT"), (a, d, "EFFECT"),
                          (b, c, "EFFECT"), (b, d, "EFFECT")],
            "focus": [a, b, c, d]}


FRAMES = {
    # negatives ~45%, matching the corpus share of multi-entity sentences with no
    # asserted relation (0.448 on the filtered training split)
    "regimen":        {"n": 2, "w": 0.16, "f": _f_regimen},
    "denial":         {"n": 2, "w": 0.11, "f": _f_denial},
    "study_only":     {"n": 2, "w": 0.00, "f": _f_study_only},
    "incompatible":   {"n": 2, "w": 0.03, "f": _f_incompatible},
    "enzyme_only":    {"n": 2, "w": 0.05, "f": _f_enzyme_only},
    "out_of_scope":   {"n": 2, "w": 0.03, "f": _f_out_of_scope},
    "sequential":     {"n": 2, "w": 0.05, "f": _f_sequential},
    # positives
    "mechanism":       {"n": 2, "w": 0.13, "f": _f_mechanism},
    "mechanism_mixed": {"n": 2, "w": 0.06, "f": _f_mechanism_mixed},
    "effect":          {"n": 2, "w": 0.10, "f": _f_effect},
    "effect_pd":       {"n": 2, "w": 0.04, "f": _f_effect_pd},
    "effect_protect":  {"n": 2, "w": 0.02, "f": _f_effect_protective},
    "effect_failure":  {"n": 2, "w": 0.04, "f": _f_effect_failure},
    "advise":          {"n": 2, "w": 0.07, "f": _f_advise},
    "advise_reason":   {"n": 2, "w": 0.04, "f": _f_advise_reason},
    "int":             {"n": 2, "w": 0.03, "f": _f_int},
    "contradictory":   {"n": 2, "w": 0.02, "f": _f_contradictory},
    "title":           {"n": 3, "w": 0.02, "f": _f_title},
    "coordinate":      {"n": 3, "w": 0.03, "f": _f_coordinate},
    "appositive":      {"n": 4, "w": 0.02, "f": _f_appositive},
}

N_ENTITIES = {2: 0.41, 3: 0.27, 4: 0.14, 5: 0.10, 6: 0.08}


def _unusable(name):
    return " and " in name.lower() or "/" in name


def _sample_axes(kind, register, rng):
    """One or two, from the pool this frame can honour. Presence is label-independent."""
    pool = AXES_FOR_KIND[kind]
    n = rng.choices([0, 1, 2], weights=[0.15, 0.45, 0.40], k=1)[0]
    out = {}
    for ax in rng.sample(pool, min(n, len(pool))):
        banned = REGISTER_BAN.get(register, {}).get(ax, set())
        opts = [o for o in AXIS_POOL[ax] if o not in banned]
        if opts:
            out[ax] = rng.choice(opts)
    return out


def make_specs(n, vocab, seed=0):
    """One spec per sentence. The frame decides the construction and the gold matrix; the
    label is never named to the model.

    vocab.sample(k, rng) returns k names, choosing groups with probability p_group. No
    typed sampling and no BRAND or DRUG_N in the vocabulary, so 14% of real entity types
    cannot be matched. Recorded rather than faked.
    """
    group_set = set(vocab.groups)
    rng = random.Random(seed)
    frame_w = {k: v["w"] for k, v in FRAMES.items()}
    specs = []

    for i in range(n):
        register = _pick(REGISTERS, rng)
        frame_name = _pick(frame_w, rng)
        fr = FRAMES[frame_name]
        k = max(fr["n"], _pick(N_ENTITIES, rng))

        surfaces, seen = [], set()
        for _ in range(200):
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
        keys = [e["key"] for e in ents]

        built = fr["f"](ents, rng)
        focus = set(built["focus"])

        spare = [x for x in keys if x not in focus]
        roled = [x for x in spare if rng.random() < P_ROLE][:2]
        roles, used = {}, set()
        for x in roled:
            avail = [r for r in ROLES if r not in used]
            if not avail:
                break
            r = rng.choice(avail)
            used.add(r)
            roles[x] = r

        matrix = {f"{a}|{b}": "NONE" for a, b in itertools.combinations(keys, 2)}
        for a, b, lab in built["positives"]:
            matrix[f"{min(a, b)}|{max(a, b)}"] = lab

        specs.append({
            "spec_index": i,
            "frame": frame_name,
            "kind": built["kind"],
            "register": register,
            "register_line": rng.choice(REGISTER_LINE[register]),
            "entities": ents,
            "shape": built.get("shape"),
            "says": built["says"],
            "avoid": built.get("avoid", []),
            "positives": [{"between": [a, b], "label": lab}
                          for a, b, lab in built["positives"]],
            "roles": roles,
            "scene": rng.choice(SCENES[register]) if rng.random() < P_SCENE else None,
            "axes": _sample_axes(built["kind"], register, rng),
            "matrix": matrix,
        })
    return specs


def render(spec):
    """Keys become surface forms. `says` renders as rows of nouns, never a clause: the
    model composes the syntax, which is where realisation diversity comes from."""
    by_key = {e["key"]: e["surface"] for e in spec["entities"]}

    def sub(t):
        for k, v in by_key.items():
            t = t.replace("{" + k + "}", v)
        return t

    lines = [spec["register_line"]]
    if spec["scene"]:
        lines.append(f"the drugs below are {spec['scene']}")
    lines.append("")

    lines.append(f"drug names, use all {len(by_key)}")
    lines += [f"  {e['surface']}" for e in spec["entities"]]
    lines.append("")

    if spec["shape"]:
        lines += ["arrangement", f"  {sub(spec['shape'])}", ""]

    if spec["says"]:
        lines.append("says")
        w = max(len(k) for k, _ in spec["says"])
        lines += [f"  {k.ljust(w)}  {sub(v)}" for k, v in spec["says"]]
        lines.append("")

    if spec["avoid"]:
        lines += ["avoid"] + [f"  {sub(s)}" for s in spec["avoid"]] + [""]

    if spec["roles"]:
        lines.append("the remaining drugs")
        lines += [f"  {by_key[k]}  {ROLES[r]}" for k, r in spec["roles"].items()]
        lines.append("")

    if spec["axes"]:
        lines += ["how to write it"] + [f"  {v}" for v in spec["axes"].values()] + [""]

    return "\n".join(lines).rstrip()


def make_sample_fn(client, model="gpt-oss-120b", temperature=0.9,
                   reasoning_effort="low", max_output_tokens=1500, api="responses"):
    """Returns {sentence}. The label lives in the spec; asking for it back would put
    label vocabulary into the prompt.

    Low effort: at medium the model spent 300-570 reasoning tokens per call, much of it
    counting words by hand, and at low it spends ~100 with no loss of quality.
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
        r = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": user}],
            text_format=Written, temperature=temperature, **kw)
        if r.output_parsed is None:
            raise ValueError(f"no parsed output (status={getattr(r, 'status', '?')})")
        return r.output_parsed.sentence

    def _chat(user):
        kw = {"max_tokens": max_output_tokens} if max_output_tokens else {}
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
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
    parts = [SYSTEM] + sorted(FRAMES)
    for pool in (PK_SITE, HARM, PD_RELATION, ADVICE):
        parts += sorted(a for v in pool.values() for a in v)
    for pool in (PK_PARAM, PK_DIRECTION, PROTECTIVE, EFFICACY_LOSS, DENIAL_OBJECT,
                 DENIAL_STRENGTH, STUDY_SETTING, INCOMPAT, ENZYME, OUT_OF_SCOPE, FIGURE):
        parts += sorted(pool)
    parts += sorted(a for v in AXIS_POOL.values() for a in v)
    parts += sorted(ROLES.values())
    parts += sorted(s for v in SCENES.values() for s in v)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


make_v14_specs = lambda n, vocab, seed=0, composition="prior": make_specs(n, vocab, seed)
render_v14 = render
make_v14_sample_fn = make_sample_fn
v14_fingerprint = fingerprint