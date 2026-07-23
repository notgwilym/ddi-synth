"""Drug vocabulary for synthetic generation.

Sources are deliberately EXTERNAL to DDI-2013 (terms.json + a curated class list),
so generation never leans on the annotated corpus the project claims not to need.
"""
import json, hashlib, re
from pathlib import Path

# ---------------------------------------------------------------- drug groups
# Class-level terms ("group" entities in DDI-2013, guidelines S3.4). terms.json is
# product/chemical-derived and thin on these, but they are a large slice of real
# DDI entities and behave differently (plural, class-level, often the hub of a
# one-to-many relation). Written from general pharmacological knowledge, not from
# the corpus. Note: no administration routes ("oral contraceptives" -> per S3.5.3
# the route is not part of the entity, so "contraceptives").
DRUG_GROUPS = [
    "NSAIDs", "nonsteroidal anti-inflammatory drugs", "salicylates", "corticosteroids",
    "MAO inhibitors", "monoamine oxidase inhibitors", "SSRIs",
    "selective serotonin reuptake inhibitors", "tricyclic antidepressants",
    "antidepressants", "phenothiazines", "antipsychotics", "atypical antipsychotics",
    "benzodiazepines", "barbiturates", "CNS depressants", "sedatives", "hypnotics",
    "anticonvulsants", "antiepileptics", "muscle relaxants", "opioid analgesics",
    "narcotic analgesics", "anesthetics", "antihistamines", "anticholinergics",
    "sympathomimetics", "beta-blockers", "beta-adrenergic blocking agents",
    "beta-agonists", "alpha-blockers", "calcium channel blockers", "ACE inhibitors",
    "angiotensin converting enzyme inhibitors", "antihypertensive drugs", "nitrates",
    "cardiac glycosides", "digitalis glycosides", "antiarrhythmics", "diuretics",
    "thiazide diuretics", "loop diuretics", "potassium-sparing diuretics",
    "anticoagulants", "vitamin K antagonists", "antiplatelet agents", "statins",
    "HMG-CoA reductase inhibitors", "aminoglycosides", "macrolide antibiotics",
    "tetracyclines", "quinolones", "fluoroquinolones", "cephalosporins", "penicillins",
    "sulfonamides", "antibiotics", "broad-spectrum antibiotics", "azole antifungals",
    "antifungals", "protease inhibitors", "antiretrovirals",
    "nucleoside reverse transcriptase inhibitors", "TNF blocking agents",
    "TNF antagonists", "immunosuppressants", "antineoplastic agents",
    "chemotherapeutic agents", "cytotoxic agents", "sulfonylureas",
    "hypoglycemic agents", "biguanides", "thiazolidinediones", "antacids",
    "proton pump inhibitors", "H2 antagonists", "laxatives", "antiemetics",
    "bronchodilators", "methylxanthines", "xanthines", "uricosurics",
    "bisphosphonates", "retinoids", "estrogens", "progestins", "androgens",
    "contraceptives", "thyroid hormones", "vitamins", "fat-soluble vitamins",
]

# junk filtering

# terms.json is product-derived: consumer goods, multi-ingredient listings and
# formulation strings that are not drug entities in any useful sense.
_PRODUCT_WORDS = {
    "sanitizer", "sanitiser", "wipe", "wipes", "lotion", "shampoo", "soap",
    "sunscreen", "spf", "cleanser", "moisturizer", "moisturiser", "toothpaste",
    "mouthwash", "deodorant", "antiperspirant", "conditioner", "scrub", "balm",
    "serum", "mask", "pad", "pads", "swab", "swabs", "kit", "bandage",
    "rinse", "foam", "powder", "wash", "hand", "body", "facial", "baby",
    "daily", "relief", "strength", "flavor", "flavour", "spray", "gel",
    "cream", "ointment", "tablet", "tablets", "capsule", "capsules", "solution",
    "suspension", "syrup", "drops", "patch", "injection", "topical", "oral",
}
_HAS_DIGIT_PCT = re.compile(r"\d\s*%|\bmg\b|\bml\b|\bmcg\b|\bunits?\b", re.I)
_CHEMY = re.compile(r"[0-9]|[\[\]\(\),]")     # marks IUPAC-ish systematic names

# terms.json is a chemistry synonym dump, so it also carries database identifiers,
# encoding corruption and systematic-chemistry morphology. Catchable by pattern:
_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-?[A-Z]?$")   # incl. truncated forms
_UNII     = re.compile(r"^(?=.*\d)[A-Z0-9]{10}$")
_UNII_PRE = re.compile(r"^UNII[-\s]", re.I)
_DB_ID    = re.compile(r"^(CHEMBL|CAS-|DTXSID|HMDB|ZINC|NSC)\d*", re.I)
_FORMULA  = re.compile(r"^(?:[A-Z][a-z]?\d{0,3}){2,}$")     # C21H40ClNO
_CAS_IN   = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_REF_STD  = re.compile(r"\b(CRS|EP Impurity|for system suitability|\(TN\))\b")
_DOTGREEK = re.compile(r"\.(alpha|beta|gamma|delta|omega)\.")
_PEPTIDE  = re.compile(r"^(Ac|h|H|Boc|Fmoc)-[A-Za-z]{3}")
_DYE      = re.compile(r"\b(blue|red|violet|scarlet|magenta|yellow|green|orange|lake|vat|"
                       r"brilliant|fast|disperse|pigment)\b", re.I)
_SHORTCODE = re.compile(r"^[A-Z0-9]{1,5}[+\-]?$")
_ATC      = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")
_CAS      = re.compile(r"^\d{2,7}-\d{2}-\d$")
_NOTATION = re.compile(r"^(WLN|SMILES|InChI)\s*[:=]", re.I)
_MOJIBAKE = re.compile(r"[^\x20-\x7E]|\\\\|<<|>>|inverted exclamation|masculine")
_ARTICLE  = re.compile(r"^(a|an|the)\s+", re.I)
# trailing nomenclature/reference annotations -- STRIP these, the name is fine
_ANNOT    = re.compile(r"\s*\((INN|USAN|BAN|JAN|Standard|RG|Salt/Mix)\)\s*$", re.I)
# systematic-chemistry morphology: legitimate but should stay in the rare tail
_CHEMMORPH = re.compile(r"(silane|oxirane|imide|anilide|sulfonic|sulphonic|carbodiimide|"
                        r"isocyanate|isothiocyanate|malonic|phthalimide|xanthene|lepidine|"
                        r"diimide|diazene|benzene|propane|butene|cyclobutene|(?<!hydr)oxide$)", re.I)


def _is_junk(term):
    t = term.strip()
    if len(t) < 3 or len(t) > 90:
        return "length"
    if _MOJIBAKE.search(t):  return "mojibake"
    if _NOTATION.match(t):   return "notation"
    if _INCHIKEY.match(t):   return "inchikey"
    if _UNII_PRE.match(t):   return "unii"
    if _DB_ID.match(t):      return "db_id"
    if _FORMULA.match(t):    return "formula"
    if _CAS_IN.search(t):    return "cas"
    if _REF_STD.search(t):   return "reference_standard"
    if _DOTGREEK.search(t):  return "dot_greek_notation"
    if _PEPTIDE.match(t):    return "peptide_notation"
    if _DYE.search(t):       return "dye"
    if _SHORTCODE.match(t):  return "short_code"
    if _ATC.match(t):        return "atc_code"
    if _CAS.match(t):        return "cas"
    if _UNII.match(t):       return "unii"
    if _ARTICLE.match(t):    return "article"
    words = re.split(r"[\s\-]+", t.lower())
    if len(words) > 5:
        return "too_many_words"
    if _PRODUCT_WORDS & set(words):
        return "product_word"
    if _HAS_DIGIT_PCT.search(t):
        return "dose_or_strength"
    if t.count(",") >= 2 and not _CHEMY.search(t):
        return "multi_ingredient"
    if not re.search(r"[A-Za-z]", t):
        return "no_letters"
    return None


def _is_systematic(term):
    """Systematic/industrial chemistry: legitimate (S3.1.2, and drug_n entities like
    MPTP are exactly this) but should stay a rare tail, not the bulk."""
    return (len(term) > 25 and bool(_CHEMY.search(term))) or bool(_CHEMMORPH.search(term))


def clean_chemicals(chemicals, verbose=True):
    """Drop product junk; split the rest into 'simple' names and 'systematic' ones."""
    simple, systematic, dropped = [], [], {}
    seen = set()
    for term in chemicals:
        t = " ".join(str(term).split())
        t = _ANNOT.sub("", t)          # "Paroxypropione (INN)" -> "Paroxypropione"
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        reason = _is_junk(t)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        (systematic if _is_systematic(t) else simple).append(t)
    if verbose:
        print(f"kept {len(simple)} simple + {len(systematic)} systematic; "
              f"dropped {sum(dropped.values())} -> {dropped}")
    return simple, systematic


class Vocab:
    """Samples entity surface forms with a controllable type mix.

    group_prob      : share of slots that are class-level terms ("NSAIDs")
    systematic_prob : share that are long IUPAC-style names -- keep small
    """
    def __init__(self, simple, systematic, groups, group_prob=0.25,
                 systematic_prob=0.05, name="terms.json+curated_groups"):
        self.simple, self.systematic, self.groups = simple, systematic, groups
        self.group_prob, self.systematic_prob, self.name = group_prob, systematic_prob, name

    def sample(self, n, rng):
        """n distinct surface forms for one sentence."""
        # only draw from buckets that actually have content, and renormalise --
        # an empty bucket should not crash with an opaque IndexError
        buckets = []
        if self.groups:
            buckets.append((self.group_prob, self.groups))
        if self.systematic:
            buckets.append((self.systematic_prob, self.systematic))
        if self.simple:
            buckets.append((max(0.0, 1 - self.group_prob - self.systematic_prob), self.simple))
        if not buckets:
            raise ValueError("vocabulary is empty -- did the vocab build step fail?")
        total = sum(w for w, _ in buckets) or 1.0
        weights = [w / total for w, _ in buckets]
        pools = [p for _, p in buckets]

        out, tries = [], 0
        while len(out) < n and tries < n * 20:
            tries += 1
            pick = rng.choice(rng.choices(pools, weights=weights, k=1)[0])
            # distinct surface forms, else span resolution gets ambiguous
            if pick.lower() not in {o.lower() for o in out}:
                out.append(pick)
        if len(out) < n:
            raise ValueError(f"could not sample {n} distinct terms "
                             f"(vocab has {sum(len(p) for p in pools)} entries)")
        return out

    def fingerprint(self):
        """Hash of the exact vocab + mix -> goes in the manifest's vocab_source."""
        h = hashlib.sha256()
        for part in (sorted(self.simple), sorted(self.systematic), sorted(self.groups)):
            h.update(json.dumps(part).encode())
        h.update(json.dumps({"group_prob": self.group_prob,
                             "systematic_prob": self.systematic_prob}).encode())
        return {"name": self.name, "sha256": h.hexdigest(),
                "n_simple": len(self.simple), "n_systematic": len(self.systematic),
                "n_groups": len(self.groups),
                "group_prob": self.group_prob, "systematic_prob": self.systematic_prob}


def build_vocab(vocab_path="datasets/other/vocab.json", **kw):
    """Load the FROZEN vocabulary built once by scripts/build_vocab.py.

    Filtering happens once, offline. This keeps generation fast and -- more
    importantly -- makes the vocabulary a fixed artefact, so a dataset made in
    week 2 can still be reproduced after the filter code changes in week 4.
    """
    v = json.loads(Path(vocab_path).read_text())
    return Vocab(v["simple"], v["systematic"], v.get("groups") or DRUG_GROUPS,
                 name=Path(vocab_path).name, **kw)


def context_terms(vocab_path="datasets/other/vocab.json", limit=None):
    """Genes/diseases: NOT entities (guidelines S3.5.4 excludes enzymes), but real
    DDI prose mentions them constantly. Usable as flavour the model may mention
    but must not annotate."""
    v = json.loads(Path(vocab_path).read_text())
    out = v.get("context", [])
    return out[:limit] if limit else out