import csv, random, unicodedata, hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OTHER = ROOT / "datasets" / "other"

_UNICODE_FIXES = {"\u2010":"-","\u2011":"-","\u2012":"-","\u2013":"-","\u2014":"-",
    "\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',"\u00a0":" ","\u200b":""}
def _norm(s):
    for a, b in _UNICODE_FIXES.items():
        s = s.replace(a, b)
    return " ".join(unicodedata.normalize("NFKC", s).split())

_JUNK = {"combinations", "various", "na", "", "other", "combinations, other"}
def _is_bad_name(n):
    """Reject codes, formulae, paragraph-in-cell junk. Note: also drops legit numbered
    names (interferon alfa-2a); DDI-2013 has few, so net positive -- logged in notes.md."""
    if not n: return True
    if n.strip().lower() in _JUNK: return True
    if len(n.split()) > 6: return True
    if re.search(r"\d", n) and re.search(r"[-/]", n): return True   # digit + separator
    if sum(c.isdigit() for c in n) >= 2: return True                # 2+ digits => code
    if n.count("-") >= 3: return True
    return False

def _load_drugbank(path):
    """Cap-at-one: one canonical name per drug (Common_Name, else first clean synonym).
    Synonyms are otherwise deferred as the future name-variation ablation."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _norm(row.get("Common_Name_of_Drug") or "")
            if not _is_bad_name(name):
                out.append(name); continue
            for syn in (row.get("Synonym") or "").split("|"):
                syn = _norm(syn)
                if syn and not _is_bad_name(syn):
                    out.append(syn); break
    return out

def _load_atc(path):
    """Tier by code length: 7-char = substance (drug), 4-5 char = subgroup (group),
    1-3 char = anatomical top level (dropped -- never a DDI entity)."""
    drugs, groups = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("atc_code") or "").strip()
            name = _norm(row.get("atc_name") or "")
            if _is_bad_name(name): continue
            if len(code) >= 7:
                drugs.append(name)
            elif len(code) in (4, 5):
                groups.append(name.title() if name.isupper() else name)
    return drugs, groups

class Vocab:
    def __init__(self, drugs, groups, sources, p_group=0.3):
        self.drugs = sorted(set(drugs))
        self.groups = sorted(set(groups))
        self.sources = sources
        self.p_group = p_group            # ~0.3: approximate group fraction from DDI-2013
                                          # guidelines, NOT measured from corpus data.

    def sample(self, k, rng):
        out = []
        for _ in range(k):
            pool = self.groups if (self.groups and rng.random() < self.p_group) else self.drugs
            out.append(rng.choice(pool))
        return out

    def fingerprint(self):
        h = hashlib.sha256()
        h.update("|".join(self.sources).encode())
        h.update(f"p_group={self.p_group}".encode())
        h.update("\n".join(self.drugs).encode())
        h.update("\n".join(self.groups).encode())
        return {"name": "drugbank+atc", "sources": self.sources, "p_group": self.p_group,
                "n_drugs": len(self.drugs), "n_groups": len(self.groups),
                "sha256": h.hexdigest()[:12]}

def build_vocab(p_group=0.3):
    db = _load_drugbank(OTHER / "DrugBank.csv")
    atc_d, atc_g = _load_atc(OTHER / "WHO-ATC-DDD.csv")
    return Vocab(db + atc_d, atc_g, sources=["drugbank", "atc"], p_group=p_group)