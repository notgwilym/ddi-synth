import json
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

_ADMIN = re.compile(r"\b(combinations?|other|others|various|excl\.?|incl\.?|"
                    r"preparations?|products?|agents,|substances|reagents?|chemicals?|"
                    r"equipment|disinfectants?|devices?|solutions?|diagnostic|"
                    r"non-therapeutic|palliation|technical)\b"
                    r"|chemotherapeutics|related", re.I)
_NONDRUG = re.compile(r"\b(crab|pollen|herbarum|spp\.?|serotype|adenovirus|vaccine|"
                      r"lecithin|starch|honey|extract)\b", re.I)

def _is_bad_group(name):
    if _is_bad_name(name): return True
    if len(name.split()) > 3: return True
    if "," in name or _ADMIN.search(name): return True
    return False

def _is_bad_drug(name):
    if _is_bad_name(name): return True
    if " and " in name.lower(): return True
    if len(name.split()) > 4: return True
    if "," in name or _ADMIN.search(name) or _NONDRUG.search(name): return True
    return False

_ADMIN = re.compile(r"\b(combinations?|other|others|various|excl\.?|incl\.?|"
                    r"preparations?|products?|agents,|substances|reagents?|chemicals?|"
                    r"equipment|disinfectants?|devices?|solutions?|diagnostic|"
                    r"non-therapeutic|palliation|technical)\b", re.I)
_NONDRUG = re.compile(r"\b(crab|pollen|herbarum|spp\.?|serotype|adenovirus|vaccine|"
                      r"lecithin|starch|honey|extract)\b", re.I)

def _is_bad_group(name):
    if _is_bad_name(name): return True
    if len(name.split()) > 3: return True
    if "," in name or _ADMIN.search(name): return True
    return False

def _is_bad_drug(name):
    if _is_bad_name(name): return True
    if len(name.split()) > 4: return True
    if "," in name or _ADMIN.search(name) or _NONDRUG.search(name): return True
    return False

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

def _load_drugbank(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = _norm(row.get("Common_Name_of_Drug") or "")
            if not _is_bad_drug(name):
                out.append(name); continue
            for syn in (row.get("Synonym") or "").split("|"):
                syn = _norm(syn)
                if syn and not _is_bad_drug(syn):
                    out.append(syn); break
    return out

def _load_atc(path):
    drugs, groups = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("atc_code") or "").strip()
            name = _norm(row.get("atc_name") or "")
            if len(code) >= 7:
                if not _is_bad_drug(name): drugs.append(name)
            elif len(code) in (4, 5):
                clean = name.title() if name.isupper() else name
                if not _is_bad_group(clean): groups.append(clean)
    return drugs, groups

FILTERED = OTHER / "vocab_filtered.json"

def build_vocab(p_group=0.3, filtered=False):
    """filtered=True loads the LLM-curated lexicon. The unfiltered path stays as a
    one-line ablation: 'curated lexicon vs raw DrugBank+ATC' is a real arm."""
    if filtered:
        if not FILTERED.exists():
            raise FileNotFoundError(f"{FILTERED} missing; run scripts/filter_vocab_llm.py")
        d = json.loads(FILTERED.read_text())
        return Vocab(d["drugs"], d["groups"], sources=["drugbank", "atc", "llm-filtered"],
                     p_group=p_group)
    db = _load_drugbank(OTHER / "DrugBank.csv")
    atc_d, atc_g = _load_atc(OTHER / "WHO-ATC-DDD.csv")
    return Vocab(db + atc_d, atc_g, sources=["drugbank", "atc"], p_group=p_group)