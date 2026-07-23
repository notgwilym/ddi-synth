"""Build the frozen vocabulary. Run ONCE (re-run only if you change the filters).

    python scripts/build_vocab.py

Writes datasets/other/vocab.json, which every generation run then loads as-is.
Freezing it means a dataset stays reproducible even after the filter code changes.
Safe to hand-edit the output afterwards -- it is the source of truth, not the code.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddi.vocab import clean_chemicals, DRUG_GROUPS

TERMS = Path("datasets/other/terms.json")
OUT = Path("datasets/other/vocab.json")


def main():
    raw = json.loads(TERMS.read_text())

    chemicals = raw["chemical"] if isinstance(raw, dict) else raw
    if isinstance(chemicals, dict):          # {term: id} style
        chemicals = list(chemicals.keys())
    simple, systematic = clean_chemicals(chemicals)

    # genes/diseases are not DDI entities, but are realistic sentence flavour
    context = []
    if isinstance(raw, dict):
        for k in ("gene", "disease"):
            vals = raw.get(k, [])
            context.extend(list(vals.keys()) if isinstance(vals, dict) else vals)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"simple": simple, "systematic": systematic,
         "groups": DRUG_GROUPS, "context": context}, indent=0))

    print(f"\nwrote {OUT}")
    print(f"  simple      {len(simple)}")
    print(f"  systematic  {len(systematic)}")
    print(f"  groups      {len(DRUG_GROUPS)}")
    print(f"  context     {len(context)}")
    print(f"  size        {OUT.stat().st_size / 1e6:.1f} MB")
    print("\nEyeball a random sample before generating, e.g.:")
    print("  python -c \"import json,random; v=json.load(open('datasets/other/vocab.json'));"
          " print(random.sample(v['simple'],50))\"")


if __name__ == "__main__":
    main()