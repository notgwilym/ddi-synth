import sys; sys.path.insert(0, "scripts")
import run_grid as rg

H, L, S, val = rg.load_pools()              # L: the model-labelled pool, with gold kept as gold_label
LF, n_rejected, n_hit = rg.filtered_pool(L) # LF: the same pool after the verifier's demotions
expected, attempted, ok = rg.verify_status(L)   # ok: sentences whose verifier request succeeded

def prf(rows, key):
    tp = sum(r[key] != "NONE" and r[key] == r["gold_label"] for r in rows)
    fp = sum(r[key] != "NONE" and r[key] != r["gold_label"] for r in rows)
    fn = sum(r["gold_label"] != "NONE" and r[key] != r["gold_label"] for r in rows)
    return f"P {tp/(tp+fp):.3f}  R {tp/(tp+fn):.3f}  positives {tp+fp}"

print("check 1  model labels against gold:", prf(L, "label"))
print("         after checking:           ", prf(LF, "label"))

pos = [(a, b) for a, b in zip(L, LF) if a["label"] != "NONE"]
groups = {"rejected (demoted)": [a for a, b in pos if b["label"] == "NONE"],
          "accepted":           [a for a, b in pos if b["label"] != "NONE" and a["sent_id"] in ok],
          "never checked":      [a for a, b in pos if b["label"] != "NONE" and a["sent_id"] not in ok]}
for name, rows in groups.items():
    k = sum(r["gold_label"] == "NONE" for r in rows)
    print(f"check 2  {name:<20} {len(rows):>5} pairs, {k:>5} NONE in gold ({k / max(len(rows), 1):.0%})")