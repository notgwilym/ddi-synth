import json, os, itertools, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import MARKERS, render_pair, Overlap

from .manifest import write_dataset, DATA_ROOT

RAW = DATA_ROOT / "raw"

# The model emits typographic unicode (non-breaking hyphens, en/em dashes, narrow
# spaces) in the SENTENCE while writing plain ASCII in the entity list so
# sentence.find(entity) fails and the sample is rejected. Normalise both sides
# identically before matching. Markdown bold is stripped for the same reason.
_UNICODE_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u200b": "",
}


def _normalise(s):
    for a, b in _UNICODE_FIXES.items():
        s = s.replace(a, b)
    s = s.replace("**", "").replace("__", "")        # markdown emphasis
    s = unicodedata.normalize("NFKC", s)             # also folds CH₃ -> CH3
    return " ".join(s.split())                       # collapse whitespace


def _is_degenerate(text, max_run=6, min_unique=0.3):
    """Catch the 'mandatory mandatory mandatory...' decoding loops."""
    words = text.split()
    if len(words) < 15:
        return False
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a.lower() == b.lower() else 1
        if run > max_run:
            return True
    return len(set(w.lower() for w in words)) / len(words) < min_unique


# shims that quack like bioc's BratDocument

class _Loc:
    def __init__(self, b, e): self._b, self._e = b, e
    def begin(self): return self._b
    def end(self): return self._e


class SynthEntity:
    def __init__(self, id, begin, end, type="drug"):
        self.id, self.locations, self.type = id, _Loc(begin, end), type


class SynthRelation:
    def __init__(self, arg1, arg2, type):
        self.arguments, self.type = {"Arg1": arg1, "Arg2": arg2}, type


class SynthDoc:
    def __init__(self, text, entities, relations, register, sent_id):
        self.text, self.entities, self.relations = text, entities, relations
        self.register, self.sent_id = register, sent_id


class Rejected(Exception):
    """Raised when a generated sample can't be turned into valid instances."""


def _find_nth(sentence, surface, n):
    """Index of the (n+1)-th occurrence of `surface`, or -1.

    Exact match first. Falls back to case-insensitive, because the model
    routinely capitalises a name at the start of a sentence ("antiplatelet
    agents" -> "Antiplatelet agents") or lower-cases one mid-sentence
    ("Quinacrine diHCl" -> "quinacrine diHCl"). The span length is unchanged
    either way, so the offsets stay valid.
    """
    for hay, needle in ((sentence, surface), (sentence.lower(), surface.lower())):
        start = -1
        for _ in range(n + 1):
            start = hay.find(needle, start + 1)
            if start == -1:
                break
        if start != -1:
            return start
    return -1

def _entities_in_sentence(text, names):
    """Every occurrence of each known drug name in this sentence -> SynthEntity list.
    name_of maps each entity object -> its lowercased drug name, for relation binding."""
    ents, name_of = [], {}
    for nm in dict.fromkeys(names):                     # unique, order preserved
        k = 0
        while True:
            pos = _find_nth(text, nm, k)
            if pos == -1:
                break
            e = SynthEntity(None, pos, pos + len(nm), "drug")
            e.name_lc = nm.lower()
            ents.append(e); name_of[id(e)] = nm.lower()
            k += 1
    ents = sorted(ents, key=lambda e: e.locations.begin())
    for a, b in zip(ents, ents[1:]):
        if b.locations.begin() < a.locations.end():
            raise Rejected("overlapping entity spans (one drug name is a substring of another?)")
    for j, e in enumerate(ents):
        e.id = f"T{j}"
    return ents, name_of

def _collapse_to_first_mention(entities):
    """Synthetic only: keep one entity per unique drug name (first by position).
    Relations are name-level here, so this gives one instance per name-pair and
    removes self-pairs + repeated-partner contradictions. Diverges deliberately
    from the human path, which enumerates every mention (gold is ID-level)."""
    seen, out = set(), []
    for e in sorted(entities, key=lambda e: e.locations.begin()):
        nm = e.name_lc if hasattr(e, "name_lc") else None
        if nm is None:
            raise Rejected("entity missing name for collapse")
        if nm not in seen:
            seen.add(nm); out.append(e)
    return out

def _make_pair_instances_synth(doc, mode="markers"):
    id2name = {e.id: e.name_lc for e in doc.entities}
    name_label = {}
    self_binds = 0
    for rel in doc.relations:
        n1 = id2name.get(rel.arguments["Arg1"])
        n2 = id2name.get(rel.arguments["Arg2"])
        if not n1 or not n2:
            continue
        if n1 == n2:
            self_binds += 1
            continue
        name_label[frozenset((n1, n2))] = rel.type

    ents = _collapse_to_first_mention(doc.entities)
    out = []
    for e1, e2 in itertools.combinations(ents, 2):
        if e1.name_lc == e2.name_lc:
            continue
        try:
            t = render_pair(doc.text, e1, e2, ents, mode)
        except Overlap:
            continue
        label = name_label.get(frozenset((e1.name_lc, e2.name_lc)), "NONE")
        out.append({"text": t, "label": label,
                    "source": doc.register, "sent_id": doc.sent_id})
    return out, self_binds


def sample_to_instances(sample, sent_id_base, register="synthetic", max_words=200, mode="markers"):
    """Vignette sample -> per-sentence {text,label,source,sent_id} instances.

    Schema: {sentences:[{text, relations:[{arg1,arg2,label}]}], entities:[{text,type}]}.
    Relations are pre-scoped by the model to their own sentence, so there is no
    cross-sentence inference and no reliance on model mention-ids. Binding is by drug
    NAME within the declared sentence; a repeated name binds to the closest co-occurring
    pair. Each sentence is paired independently via make_pair_instances, so non-asserted
    pairs (including the same drug pair appearing un-interacting in another sentence)
    fall through to NONE -- correct, in-distribution negatives."""
    if not isinstance(sample, dict):
        raise Rejected("no sample returned")
    sents = sample.get("sentences") or []
    if not sents:
        raise Rejected("no sentences")

    entity_names = []
    for e in (sample.get("entities") or []):
        txt = _normalise(e.get("text") or "")
        if not txt:
            raise Rejected("empty entity text")
        if len(txt.split()) > 12:
            raise Rejected("entity text looks like a sentence, not a name")
        entity_names.append(txt)
    if len(entity_names) < 2:
        raise Rejected(f"need >=2 entities to form a pair, got {len(entity_names)}")
    known = {n.lower() for n in entity_names}

    full = " ".join(_normalise(s.get("text") or "") for s in sents)
    if len(full.split()) > max_words:
        raise Rejected(f"passage too long ({len(full.split())} words)")
    if _is_degenerate(full):
        raise Rejected("degenerate repetition in passage")

    instances = []
    self_bind_count = 0
    for i, sd in enumerate(sents):
        text = _normalise(sd.get("text") or "")
        if not text:
            continue
        ents, name_of = _entities_in_sentence(text, entity_names)
        rels = []
        malformed = 0
        for rel in (sd.get("relations") or []):
            a1 = _normalise(rel.get("arg1") or "").lower()
            a2 = _normalise(rel.get("arg2") or "").lower()
            if not a1 or not a2 or a1 not in known or a2 not in known:
                malformed += 1                      # log it, don't raise
                continue
            if a1 == a2:
                raise Rejected("self-relation")
            m1 = [e for e in ents if name_of[id(e)] == a1]
            m2 = [e for e in ents if name_of[id(e)] == a2]
            if not m1 or not m2:
                raise Rejected(f"relation {rel.get('label')} {a1}~{a2}: both args not in its sentence")
            # a drug may legitimately repeat in one sentence; bind the closest pair
            best = min(((e1, e2) for e1 in m1 for e2 in m2),
                       key=lambda p: abs(p[0].locations.begin() - p[1].locations.begin()))
            rels.append(SynthRelation(best[0].id, best[1].id, rel["label"]))
        doc = SynthDoc(text, ents, rels, register, f"{sent_id_base}:s{i}")
        sent_insts, sb = _make_pair_instances_synth(doc, mode=mode)
        instances.extend(sent_insts)
        self_bind_count += sb
    if not instances:
        raise Rejected("no instances produced (no sentence had >=2 entities)")
    if self_bind_count:
        print(f"warning: {self_bind_count} self-relations were ignored in {sent_id_base}")
    return instances


def generate_raw(specs, sample_fn, gen_id, max_workers=64, resume=True):
    """Call sample_fn(spec) concurrently; append each result to raw/<gen_id>.jsonl.

    Appends as results arrive, so a dead pod costs only in-flight requests.
    resume=True skips specs already present in the raw file.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{gen_id}.jsonl"

    done = set()
    if resume and path.exists():
        for line in path.read_text().splitlines():
            if line:
                rec = json.loads(line)
                # only successes count as done -- errored specs must be retried,
                # otherwise transient API failures silently shrink the dataset
                if not rec.get("error"):
                    done.add(rec["spec_index"])
        print(f"resuming {gen_id}: {len(done)} already succeeded")

    todo = [(i, s) for i, s in enumerate(specs) if i not in done]
    if not todo:
        print("nothing to do")
        return path

    lock = threading.Lock()
    n_ok = n_err = 0

    def _work(item):
        i, spec = item
        try:
            return i, spec, sample_fn(spec), None
        except Exception as e:                      # network/API/parse failure
            return i, spec, None, f"{type(e).__name__}: {e}"

    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc=f"gen {gen_id}", unit="req", smoothing=0.1)
    except ImportError:
        bar = None

    with open(path, "a") as fp, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, item) for item in todo]
        for fut in as_completed(futures):
            i, spec, sample, err = fut.result()
            with lock:
                fp.write(json.dumps({"spec_index": i, "spec": spec,
                                     "sample": sample, "error": err}) + "\n")
                fp.flush()                          # survive an abrupt pod death
                if err: n_err += 1
                else:   n_ok += 1
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
                elif (n_ok + n_err) % 200 == 0:
                    print(f"  {n_ok + n_err}/{len(todo)}  (errors: {n_err})")
    if bar is not None:
        bar.close()

    print(f"stage 1 done: {n_ok} ok, {n_err} api/parse errors -> {path}")
    return path


def build_dataset_from_raw(gen_id, generator, vocab_source=None,
                           negative_strategy=None, seed=None, notes="",
                           resolver=None, mode="markers"):
    """Deterministic: raw model output -> instances -> manifested dataset.

    Free to re-run whenever the resolver or pair logic changes. Returns
    (dataset_id, stats) where stats records exactly what was thrown away and why.
    """
    path = RAW / f"{gen_id}.jsonl"
    instances, rejects = [], []
    n_api_err = 0

    for line in path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("error"):
            n_api_err += 1
            continue
        sent_id = f"synth:{gen_id}:{rec['spec_index']}"
        register = (rec.get("spec") or {}).get("register", "synthetic")
        try:
            if resolver is None:
                instances.extend(sample_to_instances(rec["sample"], sent_id, register=register, mode=mode))
            else:
                instances.extend(resolver(rec["sample"], sent_id, register=register,
                                          spec=rec["spec"], mode=mode))
        except Rejected as e:
            rejects.append({"spec_index": rec["spec_index"], "reason": str(e),
                            "sample": rec["sample"]})

    n_samples = len(instances) and len({r["sent_id"] for r in instances})
    stats = {
        "gen_id": gen_id,
        "n_api_errors": n_api_err,
        "n_rejected": len(rejects),
        "n_sentences_used": n_samples,
        "n_instances": len(instances),
        "reject_reasons": _reason_counts(rejects),
    }

    # keep the rejects: they are the diagnostic for prompt iteration
    if rejects:
        (RAW / f"{gen_id}.rejects.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rejects) + "\n")

    if not instances:
        raise Rejected(f"no usable instances from {gen_id}: {stats}")

    dataset_id = write_dataset(
        instances, provenance="synthetic", generator={**(generator or {}), "gen_id": gen_id},
        vocab_source=vocab_source, negative_strategy=negative_strategy,
        seed=seed, notes=notes, render_mode=mode,
    )
    print(f"stage 2: {stats['n_rejected']} rejected, {stats['n_instances']} instances kept")
    return dataset_id, stats


def _reason_counts(rejects):
    out = {}
    for r in rejects:
        key = r["reason"].split(":")[0].split(" not found")[0][:60]
        out[key] = out.get(key, 0) + 1
    return out