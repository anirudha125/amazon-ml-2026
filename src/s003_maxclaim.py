"""
S003 -- S002 (E018C_s42) + max-claimer conflict resolution (E020: India +0.157 [+0.113,+0.202], US +0.100 [+0.075,+0.128]
on dense labelled train slices with the same production model). No retraining, no threshold change.

Rule (deterministic, parameter-free): per country, every S2/S3 id predicted for more than one S1 is kept only for the S1 with the
highest stage-2 probability (ties -> smallest S1 id). Everything else is unchanged. Predictions only shrink, so
predictions ⊆ candidate_pairs still holds; candidate_pairs.tsv is S002's file, byte-identical.
Inputs (read-only): experiments/SUBMISSIONS/S002_E018C_s42/{matching_results.tsv, candidate_pairs.tsv.gz},
  experiments/TEST_PIPELINE/slim_preds/E018C_s42_<country>.pkl (per-pair probabilities).
Check before applying: slim preds must reproduce S002 matching_results exactly (per-S1 set equality).
Output: experiments/TEST_PIPELINE/submission_S003_E018C_s42_maxclaim/ + stats.json. Nothing is tuned on test data.
"""
import os, sys, json, gzip, pickle, shutil, collections, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2D = os.path.join(ROOT, "experiments", "SUBMISSIONS", "S002_E018C_s42")
SP = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "slim_preds")
OUT = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "submission_S003_E018C_s42_maxclaim"); os.makedirs(OUT, exist_ok=True)
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

preds = {}
for c in ["US", "India", "France"]:
    d = pickle.load(open(os.path.join(SP, f"E018C_s42_{c}.pkl"), "rb"))
    for s, v in d["preds"].items():
        preds[s] = (c, v)
rows = []
with open(os.path.join(S2D, "matching_results.tsv"), encoding="utf-8") as f:
    assert f.readline().rstrip(NL) == "source1_entity_id" + TAB + "matched_entity_ids"
    for line in f:
        s, m = line.rstrip(NL).split(TAB); rows.append(s)
        L = set(m.split(",")) if m else set()
        P = {x for x, _ in preds[s][1]} if s in preds else set()
        assert L == P, (s, L, P)
log(f"slim preds reproduce S002 exactly: {len(rows):,} rows")

claims = collections.defaultdict(list)
for s, (c, v) in preds.items():
    for x, p in v:
        claims[(c, x)].append((-p, s))
keep = {}
st = collections.defaultdict(collections.Counter)
for (c, x), L in claims.items():
    L.sort()
    if len(L) > 1:
        st[c]["contested_records"] += 1; st[c]["claims_removed"] += len(L) - 1
    for i, (_, s) in enumerate(L):
        keep[(s, x)] = i == 0
new = {}
for s, (c, v) in preds.items():
    k = [x for x, _ in v if keep[(s, x)]]
    st[c]["pred_pairs_before"] += len(v); st[c]["pred_pairs_after"] += len(k)
    if not k:
        st[c]["s1_emptied"] += 1
    new[s] = k
for c in st:
    st[c]["removed_per_1k_preds"] = round(1000 * st[c]["claims_removed"] / st[c]["pred_pairs_before"], 2)
log(json.dumps(st, indent=1))
with open(os.path.join(OUT, "matching_results.tsv"), "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id" + TAB + "matched_entity_ids" + NL)
    for s in rows:
        f.write(s + TAB + ",".join(new.get(s, [])) + NL)
with gzip.open(os.path.join(S2D, "candidate_pairs.tsv.gz"), "rb") as fi, open(os.path.join(OUT, "candidate_pairs.tsv"), "wb") as fo:
    shutil.copyfileobj(fi, fo, 1 << 24)
json.dump(dict(base="S002_E018C_s42", rule="max-claimer per country (ties -> smallest S1 id)", evidence="E020", stats=st),
          open(os.path.join(OUT, "stats.json"), "w"), indent=1)
log("wrote", OUT)
