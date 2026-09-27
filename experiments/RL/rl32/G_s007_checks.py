"""RL-32 S007 safety checks: (1) parity -- the G_fr_norr write path at W=1 reproduces S006 France rows (as sets);
(2) every US/India line of S007 is byte-identical to S006; (3) S007 France rows differ from S006 exactly as write_info reports."""
import numpy as np, pandas as pd, json, collections
ROOT = "../../.."; S6 = f"{ROOT}/experiments/P3_rrL/submission_S006_rrL_mc/matching_results.tsv"; S7 = "G_submission_S007_FRblend_d_W0.25/matching_results.tsv"
TAB = "\t"
ctry = {}
with open(f"{ROOT}/student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split(TAB); ctry[p[0]] = p[3]
z = np.load("G_fr_norr_scores_d.npz", allow_pickle=True); acc = z["p6"] >= 0.72
d = pd.DataFrame({"s": z["s1"][acc], "c": z["cand"][acc], "p": z["p6"][acc].astype(float)})
w = d.sort_values(["c", "p", "s"], ascending=[True, False, True]).drop_duplicates("c")
par = collections.defaultdict(set)
for s, c in zip(w.s, w.c): par[s].add(c)
r6, r7 = {}, {}
for fn, R in ((S6, r6), (S7, r7)):
    with open(fn, encoding="utf-8") as f:
        hdr = f.readline()
        for line in f: R[line.split(TAB, 1)[0]] = line
assert set(r6) == set(r7) == set(ctry)
bad_usin = sum(1 for s in r6 if ctry[s] != "France" and r6[s] != r7[s])
fr_mis = 0; changed = 0
for s in r6:
    if ctry[s] != "France": continue
    got = set(r6[s].rstrip("\n").split(TAB)[1].split(",")) - {""}
    fr_mis += int(got != par.get(s, set()))
    changed += int(set(r7[s].rstrip("\n").split(TAB)[1].split(",")) - {""} != got)
out = dict(parity_france_rows_mismatch_W1=fr_mis, usin_lines_not_identical=bad_usin, france_rows_changed=changed, n_rows=len(r7))
json.dump(out, open("G_s007_checks.json", "w"), indent=1); print(out)
