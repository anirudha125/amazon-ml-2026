"""E027_FR step 5: expected France dF0.5 / LB per bucket with the exact per-S1 F0.5 formula.
Assumption: every S006 link NOT touched is true and the GT of the S1 = its true links in (S006 list + added). Each moved link is
false with probability f (independent). Empty GT + empty pred = 1 (official scorer)."""
import json, collections, numpy as np
from math import comb
mv = json.load(open("moves.json")); C = json.load(open("census.json"))
M = {}
with open("../P3_rrL/submission_S006_rrL_mc/matching_results.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, m = line.rstrip("\n").split("\t", 1)
        if s.startswith("S1-"): M[s] = [x for x in m.split(",") if x]
fr = set(l.split("\t")[0] for l in open("../../student_resource/dataset/test/test_source1.tsv", encoding="utf-8") if l.rstrip("\r\n").endswith("\tFrance"))
NFR = len(fr)
def F(gt, npred, tp):
    if gt == 0: return 1.0 if npred == 0 else 0.0
    if npred == 0 or tp == 0: return 0.0
    p, r = tp / npred, tp / gt; return 1.25 * p * r / (0.25 * p + r)
def drop_gain(pairs, f):
    by = collections.Counter(s for s, _ in pairs); tot = 0.0
    for s, r in by.items():
        n = len(M[s]); k = n - r
        for x in range(r + 1):                      # x = number of false among removed
            w = comb(r, x) * f ** x * (1 - f) ** (r - x); gt = k + (r - x)
            tot += w * (F(gt, k, k) - F(gt, n, gt))
    return tot / NFR * 100
def add_gain(pairs, f):
    by = collections.Counter(s for s, _ in pairs); tot = 0.0
    for s, r in by.items():
        n = len(M[s])
        for x in range(r + 1):                      # x = number of false among added
            w = comb(r, x) * f ** x * (1 - f) ** (r - x); gt = n + (r - x)
            tot += w * (F(gt, n + r, gt) - F(gt, n, n))
    return tot / NFR * 100
cA = C["A_generic_role_sub"]; cD = C["D_legal_flip"]; cB = C["B_filler_legal_attach"]
est = dict(
    A_false_share_bg=cA["background_per_offset"] / cA["n0"],
    A_false_share_shift_excess=(cA["shift_per_offset"] - cA["background_per_offset"]) / cA["n0"],
    D_shift_false_share_bg=1 - cD["background_per_offset"] / cD["shift_per_offset"],
    B_add_false_share_bg=cB["background_per_offset"] / cB["n0"])
res = {"France_S1": NFR, "estimates": est, "buckets": {}}
for name, fn in (("A_generic_role_sub_drop", drop_gain), ("D_legal_flip_shift_drop", drop_gain), ("B_filler_legal_add", add_gain)):
    pairs = mv["moves"][name]["pairs"]; row = {"links": len(pairs), "empty_s1_before": sum(len(M[s]) == 0 for s in set(p[0] for p in pairs))}
    fs = [0.31, 0.50, 0.70]
    key = {"A_generic_role_sub_drop": ["A_false_share_bg", "A_false_share_shift_excess"], "D_legal_flip_shift_drop": ["D_shift_false_share_bg"], "B_filler_legal_add": ["B_add_false_share_bg"]}[name]
    for kname in key: fs.append(est[kname])
    for f in fs:
        g = fn(pairs, f); row[f"false={f:.3f}"] = dict(dF_France_pt=round(g, 4), dLB=round(0.14975 * g / 100, 6))
    res["buckets"][name] = row
json.dump(res, open("gain.json", "w"), indent=1); print(json.dumps(res, indent=1))
