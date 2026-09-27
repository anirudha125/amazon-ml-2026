"""E027_FR step 4: apply the pre-registered bars mechanically -> per-bucket link moves vs S006 (France only)."""
import json, pickle, collections, numpy as np
P = json.load(open("preregistration.json")); O = P["offsets"]; SHIFT = O["shift"]
C = json.load(open("census.json"))
E = pickle.load(open("ent.pkl", "rb")); ids = E["ids"]; IDX = {x: i for i, x in enumerate(ids)}
z = np.load("pairs_streq.npz"); a_all, b_all, d_all, e_all, ta, tb = z["a"], z["b"], z["d"], z["e"], z["ta"], z["tb"]
idx = np.load("idx_win.npy"); cls = np.load("cls_win.npy")
S6 = "../P3_rrL/submission_S006_rrL_mc/"
fr = set(i for i, x in enumerate(ids) if x.startswith("S1-"))
M = {}
with open(S6 + "matching_results.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, m = line.rstrip("\n").split("\t", 1)
        if s in IDX: M[IDX[s]] = set(IDX[x] for x in m.split(",") if x)
claimed = set().union(*M.values())
print("France S1 rows", len(M), "matched links", sum(map(len, M.values())), "empty", sum(not v for v in M.values()))
# rl31 p6 for diagnostics
r = np.load("../RL/rl31/test_scores/France.npz")
p6 = {(IDX[s], IDX[c]): p for s, c, p in zip(r["s1"], r["cand"], r["p6"])}
A = a_all[idx]; B = b_all[idx]; D = d_all[idx]; Eds = e_all[idx]
acc = np.array([b in M[a] for a, b in zip(A, B)])
bars = {}
# ---- bucket A: generic-role drop at delta 0
cA = C["A_generic_role_sub"]
bars["A_generic_role_sub"] = dict(shift_mirror_ratio=cA["shift_mirror_ratio"], n0=cA["n0"], bg=cA["background_per_offset"],
                                  clears_drop_bar=bool(cA["shift_mirror_ratio"] >= 5 or cA["n0"] <= cA["background_per_offset"]))
mA = (cls == "A_generic_role_sub") & (D == 0) & acc
dropA = list(zip(A[mA].tolist(), B[mA].tolist()))
# ---- bucket D: legal flip riding a shift k with per-k ratio >= 5
okk = [k for k in SHIFT if (C["D_legal_flip"]["per_k_ratio"][str(k)] or 0) >= 5]
bars["D_legal_flip_shift"] = dict(per_k_ratio=C["D_legal_flip"]["per_k_ratio"], ks_clearing=okk, clears_drop_bar=bool(okk))
mD = (cls == "D_legal_flip") & np.isin(D, okk) & acc
dropD = list(zip(A[mD].tolist(), B[mD].tolist()))
# ---- bucket B: filler/legal add at delta 0
cB = C["B_filler_legal_attach"]
bars["B_filler_legal_attach"] = dict(n0=cB["n0"], bg=cB["background_per_offset"], n0_over_bg=cB["n0_over_bg"],
                                     clears_add_bar=bool(cB["n0"] >= 5 * cB["background_per_offset"]))
good0 = (D == 0) & np.isin(cls, ["B_filler_legal_attach", "Z_exact_name"])
per_s1 = collections.Counter(A[good0].tolist()); per_c = collections.Counter(B[good0].tolist())
mB = (cls == "B_filler_legal_attach") & (D == 0) & ~acc
cand_add = [(a, b) for a, b in zip(A[mB].tolist(), B[mB].tolist()) if b not in claimed and per_s1[a] == 1 and per_c[b] == 1]
print("B d0 not accepted", int(mB.sum()), "after uniqueness+unclaimed", len(cand_add))
# candidate_pairs membership for adds
need = collections.defaultdict(set)
for a, b in cand_add: need[ids[a]].add(ids[b])
incp = set()
with open(S6 + "candidate_pairs.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s = line[:line.index("\t")]
        if s in need:
            cs = set(line.rstrip("\n").split("\t", 1)[1].split(","))
            for c in need[s]:
                if c in cs: incp.add((IDX[s], IDX[c]))
addB = [p for p in cand_add if p in incp]
print("adds in S006 candidate_pairs", len(addB))
def desc(pairs):
    ps = [p6.get(p) for p in pairs]; ps = [x for x in ps if x is not None]
    return dict(n=len(pairs), n_with_p6=len(ps), p6_quantiles=(np.quantile(ps, [.1, .25, .5, .75, .9]).round(3).tolist() if ps else None))
# empty-guard for drops
def guard(drops):
    by = collections.defaultdict(set)
    for a, b in drops: by[a].add(b)
    keep, skipped = [], 0
    for a, bs in by.items():
        if M[a] - bs: keep += [(a, b) for b in bs]
        else: skipped += 1
    return keep, skipped, len(by)
out = {}
for name, drops in (("A_generic_role_sub_drop", dropA), ("D_legal_flip_shift_drop", dropD)):
    k, sk, ns = guard(drops)
    out[name] = dict(candidates=desc(drops), applied=len(k), s1_touched=ns, s1_skipped_would_empty=sk, pairs=[(ids[a], ids[b]) for a, b in k])
out["B_filler_legal_add"] = dict(candidates=desc(addB), applied=len(addB), s1_touched=len(set(a for a, _ in addB)), pairs=[(ids[a], ids[b]) for a, b in addB])
# diagnostics: A delta0 all (accepted or not), per token pair accept rate; same-S1 other good match at same address
allA0 = (cls == "A_generic_role_sub") & (D == 0)
diag = dict(A_d0_total=int(allA0.sum()), A_d0_accepted=int((allA0 & acc).sum()),
            A_d0_acc_with_other_good_match_same_addr=int(sum(per_s1[a] >= 1 for a in A[allA0 & acc].tolist())),
            B_d0_total=int(((cls == "B_filler_legal_attach") & (D == 0)).sum()), B_d0_accepted=int(((cls == "B_filler_legal_attach") & (D == 0) & acc).sum()),
            D_shift_total=int(((cls == "D_legal_flip") & np.isin(D, okk)).sum()), D_shift_accepted=int(mD.sum()),
            A_shift_accepted=int(((cls == "A_generic_role_sub") & np.isin(D, SHIFT) & acc).sum()),
            A_mirror_accepted=int(((cls == "A_generic_role_sub") & np.isin(D, [-k for k in SHIFT]) & acc).sum()),
            acc_by_class_delta0={c: int(((cls == c) & (D == 0) & acc).sum()) for c in sorted(set(cls))},
            acc_by_class_shift={c: int(((cls == c) & np.isin(D, SHIFT) & acc).sum()) for c in sorted(set(cls))},
            acc_by_class_mirror={c: int(((cls == c) & np.isin(D, [-k for k in SHIFT]) & acc).sum()) for c in sorted(set(cls))})
json.dump(dict(bars=bars, moves=out, diag=diag), open("moves.json", "w"), indent=1)
print(json.dumps(bars, indent=1)); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in out.items()}, indent=1)); print(json.dumps(diag, indent=1))
