"""E027_FR follow-up (POST-HOC): S006-accepted links at shift-set vs mirror offsets per one-token class; drop list for the
S007FR_shift hedge; expected gain with the exact per-S1 F0.5 formula at each class's implied false share."""
import json, pickle, collections, itertools, numpy as np
P = json.load(open("preregistration.json")); SHIFT = P["offsets"]["shift"]; MIRROR = [-k for k in SHIFT]
C = json.load(open("census.json"))
E = pickle.load(open("ent.pkl", "rb")); ids = E["ids"]; IDX = {x: i for i, x in enumerate(ids)}
z = np.load("pairs_streq.npz"); idx = np.load("idx_win.npy"); cls = np.load("cls_win.npy")
A = z["a"][idx]; B = z["b"][idx]; D = z["d"][idx]
M = {}; Ml = {}
with open("../P3_rrL/submission_S006_rrL_mc/matching_results.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, m = line.rstrip("\n").split("\t", 1)
        if s in IDX: Ml[s] = [x for x in m.split(",") if x]; M[IDX[s]] = set(IDX[x] for x in Ml[s])
acc = np.array([b in M[a] for a, b in zip(A.tolist(), B.tolist())])
r = np.load("../RL/rl31/test_scores/France.npz")
p6 = {(IDX[s], IDX[c]): float(p) for s, c, p in zip(r["s1"], r["cand"], r["p6"])}
CLASSES = ["A_generic_role_sub", "B_filler_legal_attach", "C_rare_sub", "D_legal_flip", "X_content_sub", "Y_other_attach", "Z_exact_name", "W_multi_token"]
res = {}; drops = []; fshare = {}
for c in CLASSES:
    mc = cls == c
    ash = {k: int((mc & acc & (D == k)).sum()) for k in SHIFT}; ami = {k: int((mc & acc & (D == -k)).sum()) for k in SHIFT}
    ps, pm = C[c]["shift_sum"], C[c]["mirror_sum"]
    sa, ma = sum(ash.values()), sum(ami.values())
    msh = mc & acc & np.isin(D, SHIFT)
    q = [p6.get(p) for p in zip(A[msh].tolist(), B[msh].tolist())]; q = [x for x in q if x is not None]
    f = (1 - ma / sa) if sa else None
    sel = bool(ps / max(pm, 1) >= 5 and sa and (ma == 0 or sa / ma >= 5))
    res[c] = dict(accepted_shift=sa, accepted_mirror=ma, pool_shift=ps, pool_mirror=pm, pool_ratio=ps / pm if pm else None,
                  accepted_ratio=(sa / ma if ma else None), implied_false_share=f, p6_quantiles_10_25_50_75_90=(np.quantile(q, [.1, .25, .5, .75, .9]).round(3).tolist() if q else None),
                  n_with_p6=len(q), accepted_shift_per_k=ash, accepted_mirror_per_k=ami, selected=sel)
    if sel:
        fshare[c] = f
        drops += [(a, b, c) for a, b in zip(A[msh].tolist(), B[msh].tolist())]
# empty guard (all selected classes combined per S1)
by = collections.defaultdict(list)
for a, b, c in drops: by[a].append((b, c))
keep = []; skipped = collections.Counter(); skipped_s1 = 0
for a, lst in by.items():
    if M[a] - set(b for b, _ in lst): keep += [(a, b, c) for b, c in lst]
    else: skipped_s1 += 1; skipped.update(c for _, c in lst)
# exact expected gain, per-link false prob = its class's implied false share
fr_n = sum(1 for l in open("../../student_resource/dataset/test/test_source1.tsv", encoding="utf-8") if l.rstrip("\r\n").endswith("\tFrance"))
def F(gt, npred, tp):
    if gt == 0: return 1.0 if npred == 0 else 0.0
    if npred == 0 or tp == 0: return 0.0
    p, rr = tp / npred, tp / gt; return 1.25 * p * rr / (0.25 * p + rr)
def gain(pairs, fmap):
    g = collections.defaultdict(list)
    for a, b, c in pairs: g[a].append(fmap[c])
    tot = 0.0
    for a, fs in g.items():
        n = len(M[a]); k = n - len(fs)
        for pat in itertools.product((0, 1), repeat=len(fs)):      # 1 = removed link is false
            w = np.prod([f if x else 1 - f for f, x in zip(fs, pat)]); t = len(fs) - sum(pat); gt = k + t
            tot += w * (F(gt, k, k) - F(gt, n, gt))
    return tot / fr_n * 100
out = dict(label="POST-HOC extension of the pre-registered legal-flip-with-shift rule to the other one-token classes",
           rule="drop S006-accepted street-equal France links at shift-set offsets for classes with pool shift/mirror >= 5 AND accepted shift/mirror >= 5; never empty an S1",
           per_class=res, selected=list(fshare), s1_skipped_would_empty=skipped_s1, links_skipped_by_class=dict(skipped))
per = {}
for c in fshare:
    pc = [p for p in keep if p[2] == c]
    g = gain(pc, fshare); per[c] = dict(links_removed=len(pc), s1_touched=len(set(p[0] for p in pc)), f=fshare[c], dF_France_pt=round(g, 4), dLB=round(0.14975 * g / 100, 6))
g = gain(keep, fshare)
out["per_bucket"] = per
out["total"] = dict(links_removed=len(keep), s1_touched=len(set(p[0] for p in keep)), dF_France_pt=round(g, 4), dLB=round(0.14975 * g / 100, 6))
for f in (0.31, 0.5, 0.7):
    gg = gain(keep, {c: f for c in fshare}); out["total"][f"uniform_false={f}"] = dict(dF_France_pt=round(gg, 4), dLB=round(0.14975 * gg / 100, 6))
out["pairs"] = [(ids[a], ids[b], c) for a, b, c in keep]
json.dump(out, open("shift.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
