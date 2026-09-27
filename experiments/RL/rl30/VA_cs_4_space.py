"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED), step 4: where do France content-for-content (CC) swaps sit in the EXISTING
feature space, relative to the labelled V1 CC swaps? READ-ONLY; writes VA_cs_4_space.json only.
France: every same-address SWAP1 pair of the France scoring pool typed CC by A's roles (WD_2_pairs_FR.pkl; accepted AND rejected),
joined to LF + RL-27 test columns. V1: CC swaps from VA_cs_1_v1_flags.npz (A's typing; same street / any address), with y and RL-27 NEW p.
(1) medians of the columns that separate V1 CC negatives from positives; (2) V1 CC y-rate in cells of (name_token_set, rv_rank==1,
competing-owner nf) and the share of France accepted / rejected CC pairs in each cell -- is there ANY labelled cell that says the
France accepted CC pairs are negatives?"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

T0 = time.time()
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
sup = roles[(roles.occ >= 300) & (~roles.legal.astype(bool))]
content = set(sup.index[sup.add_LR < 0.05])
d = pd.read_pickle(os.path.join(OUT, "WD_2_pairs_FR.pkl"))
d = d[(d.kind == "SWAP1") & d.a.isin(content) & d.b.isin(content)].copy()
print("France CC pool pairs", len(d), "accepted", int(d.acc.sum()), flush=True)
COLS = {0: "name_lev", 2: "name_tsort", 3: "name_tset", 4: "name_jac", 5: "s1_name_logfreq", 11: "addr_lev", 14: "addr_tset", 15: "addr_jac",
        75: "tok_idfmax_co", 76: "tok_idfmax_s1o", 78: "tok_frac_co", 77: "tok_frac_s1o"}
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
need = pd.MultiIndex.from_arrays([d.s1.values, d.rec.values])
rows = []
for ch in sorted(glob.glob(os.path.join(ROOT, "experiments/P3/France/chunk_*.npz"))):
    z = np.load(ch); s1c, cc = z["s1"], z["cand"]
    hit = np.flatnonzero(pd.MultiIndex.from_arrays([s1c, cc]).isin(need))
    if not len(hit): continue
    rl = np.load(os.path.join(RL, "test_feats/France", "rl27_" + os.path.basename(ch).replace(".npz", ".npy")), mmap_mode="r")
    LFc = z["LF"][hit]
    f = pd.DataFrame(np.asarray(rl[hit]), columns=RLN)
    for j, nm in COLS.items(): f[nm] = LFc[:, j]
    f["s1"] = s1c[hit]; f["rec"] = cc[hit]; rows.append(f)
F = d.merge(pd.concat(rows).drop_duplicates(["s1", "rec"]), on=["s1", "rec"], how="left")
print("joined", int(F.rv_rank.notna().sum()), f"{time.time() - T0:.0f}s", flush=True)

# V1
m = np.load(PATHS["v1_meta"]); y = m["y"].astype(int); p = np.load(PATHS["v1_p_new"])
fl = np.load(os.path.join(OUT, "VA_cs_1_v1_flags.npz"))
cc_any = fl["SUB1"] & fl["zc"] & fl["xc"]; cc_st = cc_any & fl["ST"]
RL27 = np.load(PATHS["v1_rl27"]); LF = np.load(PATHS["v1_LF"], mmap_mode="r")
idx = np.flatnonzero(cc_any)
Vd = pd.DataFrame(RL27[idx], columns=RLN)
LFi = np.asarray(LF[idx])
for j, nm in COLS.items(): Vd[nm] = LFi[:, j]
Vd["y"] = y[idx]; Vd["p"] = p[idx]; Vd["st"] = cc_st[idx]; Vd["country"] = m["country"][m["s1idx"][idx]]
# V1 same-address analogue must match France's pool definition (same akey or same number + street); use st flag
R = {}
med_cols = list(COLS.values()) + ["rv_rank", "rv_gap", "nf_n", "af_a", "coloc"]
grp = {"FR_CC_accepted": F[F.acc], "FR_CC_rejected": F[~F.acc], "V1_CC_st_neg": Vd[Vd.st & (Vd.y == 0)], "V1_CC_st_pos": Vd[Vd.st & (Vd.y == 1)],
       "V1_CC_any_neg": Vd[~Vd.st & (Vd.y == 0)], "V1_CC_any_pos": Vd[Vd.y == 1]}
R["medians"] = {g: dict(n=len(x), **{c: round(float(x[c].median()), 3) for c in med_cols},
                        rv_rank1=round(float((x.rv_rank == 1).mean()), 3), comp_owner=round(float(((x.nf_a == 0) & (x.nf_n >= 1)).mean()), 3))
                for g, x in grp.items()}
for g, v in R["medians"].items(): print(g, v, flush=True)


def cell(x):
    tset = pd.cut(x.name_tset, [-1, 0.6, 0.75, 0.85, 0.95, 2], labels=["<.6", ".6-.75", ".75-.85", ".85-.95", ">=.95"]).astype(str)
    rv1 = np.where(x.rv_rank == 1, "rv1", "rv>1")
    co = np.where((x.nf_a == 0) & (x.nf_n >= 1), "comp", "nocomp")
    ad = np.where(x.addr_tset >= 0.9, "addr>=.9", "addr<.9")
    return pd.Series([f"{a}|{b}|{c}|{e}" for a, b, c, e in zip(tset, rv1, co, ad)], index=x.index)


F["cell"] = cell(F); Vd["cell"] = cell(Vd)
tab = []
for c in sorted(set(F.cell) | set(Vd.cell)):
    fa = F[(F.cell == c) & F.acc]; fr_ = F[(F.cell == c) & ~F.acc]; v = Vd[Vd.cell == c]; vs = v[v.st]
    tab.append(dict(cell=c, FR_acc=len(fa), FR_rej=len(fr_), FR_acc_rate=round(len(fa) / max(len(fa) + len(fr_), 1), 3),
                    V1_n=len(v), V1_pos=int(v.y.sum()), V1_acc=int((v.p >= NEW_TH).sum()), V1_st_n=len(vs), V1_st_pos=int(vs.y.sum())))
tab = sorted(tab, key=lambda r: -r["FR_acc"])
R["cells"] = tab
for r in tab[:20]: print(r, flush=True)
# share of France accepted CC pairs that fall in cells where V1 has >= 5 labelled CC pairs, and the V1 y-rate there
fa_tot = int(F.acc.sum())
cov = [r for r in tab if r["V1_n"] >= 5]
R["FR_acc_in_cells_with_V1_ge5"] = dict(n=sum(r["FR_acc"] for r in cov), share=round(sum(r["FR_acc"] for r in cov) / fa_tot, 4),
                                        V1_n=sum(r["V1_n"] for r in cov), V1_pos=sum(r["V1_pos"] for r in cov),
                                        FR_acc_weighted_V1_yrate=round(sum(r["FR_acc"] * r["V1_pos"] / r["V1_n"] for r in cov) / max(sum(r["FR_acc"] for r in cov), 1), 4))
covst = [r for r in tab if r["V1_st_n"] >= 5]
R["FR_acc_in_cells_with_V1_samestreet_ge5"] = dict(n=sum(r["FR_acc"] for r in covst), share=round(sum(r["FR_acc"] for r in covst) / fa_tot, 4),
                                                   V1_st_n=sum(r["V1_st_n"] for r in covst), V1_st_pos=sum(r["V1_st_pos"] for r in covst))
print("coverage", R["FR_acc_in_cells_with_V1_ge5"], R["FR_acc_in_cells_with_V1_samestreet_ge5"], flush=True)
# V1 CC pairs in the France-accepted signature (tset >= .75, rv1, nocomp, addr >= .9): y, p, examples
sig = (Vd.name_tset >= 0.75) & (Vd.rv_rank == 1) & ~((Vd.nf_a == 0) & (Vd.nf_n >= 1)) & (Vd.addr_tset >= 0.9)
fsig = (F.name_tset >= 0.75) & (F.rv_rank == 1) & ~((F.nf_a == 0) & (F.nf_n >= 1)) & (F.addr_tset >= 0.9)
R["france_like_signature"] = dict(V1_n=int(sig.sum()), V1_pos=int(Vd.y[sig].sum()), V1_acc=int((Vd.p[sig] >= NEW_TH).sum()),
                                  V1_p=np.round(np.sort(Vd.p[sig].values), 4).tolist()[:60],
                                  FR_n=int(fsig.sum()), FR_acc=int((F.acc & fsig).sum()), FR_acc_share_of_all_acc=round(float((F.acc & fsig).sum() / fa_tot), 4))
print("signature", R["france_like_signature"], flush=True)
T = load("train", verbose=False); S1t = T["s1"].set_index("id"); RECt = pd.concat([T["s2"], T["s3"]]).set_index("id")
ii = idx[sig.values]
R["france_like_signature_V1_examples"] = [dict(s1=S1t.name.get(m["s1_ids"][m["s1idx"][k]]), rec=RECt.name.get(m["cand"][k]),
                                               s1_addr=str(S1t.addr.get(m["s1_ids"][m["s1idx"][k]]))[:60], rec_addr=str(RECt.addr.get(m["cand"][k]))[:60],
                                               y=int(y[k]), p=round(float(p[k]), 4)) for k in ii[:30]]
R["secs"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(OUT, "VA_cs_4_space.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("done", f"{time.time() - T0:.0f}s")
