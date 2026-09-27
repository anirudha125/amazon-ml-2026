"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED), claim = A's content-for-content substitution flag. Step 2: France test (no labels).
READ-ONLY; writes VA_cs_2_france.json only.
(a) reproduce A's France flag exactly (A_part2h: content = A_roles_France add_LR < 0.05 & occ >= 300 & not legal; one-for-one core
    substitution after typo pairing; S005 kept_final) and split it by address relation (same number+street / other);
(b) join the RL-27 test columns (rv_rank, nf_n, nf_a, ...) and LF TOK16 columns for flagged pairs and for two controls
    (random accepted unflagged pairs; one-for-one substitutions whose record-only word is generic/filler) and compare with the V1
    flagged negatives/positives of VA_cs_1_v1.json: do the France flagged pairs carry the existing-feature signature that makes RL-27 NEW
    reject the V1 negatives (competing owner), or do they look like V1 positives on every existing column?
(c) S004 vs S005: how many flagged pairs entered/left the final France set, and what France-score change that implies under
    'all flagged are false' vs the LB-inferred France change (-0.06 pp, range -0.4..+0.3)."""
import os, sys, json, time, re, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s); return frozenset(re.findall(r"[a-z0-9]+", s))


def core(A): return frozenset(t for t in A if t not in LEGAL and t not in HONOR)


def typo(a, b): return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
sup = roles[(roles.occ >= 300) & (~roles.legal.astype(bool))]
content = set(sup.index[sup.add_LR < 0.05]); generic = set(sup.index[sup.add_LR >= 0.05])
D = load("test", verbose=False)
fr = D["s1"][D["s1"].country == "France"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a5 = accepted("S005_France"); a4 = accepted("S004_France")
# max-claimer reconstruction check on S005, then apply to S004
k5 = a5.sort_values("p", ascending=False).drop_duplicates("rec")
kf5 = set(zip(k5.s1, k5.rec)); chk = np.array([(s, r) in kf5 for s, r in zip(a5.s1, a5.rec)])
mc_ok = float((chk == a5.kept_final.values).mean())
k4 = a4.sort_values("p", ascending=False).drop_duplicates("rec"); kf4 = set(zip(k4.s1, k4.rec))
print("max-claimer reconstruction agreement on S005", mc_ok, "S004 final", len(kf4), "S005 final", int(a5.kept_final.sum()), flush=True)

allp = pd.concat([a5[["s1", "rec"]], a4[["s1", "rec"]]]).drop_duplicates()
s1n = fr.name.reindex(allp.s1.values).values; s1a = fr.addr.reindex(allp.s1.values).values
rn = REC.name.reindex(allp.rec.values).values; ra = REC.addr.reindex(allp.rec.values).values
sc = {}


def sp(x):
    if not isinstance(x, str): x = ""
    v = sc.get(x)
    if v is None: v = sc[x] = street_parts(x)[:2]
    return v


typ = np.full(len(allp), "", object); XT = np.full(len(allp), "", object); ZT = np.full(len(allp), "", object); ST = np.zeros(len(allp), bool)
for k in range(len(allp)):
    if not isinstance(rn[k], str): continue
    A_, B_ = ntoks(s1n[k]), ntoks(rn[k])
    pa, pb = set(core(A_ - B_)), set(core(B_ - A_))
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    if len(pa) == 1 and len(pb) == 1:
        x, z = next(iter(pa)), next(iter(pb)); XT[k] = x; ZT[k] = z
        if x in content and z in content: typ[k] = "CC"
        elif z in generic: typ[k] = "SUB_Z_GENERIC"
        elif z in content: typ[k] = "SUB_Z_CONTENT_X_OTHER"
        else: typ[k] = "SUB_OTHER"
        n1, st1 = sp(s1a[k]); n2, st2 = sp(ra[k]); ST[k] = n1 is not None and n1 == n2 and bool(st1) and st1 == st2
allp = allp.assign(typ=typ, x=XT, z=ZT, st=ST)
key = lambda d: pd.MultiIndex.from_arrays([d.s1.values, d.rec.values])
a5f = a5[a5.kept_final].merge(allp, on=["s1", "rec"], how="left")
R = {"mc_reconstruction_agreement_S005": mc_ok}
cc = a5f[a5f.typ == "CC"]
R["A_flag_reproduced"] = dict(n=len(cc), n_s1=int(cc.s1.nunique()), HI=int((cc.p >= 0.99).sum()), LO=int((cc.p < 0.99).sum()),
                              same_street=int(cc.st.sum()), same_street_HI=int((cc.st & (cc.p >= 0.99)).sum()), same_street_LO=int((cc.st & (cc.p < 0.99)).sum()),
                              n_claims_ge2=int((cc.n_claims >= 2).sum()))
print("A flag reproduced", R["A_flag_reproduced"], f"{time.time() - T0:.0f}s", flush=True)
# address relation for CC pairs
rel = Counter()
for s, r in zip(cc.s1.values, cc.rec.values):
    x, y_ = fr.addr.get(s), REC.addr.get(r)
    if isinstance(x, str) and isinstance(y_, str) and akey(x) == akey(y_): rel["exact_akey"] += 1
    else:
        n1, st1 = sp(x); n2, st2 = sp(y_)
        rel["same_num_street" if (n1 is not None and n1 == n2 and st1 and st1 == st2) else ("same_num_other_street" if n1 is not None and n1 == n2 else "diff_num_or_none")] += 1
R["CC_address_relation"] = dict(rel)

# ---- S004 vs S005 France final: flagged pairs entering/leaving
fin4 = pd.DataFrame(list(kf4), columns=["s1", "rec"]).merge(allp, on=["s1", "rec"], how="left")
f4 = set(zip(fin4.s1[fin4.typ == "CC"], fin4.rec[fin4.typ == "CC"])); f5 = set(zip(cc.s1, cc.rec))
only4 = f4 - f5; only5 = f5 - f4
tot4 = fin4.groupby("s1").size(); tot5 = a5f.groupby("s1").size()


def f05(P, Rr): return 0.0 if P + Rr == 0 else 1.25 * P * Rr / (0.25 * P + Rr)


def delta_if_false(pairs, tot):
    """sum over S1 of [F(without flagged pairs) - F(with)] assuming the S1's other predicted records are all true and complete."""
    g = Counter(s for s, _ in pairs); d = 0.0
    for s, kf in g.items():
        m = int(tot.get(s, kf)) - kf
        d += 1.0 - (f05(m / (m + kf), 1.0) if m > 0 else 0.0)
    return d


def loss_if_true_missing(pairs, tot_pred):
    """sum over S1 of [1 - F] when the S1's kf flagged true records are missing and its m predicted records are all true."""
    g = Counter(s for s, _ in pairs); d = 0.0
    for s, kf in g.items():
        m = int(tot_pred.get(s, 0))
        d += 1.0 - (f05(1.0, m / (m + kf)) if m > 0 else 0.0)
    return d


NFR = len(fr)
g5 = delta_if_false(only5, tot5); g4 = delta_if_false(only4, tot4)
lt = loss_if_true_missing(only5, tot4) - loss_if_true_missing(only4, tot5)
R["S004_vs_S005"] = dict(S004_final=len(kf4), S005_final=int(a5.kept_final.sum()), CC_in_S004=len(f4), CC_in_S005=len(f5), CC_only_S004=len(only4), CC_only_S005=len(only5),
                         France_pp_change_S004_to_S005_if_all_CC_false=round(100 * (g4 - g5) / NFR, 3),
                         France_pp_change_S004_to_S005_if_all_CC_true=round(100 * lt / NFR, 3),
                         note="ESTIMATE: other predictions assumed correct; LB-inferred France change S004->S005 = -0.06 pp (range -0.4..+0.3)")
print("S004 vs S005", R["S004_vs_S005"], flush=True)

# ---- controls and feature join
rng = np.random.default_rng(0)
ctrl_rand = a5f[a5f.typ != "CC"].sample(40000, random_state=0)
ctrl_gen = a5f[(a5f.typ == "SUB_Z_GENERIC")]
ctrl_gen = ctrl_gen.sample(min(40000, len(ctrl_gen)), random_state=0)
need = pd.concat([cc.assign(grp="CC"), ctrl_rand.assign(grp="CTRL_RAND"), ctrl_gen.assign(grp="CTRL_ZGEN")])
need = need.drop_duplicates(["s1", "rec", "grp"])
needk = set(zip(need.s1, need.rec))
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
LFJ = {3: "x22_3", 71: "tok_n_s1o", 72: "tok_n_co", 73: "tok_idf_s1o", 74: "tok_idf_co", 75: "tok_idfmax_co", 76: "tok_idfmax_s1o",
       77: "tok_frac_s1o", 78: "tok_frac_co", 81: "tok_soft_idf_s1o", 82: "tok_soft_idf_co"}
rows = []
needk_idx = pd.MultiIndex.from_tuples(list(needk))
chunks = sorted(glob.glob(os.path.join(ROOT, "experiments/P3/France/chunk_*.npz")))
for ci, ch in enumerate(chunks):
    z = np.load(ch)
    s1c, cc_ = z["s1"], z["cand"]
    kk = pd.MultiIndex.from_arrays([s1c, cc_])
    hit = np.flatnonzero(kk.isin(needk_idx))
    if len(hit):
        rl = np.load(os.path.join(RL, "test_feats/France", f"rl27_{os.path.basename(ch).replace('.npz', '.npy')}"), mmap_mode="r")
        assert rl.shape[0] == len(s1c)
        LFc = z["LF"][hit]
        d = pd.DataFrame(np.asarray(rl[hit]), columns=RLN)
        for j, nm in LFJ.items(): d[nm] = LFc[:, j]
        d["s1"] = s1c[hit]; d["rec"] = cc_[hit]; d["rank_dense"] = z["rank_dense"][hit]
        rows.append(d)
    if ci % 40 == 0: print("chunk", ci, len(hit), f"{time.time() - T0:.0f}s", flush=True)
F = pd.concat(rows).drop_duplicates(["s1", "rec"])
J = need.merge(F, on=["s1", "rec"], how="left")
R["join_coverage"] = J.groupby("grp").rv_rank.apply(lambda s: round(float(s.notna().mean()), 4)).to_dict()


def summ(g):
    out = dict(n=len(g), p_mean=round(float(g.p.mean()), 4), p_ge99=round(float((g.p >= 0.99).mean()), 4),
               rv_rank1=round(float((g.rv_rank == 1).mean()), 4), rv_rank_gt1=round(float((g.rv_rank > 1).mean()), 4),
               competing_owner_nf=round(float(((g.nf_a == 0) & (g.nf_n >= 1)).mean()), 4),
               nf_n_eq0=round(float((g.nf_n == 0).mean()), 4), nf_a1=round(float((g.nf_a == 1).mean()), 4),
               coloc_ge1=round(float((g.coloc >= 1).mean()), 4), n_claims_ge2=round(float((g.n_claims >= 2).mean()), 4))
    for c in ["rv_gap", "rv_so", "nf_n", "dupf", "tok_idfmax_co", "tok_idf_co", "tok_frac_co", "tok_idf_s1o", "x22_3"]:
        out[c + "_median"] = round(float(g[c].median()), 4)
    return out


R["features_by_group"] = {}
for grp, g in J.groupby("grp"):
    R["features_by_group"][grp] = summ(g)
    if grp == "CC":
        for band, gg in (("HI", g[g.p >= 0.99]), ("LO", g[g.p < 0.99])):
            R["features_by_group"][f"CC_{band}"] = summ(gg)
        R["features_by_group"]["CC_same_street"] = summ(g[g.st])
print(json.dumps(R["features_by_group"], indent=0), flush=True)
# where do the France CC pairs fall on the RL-27 'competing owner' axes that reject V1 negatives?
ccj = J[J.grp == "CC"]
R["CC_rv_rank_dist"] = ccj.rv_rank.value_counts(normalize=True).round(4).sort_index().to_dict()
R["CC_examples_competing_owner"] = [dict(s1=fr.name.get(s), s1_addr=str(fr.addr.get(s))[:60], rec=REC.name.get(r), rec_addr=str(REC.addr.get(r))[:60],
                                         p=round(float(p_), 4), rv_rank=float(rr), nf_n=float(nn), nf_a=float(na))
                                    for s, r, p_, rr, nn, na in ccj[(ccj.rv_rank > 1) | ((ccj.nf_a == 0) & (ccj.nf_n >= 1))]
                                    .sample(min(12, len(ccj)), random_state=1)[["s1", "rec", "p", "rv_rank", "nf_n", "nf_a"]].values]
R["CC_examples_rv1"] = [dict(s1=fr.name.get(s), rec=REC.name.get(r), p=round(float(p_), 4), rv_gap=round(float(gp), 3), nf_n=float(nn))
                        for s, r, p_, gp, nn in ccj[ccj.rv_rank == 1].sample(12, random_state=2)[["s1", "rec", "p", "rv_gap", "nf_n"]].values]
R["top_swaps_CC"] = Counter(f"{x}->{z}" for x, z in zip(cc.x, cc.z)).most_common(25)
R["secs"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(OUT, "VA_cs_2_france.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("done", f"{time.time() - T0:.0f}s")
