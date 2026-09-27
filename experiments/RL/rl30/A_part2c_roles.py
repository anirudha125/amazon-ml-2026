"""RL-30 investigator A, PART 2c (READ-ONLY). Merge Part 1 (co-located contrast) and Part 2 (noise rates) into token ROLES,
validate the pseudo-positive noise estimate against label-backed GT noise (US/India), compare role with IDF, and measure on V1
(labels, US/India) whether role separates FP from TP among accepted pairs within IDF bins.
Role definitions (France, per token; thresholds fixed a priori, not tuned):
  add_LR  = (pure record-only adds per pseudo-positive pair) / (share of France S1 names containing the token)
            ~ how often the generator ADDS the token to a true record relative to how often a random other business has it.
  drop    = pure drop rate given the S1 has the token.
  C  insufficient: occ_PP < 300 (pseudo-positive occurrences) or co-located one+two < 30
  A  filler/noise-like: add_LR >= 0.05 or drop >= 0.20
  B  distinguishing: add_LR < 0.01 and drop < 0.20 and co-located one-sided >= 30 in >= 20 groups
  M  intermediate: everything else with support (0.01 <= add_LR < 0.05)
Outputs: A_roles_France.csv, A_p2c_summary.json
"""
import os, sys, json, time, re, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from scipy.stats import spearmanr
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def rd(name):
    return pd.read_csv(os.path.join(OUT, name), keep_default_na=False, na_values=[""])


S = {}
D = load("test", verbose=False)
s1 = D["s1"]
df = {}
for C in ("France", "US", "India"):
    df[C] = Counter(t for n in s1.name.values[s1.country.values == C] for t in ntoks(n))
NC = s1.country.value_counts().to_dict()
pooled = Counter()
for C in df: pooled.update(df[C])
Npool = len(s1)

# ---------- validation of the pseudo-positive noise estimate (label-backed, US / India)
val = {}
for C in ("US", "India"):
    pp = rd(f"A_p2_noise_{C}_testPP.csv").set_index("tok"); gt = rd(f"A_p2_noise_{C}_trainGT.csv").set_index("tok")
    v1 = rd(f"A_p2_noise_{C}_V1pos.csv").set_index("tok"); v9 = rd(f"A_p2_noise_{C}_V1pos_p99.csv").set_index("tok")
    npp = json.load(open(os.path.join(OUT, "A_p2_summary.json")))[f"{C}_testPP"]["pairs"]
    ngt = json.load(open(os.path.join(OUT, "A_p2_summary.json")))[f"{C}_trainGT"]["pairs"]
    j = pp.join(gt, lsuffix="_pp", rsuffix="_gt", how="inner")
    j = j[(j.occ_pp >= 300) & (j.occ_gt >= 300)]
    j["addpp_pp"] = j.padd_pp / npp; j["addpp_gt"] = j.padd_gt / ngt
    jv = v1.join(v9, lsuffix="_all", rsuffix="_p99", how="inner"); jv = jv[(jv.occ_all >= 200) & (jv.occ_p99 >= 100)]
    val[C] = dict(n_tok=len(j),
                  spearman_drop=float(spearmanr(j.drop_rate_pp, j.drop_rate_gt, nan_policy="omit").correlation),
                  spearman_addrel=float(spearmanr(j.add_rel_pp, j.add_rel_gt, nan_policy="omit").correlation),
                  spearman_add_per_pair=float(spearmanr(j.addpp_pp, j.addpp_gt).correlation),
                  median_drop_pp=float(j.drop_rate_pp.median()), median_drop_gt=float(j.drop_rate_gt.median()),
                  median_ratio_drop_pp_over_gt=float((j.drop_rate_pp / j.drop_rate_gt.clip(lower=1e-4)).median()),
                  v1_n_tok=len(jv), v1_spearman_drop_all_vs_p99=float(spearmanr(jv.drop_rate_all, jv.drop_rate_p99, nan_policy="omit").correlation),
                  v1_median_ratio_drop_p99_over_all=float((jv.drop_rate_p99 / jv.drop_rate_all.clip(lower=1e-4)).median()),
                  top_adders_pp=[(t, int(r.padd_pp), round(float(r.add_rel_pp), 3), round(float(r.add_rel_gt), 3)) for t, r in
                                 j.sort_values("padd_pp", ascending=False).head(15).iterrows()])
    print(C, json.dumps(val[C])[:1500], flush=True)
S["validation_pp_vs_gt"] = val

# ---------- France roles
p1 = rd("A_p1_tokens_France.csv").set_index("tok")
pp = rd("A_p2_noise_France_testPP.csv").set_index("tok")
lo = rd("A_p2_noise_France_testLO.csv").set_index("tok")
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
npp = summ2["France_testPP"]["pairs"]; nlo = summ2["France_testLO"]["pairs"]
ppu = rd("A_p2_noise_US_testPP.csv").set_index("tok"); ppi = rd("A_p2_noise_India_testPP.csv").set_index("tok")
nus = summ2["US_testPP"]["pairs"]; nin = summ2["India_testPP"]["pairs"]
R = pp[["legal", "s1has", "rechas", "both", "pdrop", "padd", "typo", "occ", "noise_pure", "drop_rate", "add_rel"]].copy()
R["df_FR"] = [df["France"].get(t, 0) for t in R.index]
R["df_US"] = [df["US"].get(t, 0) for t in R.index]
R["df_IN"] = [df["India"].get(t, 0) for t in R.index]
R["per10k_FR"] = 1e4 * R.df_FR / NC["France"]; R["per10k_US"] = 1e4 * R.df_US / NC["US"]; R["per10k_IN"] = 1e4 * R.df_IN / NC["India"]
R["idf_pooled"] = [math.log((Npool - pooled.get(t, 0) + 0.5) / (pooled.get(t, 0) + 0.5) + 1.0) for t in R.index]
R["add_per_pair"] = R.padd / npp
R["add_LR"] = R.add_per_pair / (R.df_FR.clip(lower=1) / NC["France"])
for c in ["two", "one", "share", "null_share", "share_lift", "n_groups_1s", "n_s1_1s", "onlydiff", "swap", "contrast", "null_contrast",
          "nd_one", "nd_after", "core_onlydiff", "core_swap", "jac_with", "jac_without", "df_grouped"]:
    R["co_" + c] = p1[c].reindex(R.index)
R[["co_two", "co_one"]] = R[["co_two", "co_one"]].fillna(0)
R["lo_drop_rate"] = lo.drop_rate.reindex(R.index); R["lo_padd"] = lo.padd.reindex(R.index).fillna(0); R["lo_pdrop"] = lo.pdrop.reindex(R.index).fillna(0)
R["lo_add_per_pair"] = R.lo_padd / nlo
R["US_drop"] = ppu.drop_rate.reindex(R.index); R["US_add_per_pair"] = ppu.padd.reindex(R.index) / nus; R["US_occ"] = ppu.occ.reindex(R.index)
R["IN_drop"] = ppi.drop_rate.reindex(R.index); R["IN_add_per_pair"] = ppi.padd.reindex(R.index) / nin; R["IN_occ"] = ppi.occ.reindex(R.index)


def role(r):
    # C = insufficient pseudo-positive support; co-located support is required only for B
    # (record-only generator tokens such as 'associes' never occur in S1 names, so they have no co-located support by construction)
    if r.occ < 300:
        return "C"
    if r.add_LR >= 0.05 or (r.drop_rate == r.drop_rate and r.drop_rate >= 0.20):
        return "A"
    if r.add_LR < 0.01:
        return "B" if (r.co_one >= 30 and (r.co_n_groups_1s if r.co_n_groups_1s == r.co_n_groups_1s else 0) >= 20) else "B_weak"
    return "M"


R["role"] = [role(r) for r in R.itertuples()]
R = R.sort_values("occ", ascending=False)
R.to_csv(os.path.join(OUT, "A_roles_France.csv"))
sup = R[R.role != "C"]
supc = sup[sup.co_one + sup.co_two >= 100]
S["France_share_lift_quantiles_colocsupport>=100"] = supc.co_share_lift.dropna().quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
S["France_n_tokens_colocsupport>=100"] = int(len(supc))
S["France_role_counts"] = R.role.value_counts().to_dict()
S["France_role_occ_share"] = (R.groupby("role").occ.sum() / R.occ.sum()).round(4).to_dict()
S["France_spearman_addLR_vs_idf"] = float(spearmanr(sup.add_LR, sup.idf_pooled).correlation)
S["France_spearman_drop_vs_idf"] = float(spearmanr(sup.drop_rate, sup.idf_pooled).correlation)
S["France_spearman_colocshare_vs_df"] = float(spearmanr(supc.co_share, supc.df_FR, nan_policy="omit").correlation)
S["France_share_lift_quantiles_supported"] = sup.co_share_lift.dropna().quantile([.05, .25, .5, .75, .95]).round(3).to_dict()
cols = ["role", "legal", "occ", "drop_rate", "add_per_pair", "add_LR", "idf_pooled", "per10k_FR", "per10k_US", "per10k_IN", "co_one", "co_two",
        "co_share_lift", "co_contrast", "co_null_contrast", "co_nd_one", "co_core_swap", "lo_drop_rate", "lo_add_per_pair", "US_drop", "US_add_per_pair", "IN_drop", "IN_add_per_pair"]
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(R[R.role == "A"].sort_values("padd", ascending=False)[cols].head(45).round(4).to_string())
print(R[R.role == "B"].sort_values("co_one", ascending=False)[cols].head(40).round(4).to_string())
print(R[R.role == "M"].sort_values("co_one", ascending=False)[cols].head(25).round(4).to_string())
S["top_A"] = R[R.role == "A"].sort_values("padd", ascending=False)[cols].head(30).round(4).reset_index().to_dict("records")
S["top_B"] = R[R.role == "B"].sort_values("co_one", ascending=False)[cols].head(30).round(4).reset_index().to_dict("records")
S["top_M"] = R[R.role == "M"].sort_values("co_one", ascending=False)[cols].head(20).round(4).reset_index().to_dict("records")

# ---------- V1 label check (US/India): does role separate FP among accepted pairs, within IDF bins?
V = pd.read_pickle(os.path.join(OUT, "A_p2b_V1_pairs.pkl"))
DT = load("train", verbose=False)
dft = Counter(t for n in DT["s1"].name.values for t in ntoks(n)); Nt = len(DT["s1"])
idf_t = lambda t: math.log((Nt - dft.get(t, 0) + 0.5) / (dft.get(t, 0) + 0.5) + 1.0)
v1res = {}
for C, tab, npair in (("US", ppu, nus), ("India", ppi, nin)):
    NC_tr = int((DT["s1"].country == C).sum())
    dfc = Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in ntoks(n))
    addLR = (tab.padd / npair) / (pd.Series({t: dfc.get(t, 0) for t in tab.index}).clip(lower=1) / NC_tr)
    occ = tab.occ; dr = tab.drop_rate
    v = V[(V.country == C) & (V.p >= NEW_TH) & (V.n_reconly > 0)].copy()

    def cls(s):
        ts = s.split()
        ok = [t for t in ts if t in occ.index and occ[t] >= 300]
        if len(ok) < len(ts):
            return "unknown"
        return "filler_only" if all(addLR[t] >= 0.05 for t in ts) else ("has_distinguishing" if any(addLR[t] < 0.01 for t in ts) else "intermediate")

    v["rcls"] = v.reconly.map(cls)
    v["idfmax"] = v.reconly.map(lambda s: max(idf_t(t) for t in s.split()))
    v["idfbin"] = pd.qcut(v.idfmax, 4, labels=False, duplicates="drop")
    out = {k: dict(n=int(len(g)), fp=int((g.y == 0).sum()), fp_rate=round(float(1 - g.y.mean()), 4)) for k, g in v.groupby("rcls")}
    out["by_idf_quartile"] = {str(b): {k: dict(n=int(len(g)), fp_rate=round(float(1 - g.y.mean()), 4)) for k, g in gb.groupby("rcls")}
                              for b, gb in v.groupby("idfbin")}
    # S1-only side: high-drop (>=0.20) vs low-drop tokens
    w = V[(V.country == C) & (V.p >= NEW_TH) & (V.n_s1only > 0) & (V.n_reconly == 0)].copy()
    w["scls"] = w.s1only.map(lambda s: "unknown" if any(t not in occ.index or occ[t] < 300 for t in s.split())
                             else ("all_high_drop" if all(dr[t] >= 0.20 for t in s.split()) else "has_low_drop"))
    out["s1only_side"] = {k: dict(n=int(len(g)), fp_rate=round(float(1 - g.y.mean()), 4)) for k, g in w.groupby("scls")}
    base = V[(V.country == C) & (V.p >= NEW_TH)]
    out["all_accepted"] = dict(n=int(len(base)), fp_rate=round(float(1 - base.y.mean()), 4))
    v1res[C] = out
    print(C, "V1 role check", json.dumps(out), flush=True)
S["V1_role_check"] = v1res

# ---------- France volume: accepted S005 France pairs carrying a record-only B-token / only A-tokens
roleD = R.role.to_dict()
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a = accepted("S005_France"); a = a[a.kept_final]
S1i = s1.set_index("id")
s1n = S1i.name.reindex(a.s1.values).values; rn = REC.name.reindex(a.rec.values).values
cnt = Counter(); exB = []
for k in range(len(a)):
    A_, B_ = ntoks(s1n[k]), ntoks(rn[k] if isinstance(rn[k], str) else "")
    pb = set(B_ - A_); pa = set(A_ - B_)
    for _, x, y_ in sorted(((LV.normalized_similarity(x, y_), x, y_) for x in pa for y_ in pb if (LV.distance(x, y_) <= 1 or LV.normalized_similarity(x, y_) >= 0.75)), reverse=True):
        if x in pa and y_ in pb: pa.discard(x); pb.discard(y_)
    band = "HI" if a.p.values[k] >= 0.99 else "LO"
    cnt[f"{band}_n"] += 1
    rB = [t for t in pb if roleD.get(t) == "B"]; sB = [t for t in pa if roleD.get(t) == "B"]
    if rB: cnt[f"{band}_rec_only_B"] += 1
    if pb and all(roleD.get(t) == "A" for t in pb): cnt[f"{band}_rec_only_all_A"] += 1
    if rB and sB: cnt[f"{band}_B_for_B_swap"] += 1
    if rB and sB and len(exB) < 40 and k % 7 == 0:
        exB.append(dict(s1=s1n[k], rec=rn[k], p=round(float(a.p.values[k]), 4), n_claims=int(a.n_claims.values[k])))
S["France_accepted_role_volume"] = dict(cnt)
S["France_B_for_B_swap_examples"] = exB
print(json.dumps(dict(cnt)), flush=True)
json.dump(S, open(os.path.join(OUT, "A_p2c_summary.json"), "w"), indent=1, default=str, ensure_ascii=False)
print("done", f"{time.time() - T0:.0f}s")
