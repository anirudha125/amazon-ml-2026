"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED / NOT INCREMENTAL), claim = Investigator A's
'distinguishing content-word substitution' signal. Step 1: labelled V1 (US/India). READ-ONLY; writes VA_cs_1_v1.json only.
Flag = A's own definition (A_part2j / A_part2h): name-token sets differ by <= 4 tokens, after typo pairing exactly one core S1-only token x
and exactly one core record-only token z; kind(t) from A_p2_noise_<C>_testPP.csv (occ >= 300; add_LR >= 0.05 generic else content;
occ < 300 -> rare). Variants:
  IN_C_ST  z content, same house number + same street-name tokens         (A's V1 evidence: India 139/143, US 41/58 negative)
  CC_ST    x and z content, same street                                    (France definition, street-restricted)
  CC_ANY   x and z content, any address                                    (A's France count in A_part2h has NO street condition)
  IN_C_ANY z content, any address
Questions: (1) is RL-27 NEW already right on these pairs (confusion at 0.78, oracle macro delta)? (2) does the flag change y-rate at fixed p?
(3) WHY are the V1 negatives negative: does the record belong to another train S1 (competing owner, captured by RL-27 nf/rv + max-claimer)
or is it an orphan (the France-like decoy population)? (4) single-column AUC of existing LF / RL-27 columns within the flag."""
import os, sys, json, time, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV
from sklearn.metrics import roc_auc_score

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def core(A): return frozenset(t for t in A if t not in LEGAL and t not in HONOR)


def typo(a, b): return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
NCt = DT["s1"].country.value_counts().to_dict()
KIND = {}
for C in ("US", "India"):
    dfc = Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in ntoks(n))
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    lr = (t.padd / summ2[f"{C}_testPP"]["pairs"]) / (pd.Series({k: dfc.get(k, 0) for k in t.index}).clip(lower=1) / NCt[C])
    KIND[C] = lr[t.occ >= 300].to_dict()
kind = lambda C, w: "rare" if w not in KIND[C] else ("generic" if KIND[C][w] >= 0.05 else "content")

V = S2.load_set("V1", "a50n10d10a"); V["country_s1"] = V["country"]
p = np.load(PATHS["v1_p_new"]); y = V["y"].astype(int); th = NEW_TH
RL27 = np.load(PATHS["v1_rl27"]); LF = np.load(PATHS["v1_LF"], mmap_mode="r")
base = S2.summarize(V, p, th)
print("baseline V1 macro", round(base["macro"] * 100, 3), "TP", base["tp"], "FP", base["fp"], flush=True)
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]; n = len(y)
s1n = S1t.name.reindex(s1ids).values; s1a = S1t.addr.reindex(s1ids).values
rn = RECt.name.reindex(cand).values; ra = RECt.addr.reindex(cand).values
nc, sc = {}, {}


def nt(s):
    if not isinstance(s, str): s = ""
    v = nc.get(s)
    if v is None: v = nc[s] = ntoks(s)
    return v


def sp(a):
    if not isinstance(a, str): a = ""
    v = sc.get(a)
    if v is None: v = sc[a] = street_parts(a)[:2]
    return v


SUB1 = np.zeros(n, bool); ST = np.zeros(n, bool); KX = np.empty(n, object); KZ = np.empty(n, object); XT = np.empty(n, object); ZT = np.empty(n, object)
for k in range(n):
    if not isinstance(rn[k], str): continue
    A, B = nt(s1n[k]), nt(rn[k])
    if len(A ^ B) > 4 or not (A ^ B): continue
    pa, pb = set(core(A - B)), set(core(B - A))
    if not pa or not pb: continue
    for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
        if x in pa and z in pb: pa.discard(x); pb.discard(z)
    if not (len(pa) == 1 and len(pb) == 1): continue
    x, z = next(iter(pa)), next(iter(pb))
    SUB1[k] = True; XT[k] = x; ZT[k] = z; KX[k] = kind(ctry[k], x); KZ[k] = kind(ctry[k], z)
    n1, st1 = sp(s1a[k]); n2, st2 = sp(ra[k])
    ST[k] = n1 is not None and n1 == n2 and bool(st1) and st1 == st2
print("SUB1", int(SUB1.sum()), "same-street", int((SUB1 & ST).sum()), f"{time.time() - T0:.0f}s", flush=True)

zc = np.array([v == "content" for v in KZ]); xc = np.array([v == "content" for v in KX])
FLAGS = dict(IN_C_ST=SUB1 & ST & zc, CC_ST=SUB1 & ST & zc & xc, CC_ANY=SUB1 & zc & xc, IN_C_ANY=SUB1 & zc)
acc = p >= th
FN_all = int(((~acc) & (y == 1)).sum()); FP_all = int((acc & (y == 0)).sum())

# ---- ownership of each record in the train ground truth
gt = DT["gt"]
own = gt.groupby("rec").s1.agg(lambda s: tuple(s)).to_dict()
v1set = set(V["s1_ids"])
s1_street = {}


def owner_kind(k):
    o = own.get(cand[k])
    if o is None: return "orphan"
    if s1ids[k] in o: return "own"
    others = [s for s in o if s != s1ids[k]]
    nA, stA = sp(s1a[k])
    same = False
    for s in others:
        if s in S1t.index:
            nB, stB = sp(S1t.addr[s])
            if nA is not None and nA == nB and stA and stA == stB: same = True
    return ("other_S1_same_street" if same else "other_S1_elsewhere") + ("_inV1" if any(s in v1set for s in others) else "_notV1")


def conf(m):
    return dict(n=int(m.sum()), pos=int((y[m] == 1).sum()), neg=int((y[m] == 0).sum()), TP=int((m & acc & (y == 1)).sum()),
                FN=int((m & ~acc & (y == 1)).sum()), FP=int((m & acc & (y == 0)).sum()), TN=int((m & ~acc & (y == 0)).sum()))


def macro_with(new_acc):
    s = S2.summarize(V, np.where(new_acc, 1.0, 0.0), 0.5)
    return round(s["macro"] * 100, 4), s["tp"], s["fp"]


bins = [0, 0.01, 0.05, 0.2, 0.5, th, 0.9, 0.99, 1.0001]
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
TOKN = ["tok_n_s1o", "tok_n_co", "tok_idf_s1o", "tok_idf_co", "tok_idfmax_co", "tok_idfmax_s1o", "tok_frac_s1o", "tok_frac_co",
        "tok_soft_n_s1o", "tok_soft_n_co", "tok_soft_idf_s1o", "tok_soft_idf_co", "tok_a_s1o", "tok_a_co", "tok_a_s1o_f", "tok_a_co_f"]
res = dict(baseline=dict(macro=round(base["macro"] * 100, 3), TP=base["tp"], FP=base["fp"], FN_in_pool=FN_all, FP_all=FP_all),
           n_SUB1=int(SUB1.sum()), n_SUB1_street=int((SUB1 & ST).sum()))
rng = np.random.default_rng(0)
for name, f in FLAGS.items():
    r = {C: conf(f & (ctry == C)) for C in ("US", "India")}
    r["all"] = conf(f)
    r["share_of_all_FN_in_pool"] = round(r["all"]["FN"] / FN_all, 5); r["share_of_all_FP"] = round(r["all"]["FP"] / FP_all, 5)
    b = np.digitize(p, bins) - 1
    r["by_p_bin"] = {f"[{bins[i]:.2f},{bins[i + 1]:.2f})": dict(flag_n=int((f & (b == i)).sum()),
                                                                 flag_y=round(float(y[f & (b == i)].mean()), 3) if (f & (b == i)).any() else None,
                                                                 unflag_y=round(float(y[~f & (b == i)].mean()), 4) if (~f & (b == i)).any() else None)
                     for i in range(len(bins) - 1)}
    orc = acc.copy(); orc[f] = y[f] == 1
    r["oracle_macro"], r["oracle_TP"], r["oracle_FP"] = macro_with(orc)
    r["oracle_delta_pp"] = round(r["oracle_macro"] - base["macro"] * 100, 4)
    veto = acc & ~f
    r["veto_all_flagged_macro"], _, _ = macro_with(veto); r["veto_delta_pp"] = round(r["veto_all_flagged_macro"] - base["macro"] * 100, 4)
    idx = np.flatnonzero(f)
    if len(set(y[idx])) == 2:
        r["auc_p_within_flag"] = round(float(roc_auc_score(y[idx], p[idx])), 4)
        aucs = {}
        for j, nm in enumerate(RLN):
            v = RL27[idx, j]; ok = ~np.isnan(v)
            if ok.sum() > 10 and len(set(y[idx][ok])) == 2: aucs[nm] = round(float(roc_auc_score(y[idx][ok], v[ok])), 3)
        LFi = np.asarray(LF[idx])
        for j in range(87):
            v = LFi[:, j]; ok = ~np.isnan(v)
            if ok.sum() > 10 and len(set(y[idx][ok])) == 2:
                a_ = float(roc_auc_score(y[idx][ok], v[ok])); aucs[f"LF{j}" + (f"_{TOKN[j - 71]}" if j >= 71 else "")] = round(a_, 3)
        r["single_col_auc_top"] = dict(sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:15])
    # ownership of negatives / positives
    ok_ = Counter(); okp = Counter()
    for k in idx:
        (ok_ if y[k] == 0 else okp)[owner_kind(k)] += 1
    r["neg_owner"] = dict(ok_); r["pos_owner"] = dict(okp)
    neg = idx[y[idx] == 0]; pos = idx[y[idx] == 1]
    orph = np.array([own.get(cand[k]) is None for k in neg], bool) if len(neg) else np.zeros(0, bool)
    r["neg_orphan"] = dict(n=int(orph.sum()), accepted=int((p[neg][orph] >= th).sum()) if orph.any() else 0,
                           p_mean=round(float(p[neg][orph].mean()), 4) if orph.any() else None,
                           p_max=round(float(p[neg][orph].max()), 4) if orph.any() else None)
    r["neg_owned_by_other"] = dict(n=int((~orph).sum()), accepted=int((p[neg][~orph] >= th).sum()) if (~orph).any() else 0,
                                   p_mean=round(float(p[neg][~orph].mean()), 4) if (~orph).any() else None)
    for grp, g in (("neg", neg), ("pos", pos)):
        if len(g):
            r[f"{grp}_rl27_means"] = {nm: round(float(np.nanmean(RL27[g, j])), 3) for j, nm in enumerate(RLN)}
            r[f"{grp}_rv_rank1_frac"] = round(float(np.mean(RL27[g, 6] == 1)), 3)
            r[f"{grp}_nf_a0_nfn_ge1_frac"] = round(float(np.mean((RL27[g, 1] == 0) & (RL27[g, 0] >= 1))), 3)
            r[f"{grp}_p_quant"] = np.round(np.quantile(p[g], [0.1, 0.5, 0.9, 0.99]), 4).tolist()
    pick = lambda ids: [dict(c=str(ctry[i]), s1=str(s1n[i]), rec=str(rn[i]), s1_addr=str(s1a[i])[:70], rec_addr=str(ra[i])[:70],
                             x=XT[i], z=ZT[i], p=round(float(p[i]), 4), y=int(y[i]), owner=owner_kind(i), rv_rank=float(RL27[i, 6]),
                             nf_n=float(RL27[i, 0]), nf_a=float(RL27[i, 1]))
                        for i in (rng.choice(ids, min(10, len(ids)), replace=False) if len(ids) else [])]
    r["ex_neg"] = pick(neg); r["ex_pos"] = pick(pos)
    r["ex_errors"] = pick(np.flatnonzero(f & (acc != (y == 1))))
    r["top_swaps_neg"] = Counter(f"{XT[k]}->{ZT[k]}" for k in neg).most_common(15)
    r["top_swaps_pos"] = Counter(f"{XT[k]}->{ZT[k]}" for k in pos).most_common(15)
    res[name] = r
    print(name, json.dumps({k: v for k, v in r.items() if not k.startswith("ex_") and k != "by_p_bin"}, default=str)[:3000], flush=True)
    print("   by_bin", json.dumps(r["by_p_bin"]), flush=True)

# ---- the France-like analogue in V1: same-street one-word core substitution with an ORPHAN record (no owner S1 anywhere in train),
# any kinds -- how does RL-27 NEW score them, and how often are they accepted?
orph_all = np.array([own.get(c) is None for c in cand], bool)
for name, f in (("SUB1_ST_orphan", SUB1 & ST & orph_all), ("SUB1_ST_zcontent_orphan", SUB1 & ST & zc & orph_all)):
    res[name] = dict(n=int(f.sum()), accepted=int((f & acc).sum()), p_quant=np.round(np.quantile(p[f], [0.5, 0.9, 0.99]), 4).tolist() if f.any() else None,
                     by_country={C: dict(n=int((f & (ctry == C)).sum()), accepted=int((f & acc & (ctry == C)).sum())) for C in ("US", "India")})
    print(name, res[name], flush=True)
res["secs"] = round(time.time() - T0, 1)
np.savez_compressed(os.path.join(OUT, "VA_cs_1_v1_flags.npz"), SUB1=SUB1, ST=ST, zc=zc, xc=xc)
json.dump(res, open(os.path.join(OUT, "VA_cs_1_v1.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("done", f"{time.time() - T0:.0f}s")
