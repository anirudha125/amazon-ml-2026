"""RL-30 adversarial verifier VA (lens: ALREADY CAPTURED), step 1: labelled V1 (US/India) check of Investigator A's
'generator filler tokens' signal against RL-27 NEW (seed 42, OOF-protocol threshold 0.78). READ-ONLY; writes VA_1_v1.json only.
Filler set per country = A's definition (A_part2j): add_LR >= 0.05 with occ >= 300 in A_p2_noise_<C>_testPP.csv
(add_LR = pure record-only adds per pseudo-positive pair / share of train S1 of that country containing the token);
droppable = drop_rate >= 0.20 with occ >= 300.
Per V1 pair (all 2.55M pool rows): S1-only / record-only name tokens after A's typo pairing.
  F_add   record-only tokens non-empty and all fillers                                    (A's 'record-only filler = noise')
  F_only  F_add and every S1-only token is filler / droppable / legal / honorific         (the whole name diff is filler)
  F_sub1  exactly one S1-only content token dropped and record-only tokens all fillers     (A's 'filler substitution')
  *_st    same variants restricted to same house number + same street-name tokens
For each: y counts, RL-27 NEW confusion at 0.78, y-rate by p bin vs unflagged (does the flag carry information beyond p?),
and macro-F0.5 deltas of (i) an oracle that labels every flagged pair correctly, (ii) rules 'flag & p >= t -> accept'."""
import os, sys, json, time, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
NCt = DT["s1"].country.value_counts().to_dict()
FILL, DROP = {}, {}
for C in ("US", "India"):
    dfc = Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in ntoks(n))
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    t = t[t.occ >= 300]
    lr = (t.padd / summ2[f"{C}_testPP"]["pairs"]) / (pd.Series({k: dfc.get(k, 0) for k in t.index}).clip(lower=1) / NCt[C])
    FILL[C] = set(lr[lr >= 0.05].index); DROP[C] = set(t.index[t.drop_rate.fillna(0) >= 0.20])
    print(C, "fillers", len(FILL[C]), sorted(FILL[C])[:80], "droppable", len(DROP[C]), sorted(DROP[C])[:60], flush=True)

V = S2.load_set("V1", "a50n10d10a"); V["country_s1"] = V["country"]
p = np.load(PATHS["v1_p_new"]); y = V["y"].astype(int); th = NEW_TH
base = S2.summarize(V, p, th)
print("baseline V1 macro", round(base["macro"] * 100, 3), "TP", base["tp"], "FP", base["fp"], flush=True)
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]
n = len(y)
s1n = S1t.name.reindex(s1ids).values; s1a = S1t.addr.reindex(s1ids).values
rn = RECt.name.reindex(cand).values; ra = RECt.addr.reindex(cand).values
cache = {}


def nt(s):
    if not isinstance(s, str): s = ""
    v = cache.get(s)
    if v is None:
        v = cache[s] = ntoks(s)
    return v


scache = {}


def sp(a):
    if not isinstance(a, str): a = ""
    v = scache.get(a)
    if v is None:
        v = scache[a] = street_parts(a)[:2]
    return v


F_add = np.zeros(n, bool); F_only = np.zeros(n, bool); F_sub1 = np.zeros(n, bool); ST = np.zeros(n, bool)
n_rec_only = np.zeros(n, np.int16); n_s1_only = np.zeros(n, np.int16)
for k in range(n):
    C = ctry[k]
    A, B = nt(s1n[k]), nt(rn[k])
    oa, ob = A - B, B - A
    if not ob:
        continue
    pa, pb = set(oa), set(ob)
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    n_rec_only[k] = len(pb); n_s1_only[k] = len(pa)
    if not pb:
        continue
    fl, dr = FILL[C], DROP[C]
    if all(t in fl for t in pb):
        F_add[k] = True
        rest = [t for t in pa if t not in fl and t not in dr and t not in LEGAL and t not in HONOR]
        if not rest:
            F_only[k] = True
        elif len(rest) == 1:
            F_sub1[k] = True
        n1, st1 = sp(s1a[k]); n2, st2 = sp(ra[k])
        ST[k] = n1 is not None and n1 == n2 and bool(st1) and st1 == st2
    if k % 500000 == 0:
        print(k, f"{time.time() - T0:.0f}s", flush=True)

acc = p >= th
bins = [0, 0.01, 0.05, 0.2, 0.5, th, 0.9, 0.99, 1.0001]
res = dict(baseline=dict(macro=round(base["macro"] * 100, 3), us=round(base["us"] * 100, 3), india=round(base["india"] * 100, 3),
                         TP=base["tp"], FP=base["fp"], FN_in_pool=int(((~acc) & (y == 1)).sum()), FN_total=int(V["n_gt"].sum() - base["tp"])),
           fillers={C: sorted(FILL[C]) for C in FILL}, droppable={C: sorted(DROP[C]) for C in DROP})
FN_all = int(((~acc) & (y == 1)).sum()); FP_all = int((acc & (y == 0)).sum())


def conf(mask):
    return dict(n=int(mask.sum()), pos=int((y[mask] == 1).sum()), neg=int((y[mask] == 0).sum()),
                TP=int((mask & acc & (y == 1)).sum()), FN=int((mask & ~acc & (y == 1)).sum()),
                FP=int((mask & acc & (y == 0)).sum()), TN=int((mask & ~acc & (y == 0)).sum()))


def by_bin(mask):
    out = {}
    b = np.digitize(p, bins) - 1
    for i in range(len(bins) - 1):
        f = mask & (b == i); u = (~mask) & (b == i)
        out[f"[{bins[i]:.2f},{bins[i + 1]:.2f})"] = dict(flag_n=int(f.sum()), flag_yrate=round(float(y[f].mean()), 4) if f.any() else None,
                                                         unflag_n=int(u.sum()), unflag_yrate=round(float(y[u].mean()), 4) if u.any() else None,
                                                         flag_mean_p=round(float(p[f].mean()), 4) if f.any() else None)
    return out


def macro_with(new_acc):
    pp = np.where(new_acc, 1.0, 0.0)
    s = S2.summarize(V, pp, 0.5)
    return round(s["macro"] * 100, 4), s["tp"], s["fp"]


flags = dict(F_add=F_add, F_only=F_only, F_sub1=F_sub1, F_add_st=F_add & ST, F_only_st=F_only & ST, F_sub1_st=F_sub1 & ST)
ex = {}
rng = np.random.default_rng(0)
for name, f in flags.items():
    r = {}
    for C in ("US", "India"):
        m = f & (ctry == C)
        r[C] = conf(m)
    r["all"] = conf(f)
    r["share_of_all_FN_in_pool"] = round(r["all"]["FN"] / FN_all, 4)
    r["share_of_all_FP"] = round(r["all"]["FP"] / FP_all, 4)
    r["by_p_bin"] = by_bin(f)
    orc = acc.copy(); orc[f] = y[f] == 1
    r["oracle_macro"], r["oracle_TP"], r["oracle_FP"] = macro_with(orc)
    r["oracle_delta_pp"] = round(r["oracle_macro"] - base["macro"] * 100, 4)
    rules = {}
    for t in (0.0, 0.05, 0.2, 0.5):
        na = acc | (f & (p >= t))
        mm, tp, fp = macro_with(na)
        rules[f"accept_flag_p>={t}"] = dict(macro=mm, delta_pp=round(mm - base["macro"] * 100, 4), dTP=tp - base["tp"], dFP=fp - base["fp"])
    r["rules"] = rules
    fn_idx = np.flatnonzero(f & ~acc & (y == 1)); fp_idx = np.flatnonzero(f & acc & (y == 0)); tn_idx = np.flatnonzero(f & ~acc & (y == 0))
    pick = lambda idx: [dict(country=str(ctry[i]), s1=str(s1n[i]), rec=str(rn[i]), s1_addr=str(s1a[i]), rec_addr=str(ra[i]), p=round(float(p[i]), 4), y=int(y[i]))
                        for i in (rng.choice(idx, min(8, len(idx)), replace=False) if len(idx) else [])]
    ex[name] = dict(FN=pick(fn_idx), FP=pick(fp_idx), TN=pick(tn_idx))
    res[name] = r
    print(name, json.dumps({k: v for k, v in r.items() if k != "by_p_bin"}), flush=True)
    print("   by_bin", json.dumps(r["by_p_bin"]), flush=True)

# how much of the whole V1 FN pool has a record-only name token at all, vs record-only all-filler
res["FN_in_pool_breakdown"] = dict(
    FN=FN_all,
    rec_only_nonempty=int(((~acc) & (y == 1) & (n_rec_only > 0)).sum()),
    rec_only_all_filler=int(((~acc) & (y == 1) & F_add).sum()),
    no_name_diff=int(((~acc) & (y == 1) & (n_rec_only == 0) & (n_s1_only == 0)).sum()),
)
res["examples"] = ex
res["secs"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(OUT, "VA_1_v1.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(res["FN_in_pool_breakdown"]), "done", f"{time.time() - T0:.0f}s")
