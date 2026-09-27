"""RL-30 investigator A, PART 2e (READ-ONLY). Label-free 'extra record' test: are France content-word substitutions NOISE on
true records or EXTRA (decoy) records?
Logic: per-S1 record counts are under-dispersed (train: mean 3.46, var 2.91). If a record type X is noise applied to true
records with small per-record probability, S1s carrying >=1 X-record are size-biased: E[#other records | X] = E[n^2]/E[n] - 1.
If X-records are extra records attached independently of the true count, E[#other records | X] = E[n] (not size-biased).
Calibration on V1 with labels (US/India): X = true link with a generic substitution (noise) vs X = same-street y=0 candidate
with a one-token core difference (decoy). Then apply to TEST accepted S005 pairs (kept_final) per country and band.
Outputs: A_p2e_summary.json
"""
import os, sys, json, time, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def core(A):
    return frozenset(t for t in A if t not in LEGAL and t not in HONOR)


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure_core_diff(A, B):
    pa, pb = set(core(A - B)), set(core(B - A))
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb: pa.discard(a); pb.discard(b)
    return pa, pb


def rd(name):
    return pd.read_csv(os.path.join(OUT, name), keep_default_na=False, na_values=[""])


summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
D = load("test", verbose=False)
s1 = D["s1"]; NC = s1.country.value_counts().to_dict()
dfc = {C: Counter(t for n in s1.name.values[s1.country.values == C] for t in ntoks(n)) for C in NC}
addLR = {}
for C in ("France", "US", "India"):
    t = rd(f"A_p2_noise_{C}_testPP.csv").set_index("tok"); n = summ2[f"{C}_testPP"]["pairs"]
    lr = (t.padd / n) / (pd.Series({k: dfc[C].get(k, 0) for k in t.index}).clip(lower=1) / NC[C])
    addLR[C] = lr[t.occ >= 300].to_dict()


def kind(C, tok):
    v = addLR[C].get(tok)
    return "rare" if v is None else ("generic" if v >= 0.05 else "content")


def rtype(C, A, B):
    pa, pb = pure_core_diff(A, B)
    if not pa and not pb:
        return "nodiff"
    if len(pa) == 1 and len(pb) == 1:
        x, y_ = next(iter(pa)), next(iter(pb))
        kx, ky = kind(C, x), kind(C, y_)
        if kx == "content" and ky == "content": return "sub_content"
        if ky == "generic": return "sub_generic"
        return "sub_other"
    if len(pa) == 1 and not pb:
        return "drop_" + kind(C, next(iter(pa)))
    if len(pb) == 1 and not pa:
        return "add_" + kind(C, next(iter(pb)))
    return "multi"


def count_test(n_by_s1, flags_by_s1):
    """n_by_s1: Series of per-S1 total counts; flags_by_s1: dict type -> Counter(s1 -> #records of that type)"""
    n = n_by_s1.values.astype(float)
    pred_noise = float((n ** 2).mean() / n.mean() - 1); pred_extra = float(n.mean())
    out = dict(n_s1=int(len(n)), mean_n=pred_extra, var_n=float(n.var()), pred_noise_sizebiased_minus1=pred_noise, pred_extra=pred_extra)
    for X, cnt in flags_by_s1.items():
        ids = list(cnt.keys())
        if len(ids) < 30:
            continue
        other = n_by_s1.reindex(ids).values - np.array([cnt[i] for i in ids])
        m, se = float(np.mean(other)), float(np.std(other) / np.sqrt(len(other)))
        pos = (m - pred_noise) / (pred_extra - pred_noise) if pred_extra != pred_noise else np.nan
        out[X] = dict(n_s1=len(ids), mean_other=round(m, 4), se=round(se, 4), position_0noise_1extra=round(float(pos), 3))
    return out


S = {}
# ---------------- calibration on V1 (labels)
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
m = np.load(PATHS["v1_meta"], allow_pickle=True)
y = m["y"]; s1ids = m["s1_ids"]; si = m["s1idx"]; cand = m["cand"]; vc = m["country"]
ngt = pd.Series(m["n_gt"], index=s1ids)
for C in ("US", "India"):
    ids_c = s1ids[vc == C]
    sel = np.where((vc[si] == C))[0]
    # restrict to candidates sharing house number + street with the S1 (cheap pre-filter on name similarity: at most 1 core diff each side)
    s1n = S1t.name.reindex(s1ids[si[sel]]).values; s1a = S1t.addr.reindex(s1ids[si[sel]]).values
    rn = RECt.name.reindex(cand[sel]).values; ra = RECt.addr.reindex(cand[sel]).values
    flags = defaultdict(Counter)
    for q, k in enumerate(sel):
        if not isinstance(rn[q], str):
            continue
        A, B = ntoks(s1n[q]), ntoks(rn[q])
        if len(A ^ B) > 4:
            continue
        pa, pb = pure_core_diff(A, B)
        if len(pa) > 1 or len(pb) > 1 or not (pa or pb):
            continue
        n1, st1, _ = street_parts(s1a[q]); n2, st2, _ = street_parts(ra[q])
        if not (n1 is not None and n1 == n2 and st1 and st1 == st2):
            continue
        t = rtype(C, A, B)
        sid = s1ids[si[k]]
        if y[k] == 1:
            flags["TRUE_" + t][sid] += 1
        else:
            flags["DECOY_y0_" + t][sid] += 1
    # true-link noise: other = n_gt - #flagged true links; decoys: other = n_gt (flag count not subtracted)
    res = count_test(ngt.reindex(ids_c), {k: v for k, v in flags.items() if k.startswith("TRUE_")})
    n = ngt.reindex(ids_c).values.astype(float)
    for k, v in flags.items():
        if k.startswith("DECOY") and len(v) >= 30:
            o = ngt.reindex(list(v.keys())).values
            res[k] = dict(n_s1=len(v), mean_other=round(float(o.mean()), 4), se=round(float(o.std() / np.sqrt(len(o))), 4),
                          position_0noise_1extra=round(float((o.mean() - res["pred_noise_sizebiased_minus1"]) / (res["pred_extra"] - res["pred_noise_sizebiased_minus1"])), 3))
    S[f"V1_{C}_calibration"] = res
    print(C, "V1", json.dumps(res), f"{time.time() - T0:.0f}s", flush=True)
del DT, S1t, RECt

# ---------------- TEST accepted (kept_final)
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
for C in ("France", "US", "India"):
    a = accepted(f"S005_{C}"); a = a[a.kept_final]
    sc = s1[s1.country == C].set_index("id")
    s1n = sc.name.reindex(a.s1.values).values; rn = REC.name.reindex(a.rec.values).values
    nby = a.groupby("s1").size()
    flags = defaultdict(Counter)
    pv = a.p.values; sv = a.s1.values
    for k in range(len(a)):
        if not isinstance(rn[k], str):
            continue
        t = rtype(C, ntoks(s1n[k]), ntoks(rn[k]))
        band = "HI" if pv[k] >= 0.99 else "LO"
        flags[f"{band}_{t}"][sv[k]] += 1
    res = count_test(nby, flags)
    S[f"TEST_{C}"] = res
    print(C, "TEST", json.dumps(res), f"{time.time() - T0:.0f}s", flush=True)
json.dump(S, open(os.path.join(OUT, "A_p2e_summary.json"), "w"), indent=1)
print("done", f"{time.time() - T0:.0f}s")
