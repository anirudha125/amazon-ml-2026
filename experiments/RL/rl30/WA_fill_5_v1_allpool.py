"""RL-30 adversarial verifier WA (filler claim), PART 5 (READ-ONLY, label-backed): the WA_fill_4 key-based classes on the WHOLE V1 pool
(US/India, RL-27 NEW s42 OOF p, threshold 0.78), split by address relation (exact / street / other_addr).
Filler set per country = GT record-side adders (add_LR_gt >= 0.05, occ >= 300, from A_p2_noise_{C}_trainGT.csv) | A's France role-A set | LEGAL.
Reports n, positives, TP/FP/FN/TN, P(y=1 | rejected) -- the quantity that decides whether a 'filler = noise' feature can recover FNs.
Output: WA_fill_5_results.json
"""
import os, sys, json, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
rd = lambda n: pd.read_csv(os.path.join(OUT, n), keep_default_na=False, na_values=[""]).set_index("tok")
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
roles = rd("A_roles_France.csv")
FILLER_FR = set(roles.index[roles.role == "A"])
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
NCt = DT["s1"].country.value_counts().to_dict()
dft = {C: Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in set(toks(n))) for C in ("US", "India")}
FILL = {}
for C in ("US", "India"):
    g = rd(f"A_p2_noise_{C}_trainGT.csv"); g = g[g.occ >= 300]
    lr = (g.padd / summ2[f"{C}_trainGT"]["pairs"]) / (pd.Series({t: dft[C].get(t, 0) for t in g.index}).clip(lower=1) / NCt[C])
    FILL[C] = set(g.index[lr >= 0.05]) | FILLER_FR | LEGAL


def skey(addr):
    n, s, _ = street_parts(addr)
    return f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else ""


m = np.load(PATHS["v1_meta"], allow_pickle=True); p = np.load(PATHS["v1_p_new"])
y = m["y"].astype(int); si = m["s1idx"]; s1ids = m["s1_ids"]; cand = m["cand"]; ctry = m["country"][si]
sid = s1ids[si]
s1n = S1t.name.reindex(sid).values; s1a = S1t.addr.reindex(sid).values
rn = RECt.name.reindex(cand).values; ra = RECt.addr.reindex(cand).values
tk = {}
def T(x):
    v = tk.get(x)
    if v is None:
        v = frozenset(toks(x)) if isinstance(x, str) else frozenset(); tk[x] = v
    return v
ak = {}; sk = {}
def AK(x):
    v = ak.get(x)
    if v is None: v = akey(x) if isinstance(x, str) else ""; ak[x] = v
    return v
def SK(x):
    v = sk.get(x)
    if v is None: v = skey(x) if isinstance(x, str) else ""; sk[x] = v
    return v
cnt = Counter()
for k in range(len(y)):
    C = ctry[k]; F = FILL[C]; FNL = F - LEGAL; DROP = F | LEGAL | HONOR
    A_, B_ = T(s1n[k]), T(rn[k])
    ca, cb = A_ - DROP, B_ - DROP
    cl = None
    if ca and ca == cb and ((A_ ^ B_) & FNL):
        cl = "filler_only_any"
    elif cb and cb < ca and len(ca - cb) == 1 and ((B_ - A_) & FNL):
        cl = "filler_sub_any"
    if cl is None:
        continue
    a1, a2 = AK(s1a[k]), AK(ra[k])
    geo = "exact" if (a1 and a1 == a2) else ("street" if (SK(s1a[k]) and SK(s1a[k]) == SK(ra[k])) else "other_addr")
    acc = p[k] >= NEW_TH
    cnt[(C, cl, geo, "TP" if acc and y[k] else "FP" if acc else "FN" if y[k] else "TN")] += 1
    if k % 500000 == 0: L(k)
R = {}
for (C, cl, geo, o), v in cnt.items():
    R.setdefault(C, {}).setdefault(cl, {}).setdefault(geo, {})[o] = v
for C in R:
    for cl in R[C]:
        for geo, d in R[C][cl].items():
            rej = d.get("FN", 0) + d.get("TN", 0)
            d["P_y1_given_rejected"] = round(d.get("FN", 0) / rej, 4) if rej else None
            d["P_y1_given_accepted"] = round(d.get("TP", 0) / (d.get("TP", 0) + d.get("FP", 0)), 4) if d.get("TP", 0) + d.get("FP", 0) else None
json.dump(R, open(os.path.join(OUT, "WA_fill_5_results.json"), "w"), indent=1)
L(json.dumps(R))
