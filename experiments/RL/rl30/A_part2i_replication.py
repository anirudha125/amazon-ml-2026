"""RL-30 investigator A, PART 2i (READ-ONLY). Replication test for France content-for-content substitutions.
Per-record name noise draws independently per record, so a substituted name variant should rarely re-occur on ANOTHER record at the
same street address.  A distinct (unlinked / phantom) business at the same address would carry its name on several records.
Key = (house number, street_name tokens, core name tokens).  For each accepted pair type, count OTHER records (S2+S3, same country)
with the record's key (rep_rec) and with the S1's key (rep_s1).
Calibration (labels, V1 US/India): same-street candidates with a 1-token core difference, y=1 (noise) vs y=0 (decoy / other business).
Output: A_p2i_summary.json
"""
import os, sys, json, time, re
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


def pdiff(A, B):
    pa, pb = set(core(A - B)), set(core(B - A))
    if pa and pb:
        for _, x, y_ in sorted(((LV.normalized_similarity(x, y_), x, y_) for x in pa for y_ in pb if typo(x, y_)), reverse=True):
            if x in pa and y_ in pb: pa.discard(x); pb.discard(y_)
    return pa, pb


def skey(addr):
    n, st, _ = street_parts(addr)
    return (n, st) if n is not None and st else None


def build_index(recs):
    idx = Counter()
    keys = {}
    for i, n, a in zip(recs.id.values, recs.name.values, recs.addr.values):
        k = skey(a)
        if k is None: continue
        kk = (k, core(ntoks(n)))
        idx[kk] += 1; keys[i] = kk
    return idx, keys


roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
content_fr = set(roles.index[(roles.add_LR < 0.05) & (roles.occ >= 300) & (~roles.legal.astype(bool))])
S = {}

# ---------- calibration on train (V1 labels)
DT = load("train", verbose=False)
m = np.load(PATHS["v1_meta"], allow_pickle=True)
y = m["y"]; s1ids = m["s1_ids"]; si = m["s1idx"]; cand = m["cand"]; vc = m["country"]
S1t = DT["s1"].set_index("id")
for C in ("US", "India"):
    recs = pd.concat([DT["s2"], DT["s3"]]); recs = recs[recs.country == C]
    idx, keys = build_index(recs)
    RI = recs.set_index("id")
    sel = np.where(vc[si] == C)[0]
    s1n = S1t.name.reindex(s1ids[si[sel]]).values; s1a = S1t.addr.reindex(s1ids[si[sel]]).values
    rn = RI.name.reindex(cand[sel]).values
    st = defaultdict(list)
    for q, k in enumerate(sel):
        if not isinstance(rn[q], str): continue
        A, B = ntoks(s1n[q]), ntoks(rn[q])
        if len(A ^ B) > 4: continue
        kk = keys.get(cand[k])
        if kk is None or kk[0] != skey(s1a[q]): continue
        pa, pb = pdiff(A, B)
        if len(pa) == 1 and len(pb) == 1:
            t = "sub"
        elif len(pb) == 1 and not pa:
            t = "add"
        elif len(pa) == 1 and not pb:
            t = "drop"
        else:
            continue
        st[("y1" if y[k] else "y0") + "_" + t].append(idx[kk] - 1)
    S[f"V1_{C}"] = {k: dict(n=len(v), frac_rep_ge1=round(float(np.mean(np.array(v) >= 1)), 4), mean_rep=round(float(np.mean(v)), 3)) for k, v in st.items()}
    print(C, S[f"V1_{C}"], f"{time.time() - T0:.0f}s", flush=True)
del DT, S1t

# ---------- France test accepted
D = load("test", verbose=False)
recs = pd.concat([D["s2"], D["s3"]]); recs = recs[recs.country == "France"]
idx, keys = build_index(recs)
RI = recs.set_index("id")
fr = D["s1"][D["s1"].country == "France"].set_index("id")
a = accepted("S005_France"); a = a[a.kept_final]
s1n = fr.name.reindex(a.s1.values).values; s1a = fr.addr.reindex(a.s1.values).values
rn = RI.name.reindex(a.rec.values).values
st = defaultdict(list); ex = defaultdict(list)
for k in range(len(a)):
    if not isinstance(rn[k], str): continue
    kk = keys.get(a.rec.values[k])
    sk = skey(s1a[k])
    if kk is None or sk is None or kk[0] != sk: continue
    A, B = ntoks(s1n[k]), ntoks(rn[k])
    pa, pb = pdiff(A, B)
    band = "HI" if a.p.values[k] >= 0.99 else "LO"
    if not pa and not pb:
        t = "nodiff"
    elif len(pa) == 1 and len(pb) == 1:
        x, y_ = next(iter(pa)), next(iter(pb))
        t = "sub_content" if (x in content_fr and y_ in content_fr) else ("sub_generic" if roles.add_LR.get(y_, 0) >= 0.05 else "sub_other")
    elif len(pa) == 1 and not pb:
        t = "drop"
    elif len(pb) == 1 and not pa:
        t = "add"
    else:
        t = "multi"
    rep = idx[kk] - 1
    st[f"{band}_{t}"].append(rep)
    if t == "sub_content" and rep >= 1 and len(ex[band]) < 12 and k % 5 == 0:
        ex[band].append(dict(s1=s1n[k], rec=rn[k], p=round(float(a.p.values[k]), 4), n_other_records_same_name_same_street=int(rep)))
S["France_TEST"] = {k: dict(n=len(v), frac_rep_ge1=round(float(np.mean(np.array(v) >= 1)), 4), mean_rep=round(float(np.mean(v)), 3)) for k, v in sorted(st.items())}
S["France_examples_replicated_content_sub"] = ex
print(json.dumps(S["France_TEST"]), flush=True)
json.dump(S, open(os.path.join(OUT, "A_p2i_summary.json"), "w"), indent=1, ensure_ascii=False)
print("done", f"{time.time() - T0:.0f}s")
