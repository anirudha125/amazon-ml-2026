"""RL-30 investigator A, PART 2j (READ-ONLY). What do label-known same-street one-word-substitution NEGATIVES look like in V1 (US/India)?
Kind of substituted-in token (generic adder vs content word) for y=0 vs y=1, with examples. Output: A_p2j_summary.json"""
import os, sys, json, re, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")
def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s); return frozenset(re.findall(r"[a-z0-9]+", s))
def core(A): return frozenset(t for t in A if t not in LEGAL and t not in HONOR)
def typo(a, b): return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
dft = {C: Counter(t for n in DT["s1"].name.values[DT["s1"].country.values == C] for t in ntoks(n)) for C in ("US", "India")}
NCt = DT["s1"].country.value_counts().to_dict()
m = np.load(PATHS["v1_meta"], allow_pickle=True); pnew = np.load(PATHS["v1_p_new"])
y = m["y"]; s1ids = m["s1_ids"]; si = m["s1idx"]; cand = m["cand"]; vc = m["country"]
S = {}; rng = random.Random(0)
for C in ("US", "India"):
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    lr = (t.padd / summ2[f"{C}_testPP"]["pairs"]) / (pd.Series({k: dft[C].get(k, 0) for k in t.index}).clip(lower=1) / NCt[C])
    lr = lr[t.occ >= 300].to_dict()
    kind = lambda w: "rare" if w not in lr else ("generic" if lr[w] >= 0.05 else "content")
    sel = np.where(vc[si] == C)[0]
    s1n = S1t.name.reindex(s1ids[si[sel]]).values; s1a = S1t.addr.reindex(s1ids[si[sel]]).values
    rn = RECt.name.reindex(cand[sel]).values; ra = RECt.addr.reindex(cand[sel]).values
    cnt = Counter(); ex = defaultdict(list)
    for q, k in enumerate(sel):
        if not isinstance(rn[q], str): continue
        A, B = ntoks(s1n[q]), ntoks(rn[q])
        if len(A ^ B) > 4: continue
        pa, pb = set(core(A - B)), set(core(B - A))
        if pa and pb:
            for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
                if x in pa and z in pb: pa.discard(x); pb.discard(z)
        if not (len(pa) == 1 and len(pb) == 1): continue
        n1, st1, _ = street_parts(s1a[q]); n2, st2, _ = street_parts(ra[q])
        if not (n1 is not None and n1 == n2 and st1 and st1 == st2): continue
        x, z = next(iter(pa)), next(iter(pb))
        key = f"y{int(y[k])}_in_{kind(z)}_out_{kind(x)}"
        cnt[key] += 1
        if len(ex[key]) < 6 and rng.random() < 0.3:
            ex[key].append((s1n[q], rn[q], round(float(pnew[k]), 3)))
    S[C] = dict(counts=dict(cnt), examples=dict(ex))
    print(C, dict(cnt), flush=True)
    for kk, v in ex.items(): print("  ", kk, v[:4])
json.dump(S, open(os.path.join(OUT, "A_p2j_summary.json"), "w"), indent=1, ensure_ascii=False)
print("done")
