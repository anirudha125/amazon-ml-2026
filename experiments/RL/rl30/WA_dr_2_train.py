"""RL-30 adversarial verifier WA, claim 'S1_side_token_drop_rate', PART 2 (READ-ONLY): label-backed side (TRAIN / V1, US+India).
 (a) TRAIN GT links (random 1.6M, random_state=1; A used 0), street-matched and ALL (no street filter): per-token pure drop rates
     (TB = rl30_lib.toks set; TA = A's acronym-collapsed set) + per-occurrence position/name-length drop counts.
 (b) V1 labelled pool (RL-27 NEW OOF p): every pair with p>=0.02 or y==1: street flag, S1-only pure tokens (TB), rec-only pure tokens.
Outputs: WA_dr_2_trainGT_tok.csv, WA_dr_2_trainGT_pos.csv, WA_dr_2_V1_pairs.pkl, WA_dr_2_summary.json
"""
import os, sys, json, time, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ta_tokens(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


_sp = {}


def sp(addr):
    v = _sp.get(addr)
    if v is None:
        n, s, _ = street_parts(addr); v = (n, s); _sp[addr] = v
    return v


def same_street(a1, a2):
    if not isinstance(a1, str) or not isinstance(a2, str):
        return False
    n1, s1 = sp(a1); n2, s2 = sp(a2)
    return n1 is not None and n1 == n2 and bool(s1) and s1 == s2


SUM = {}
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
gt = DT["gt"]
SUM["train_gt_links"] = int(len(gt))
gt = gt.sample(n=min(len(gt), 1_600_000), random_state=1)
ctry = S1t.country.reindex(gt.s1.values).values
g1n = S1t.name.reindex(gt.s1.values).values; g1a = S1t.addr.reindex(gt.s1.values).values
grn = RECt.name.reindex(gt.rec.values).values; gra = RECt.addr.reindex(gt.rec.values).values
L("train loaded")
tokc = Counter(); posc = Counter(); npair = Counter()
for k in range(len(gt)):
    C = ctry[k]
    if C not in ("US", "India") or not isinstance(grn[k], str) or not isinstance(g1n[k], str):
        continue
    st = same_street(g1a[k], gra[k])
    npair[(C, st)] += 1
    A, B = ta_tokens(g1n[k]), ta_tokens(grn[k]); pa, _ = pure(A, B)
    for t in A:
        tokc[(C, st, "TA", t, "has")] += 1
    for t in pa:
        tokc[(C, st, "TA", t, "drop")] += 1
    tl = toks(g1n[k]); order = list(dict.fromkeys(tl)); A2 = frozenset(order); pa2, _ = pure(A2, frozenset(toks(grn[k])))
    nl = len(order)
    for i, t in enumerate(order):
        d = t in pa2
        tokc[(C, st, "TB", t, "has")] += 1
        if d:
            tokc[(C, st, "TB", t, "drop")] += 1
        pb_ = "first" if i == 0 else ("last" if i == nl - 1 else "mid")
        cls = "legal" if t in LEGAL else ("honor" if t in HONOR else ("stop" if t in STOP else "content"))
        posc[(C, st, pb_, min(nl, 6), cls, "has")] += 1
        if d:
            posc[(C, st, pb_, min(nl, 6), cls, "drop")] += 1
    if k % 300000 == 0:
        L("gt", k)
SUM["train_gt_pairs_by_country_street"] = {f"{a}_{b}": int(v) for (a, b), v in npair.items()}
T = pd.DataFrame([(*k[:4], k[4], v) for k, v in tokc.items()], columns=["country", "street", "scheme", "tok", "kind", "n"])
T = T.pivot_table(index=["country", "street", "scheme", "tok"], columns="kind", values="n", fill_value=0).reset_index()
T.to_csv(os.path.join(OUT, "WA_dr_2_trainGT_tok.csv"), index=False)
P = pd.DataFrame([(*k[:5], k[5], v) for k, v in posc.items()], columns=["country", "street", "pos", "nlen", "cls", "kind", "n"])
P = P.pivot_table(index=["country", "street", "pos", "nlen", "cls"], columns="kind", values="n", fill_value=0).reset_index()
P.to_csv(os.path.join(OUT, "WA_dr_2_trainGT_pos.csv"), index=False)
L("gt tables written")

# ---------------- (b) V1 pairs
m = np.load(PATHS["v1_meta"], allow_pickle=True)
p = np.load(PATHS["v1_p_new"]).astype(np.float64)
y = m["y"].astype(np.int8); s1idx = m["s1idx"]; s1ids = m["s1_ids"][s1idx]; cand = m["cand"]; vc = m["country"][s1idx]
SUM["V1_pairs"] = int(len(y)); SUM["V1_pos"] = int(y.sum()); SUM["V1_n_gt"] = int(m["n_gt"].sum())
sel = np.where((p >= 0.02) | (y == 1))[0]
SUM["V1_selected_pairs"] = int(len(sel))
v1n = S1t.name.reindex(s1ids[sel]).values; v1a = S1t.addr.reindex(s1ids[sel]).values
vrn = RECt.name.reindex(cand[sel]).values; vra = RECt.addr.reindex(cand[sel]).values
out = []
for q, k in enumerate(sel):
    if not isinstance(vrn[q], str) or not isinstance(v1n[q], str):
        continue
    A2 = frozenset(toks(v1n[q])); B2 = frozenset(toks(vrn[q])); pa2, pb2 = pure(A2, B2)
    out.append((int(k), vc[k], int(y[k]), float(p[k]), same_street(v1a[q], vra[q]), " ".join(sorted(pa2)), " ".join(sorted(pb2)), len(A2)))
    if q % 200000 == 0:
        L("v1", q, len(sel))
V = pd.DataFrame(out, columns=["k", "country", "y", "p", "street", "s1only", "reconly", "n_s1tok"])
V.to_pickle(os.path.join(OUT, "WA_dr_2_V1_pairs.pkl"))
SUM["V1_rows_written"] = int(len(V))
json.dump(SUM, open(os.path.join(OUT, "WA_dr_2_summary.json"), "w"), indent=1)
L("done", json.dumps(SUM))
