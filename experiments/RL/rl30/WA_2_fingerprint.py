"""RL-30 adversarial verifier WA, step 2 (READ-ONLY). Is the France content-for-content substitution set shaped like the V1 decoys
(the label-backed analog the claim relies on) or like true noisy records?
 (a) ACCEPTANCE: among SAME-STREET candidates in the pool with a 1-for-1 core substitution whose two words are both 'content',
     what share does the model accept (p>=0.78)?  V1 (US/India, labels) vs France test pool (sample of P3 chunks).
 (b) FINGERPRINTS (label-free in France, calibrated on V1 labels): legal-form identical, address string identical, record
     all-uppercase, record source S2, extra non-core differences, typo-paired tokens, replication of the record's
     (house number, street, core name) key on another record of the same country.
     France groups: FLAG (content-for-content, accepted, HI/LO), controls (accepted same-street core-identical HI; accepted
     same-street filler substitution HI), and REJ (same shape in the pool but not accepted).
Output: WA_2_results.json
"""
import os, sys, json, time, glob, math, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
R = {}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


_cc = {}


def ctok(s):
    v = _cc.get(s)
    if v is None:
        v = frozenset(t for t in toks(s) if t not in LEGAL and t not in HONOR); _cc[s] = v
    return v


def istypo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pdiff(A, B):
    pa, pb = set(A - B), set(B - A); ntypo = 0
    if pa and pb:
        for _, x, y in sorted(((LV.normalized_similarity(x, y), x, y) for x in pa for y in pb if istypo(x, y)), reverse=True):
            if x in pa and y in pb:
                pa.discard(x); pb.discard(y); ntypo += 1
    return pa, pb, ntypo


_sk = {}


def skey(addr):
    v = _sk.get(addr, 0)
    if v == 0:
        n, st, _ = street_parts(addr); v = (n, st) if (n is not None and st) else None; _sk[addr] = v
    return v


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(ph, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def legal(s):
    return frozenset(t for t in toks(s) if t in LEGAL)


def fp(s1name, s1addr, rid, rname, raddr, ntypo, repidx):
    letters = re.sub(r"[^A-Za-z]", "", rname)
    extra = len(frozenset(toks(s1name)) ^ frozenset(toks(rname))) - 2
    kk = skey(raddr)
    rep = (repidx.get((kk, ctok(rname)), 1) - 1) if kk is not None else 0
    return dict(legal_eq=legal(s1name) == legal(rname), addr_exact=fold(s1addr).strip() == fold(raddr).strip(),
                upper=bool(letters) and letters.isupper(), s2=rid.startswith("S2"), extra_diff=extra, typo=ntypo, rep=rep >= 1)


def summarize(rows):
    if not rows:
        return dict(n=0)
    df = pd.DataFrame(rows)
    out = dict(n=int(len(df)))
    for c in ("legal_eq", "addr_exact", "upper", "s2", "rep"):
        out[c] = round(float(df[c].mean()), 4)
    out["extra_diff_mean"] = round(float(df.extra_diff.mean()), 3); out["typo_mean"] = round(float(df.typo.mean()), 3)
    out["extra_diff_gt0"] = round(float((df.extra_diff > 0).mean()), 4)
    return out


def build_rep(recs):
    idx = Counter()
    for n, a in zip(recs.name.values, recs.addr.values):
        k = skey(a)
        if k is not None:
            idx[(k, ctok(n))] += 1
    return idx


roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
contFR = frozenset(roles.index[(roles.add_LR < 0.05) & (roles.occ >= 300) & (~roles.legal.astype(bool))])
fillFR = frozenset(roles.index[(roles.add_LR >= 0.05) & (roles.occ >= 300)])
W1 = json.load(open(os.path.join(OUT, "WA_1_results.json")))

# ============================ V1 calibration
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
summ2 = json.load(open(os.path.join(OUT, "A_p2_summary.json")))
NCt = DT["s1"].country.value_counts().to_dict()
V = pd.read_pickle(os.path.join(OUT, "WA_1_V1_subs.pkl"))
mv = np.load(PATHS["v1_meta"], allow_pickle=True)
cand = mv["cand"]
for C in ("US", "India"):
    dft = Counter(t for s in DT["s1"].name.values[DT["s1"].country.values == C] for t in ctok(s))
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
    npp = summ2[f"{C}_testPP"]["pairs"]
    lr = {tk: (r.padd / npp) / (max(dft.get(tk, 0), 1) / NCt[C]) for tk, r in zip(t.index, t.itertuples()) if r.occ >= 300}
    cont = frozenset(tk for tk, v in lr.items() if v < 0.05 and tk not in LEGAL and tk not in HONOR)
    fill = frozenset(tk for tk, v in lr.items() if v >= 0.05)
    recs = pd.concat([DT["s2"], DT["s3"]]); recs = recs[recs.country == C]
    rep = build_rep(recs)
    v = V[(V.country == C) & V.street].copy()
    v["cc"] = v.x.isin(cont) & v.y_tok.isin(cont)
    v["fill"] = v.y_tok.isin(fill)
    groups = defaultdict(list)
    for r in v.itertuples():
        rid = cand[r.k]; s1a = S1t.addr[r.s1]; ra = RECt.addr[rid]
        _, _, nt = pdiff(ctok(r.s1name), ctok(r.recname))
        f = fp(r.s1name, s1a, rid, r.recname, ra, nt, rep)
        kind = "cc" if r.cc else ("fill" if r.fill else "other")
        groups[f"y{r.y}_{kind}"].append(f); groups[f"y{r.y}_all"].append(f)
        groups[f"y{r.y}_all_{'acc' if r.p >= NEW_TH else 'rej'}"].append(f)
        if r.cc:
            groups[f"cc_{'acc' if r.p >= NEW_TH else 'rej'}"].append(f)
            groups[f"cc_{'acc' if r.p >= NEW_TH else 'rej'}_y{r.y}"].append(f)
    R[f"V1_{C}_fingerprints_samestreet_1for1"] = {k: summarize(g) for k, g in sorted(groups.items())}
    cc = v[v.cc]
    R[f"V1_{C}_acceptance_cc_samestreet"] = dict(n=int(len(cc)), accepted=int((cc.p >= NEW_TH).sum()),
                                                 acc_rate=wilson(int((cc.p >= NEW_TH).sum()), len(cc)),
                                                 neg_rate_all=wilson(int((cc.y == 0).sum()), len(cc)),
                                                 neg_rate_accepted=wilson(int(((cc.y == 0) & (cc.p >= NEW_TH)).sum()), int((cc.p >= NEW_TH).sum())))
    log(C, json.dumps(R[f"V1_{C}_acceptance_cc_samestreet"]))
    for k, s in R[f"V1_{C}_fingerprints_samestreet_1for1"].items():
        log("  ", k, s)
del DT, S1t, RECt, V

# ============================ France
D = load("test", verbose=False)
fr = D["s1"][D["s1"].country == "France"].set_index("id")
recs = pd.concat([D["s2"], D["s3"]]); recs = recs[recs.country == "France"]
rep = build_rep(recs)
RI = recs.set_index("id")
del D
a = accepted("S005_France")
acc_key = set(zip(a.s1.values, a.rec.values))
kept = a[a.kept_final]
s1n = fr.name.reindex(kept.s1.values).values; s1a = fr.addr.reindex(kept.s1.values).values
rn = RI.name.reindex(kept.rec.values).values; ra = RI.addr.reindex(kept.rec.values).values
groups = defaultdict(list)
rng = np.random.default_rng(0)
ctrl_take = rng.random(len(kept)) < 0.08
for k in range(len(kept)):
    if not isinstance(rn[k], str):
        continue
    sk = skey(s1a[k])
    if sk is None or sk != skey(ra[k]):
        continue
    A, B = ctok(s1n[k]), ctok(rn[k])
    pa, pb, nt = pdiff(A, B)
    band = "HI" if kept.p.values[k] >= 0.99 else "LO"
    if len(pa) == 1 and len(pb) == 1:
        x, y = next(iter(pa)), next(iter(pb))
        if x in contFR and y in contFR and x != y:
            groups[f"FLAG_{band}"].append(fp(s1n[k], s1a[k], kept.rec.values[k], rn[k], ra[k], nt, rep)); continue
        if y in fillFR and ctrl_take[k]:
            groups[f"CTRL_fillsub_{band}"].append(fp(s1n[k], s1a[k], kept.rec.values[k], rn[k], ra[k], nt, rep)); continue
    if not pa and not pb and ctrl_take[k]:
        groups[f"CTRL_coreident_{band}"].append(fp(s1n[k], s1a[k], kept.rec.values[k], rn[k], ra[k], nt, rep))
log("France accepted groups", {k: len(v) for k, v in groups.items()})

# pool: same shape, not accepted (sample of chunks)
RN = dict(zip(recs.id.values, recs.name.values)); RA = dict(zip(recs.id.values, recs.addr.values))
FN = fr.name.to_dict(); FA = fr.addr.to_dict()
chunks = sorted(glob.glob(PATHS["test_chunks"].format(country="France")))
sel = chunks[::4]
n_pool = n_street_cc = n_street_cc_acc = 0
n_s1_seen = set()
for fch in sel:
    z = np.load(fch, allow_pickle=True)
    s1c, cc_ = z["s1"], z["cand"]
    n_pool += len(s1c)
    for s, c in zip(s1c, cc_):
        n_s1_seen.add(s)
        rname = RN.get(c)
        if rname is None:
            continue
        sname = FN[s]
        A, B = ctok(sname), ctok(rname)
        if len(A ^ B) > 6 or len(A ^ B) < 2:
            continue
        pa, pb, nt = pdiff(A, B)
        if not (len(pa) == 1 and len(pb) == 1):
            continue
        x, y = next(iter(pa)), next(iter(pb))
        if not (x in contFR and y in contFR and x != y):
            continue
        s1ad = FA[s]; rad = RA[c]
        sk = skey(s1ad)
        if sk is None or sk != skey(rad):
            continue
        n_street_cc += 1
        isacc = (s, c) in acc_key
        n_street_cc_acc += isacc
        if not isacc:
            groups["REJ_pool"].append(fp(sname, s1ad, c, rname, rad, nt, rep))
    log("chunk", os.path.basename(fch), n_pool, n_street_cc, n_street_cc_acc)
    if time.time() - T0 > 700:
        break
R["France_pool_sample"] = dict(n_chunks=len(sel), n_S1=len(n_s1_seen), n_pool_rows=int(n_pool), n_samestreet_cc=int(n_street_cc),
                               n_accepted=int(n_street_cc_acc), acc_rate=wilson(int(n_street_cc_acc), int(n_street_cc)))
R["France_fingerprints"] = {k: summarize(g) for k, g in sorted(groups.items())}
for k, s in R["France_fingerprints"].items():
    log("  ", k, s)
log(json.dumps(R["France_pool_sample"]))
json.dump(R, open(os.path.join(OUT, "WA_2_results.json"), "w"), indent=1, default=str)
log("done")
