"""RL-30 adversarial verification (investigator WD) of investigator D's signal
'same core name, same street, different house number (delta class)'.
Independent re-implementation: own name core (dotted acronyms collapsed, legal/honorific/stopwords removed),
own street/house-number parser (first comma component whose first real token is a number), own delta taxonomy.
READ-ONLY. Writes rl30/WD_verify.json (NEW file).
Usage: OMP_NUM_THREADS=4 nice -n 10 python WD_verify.py
"""
import sys, os, re, json, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, PATHS, NEW_TH, fold, LEGAL, HONOR, STOP, STREET_TYPE, UNIT
from rapidfuzz import fuzz

t0 = time.time()
L = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
R = {}

_DOT = re.compile(r"\b([a-z])\.\s?(?=[a-z]\b)")
_TOK = re.compile(r"[a-z0-9]+")


def wcore(name):
    f = fold(name)
    for _ in range(3):
        f = _DOT.sub(r"\1", f)
    return frozenset(t for t in _TOK.findall(f) if t not in LEGAL and t not in HONOR and t not in STOP)


SKIP = {"n", "no", "num", "numero"}


def wstreet(addr):
    for c in fold(addr).split(","):
        tk = _TOK.findall(c)
        i = 0
        while i < len(tk) and tk[i] in SKIP:
            i += 1
        if i < len(tk) and tk[i].isdigit():
            words = frozenset(t for t in tk[i + 1:] if not t.isdigit() and t not in STREET_TYPE and t not in UNIT and t not in STOP and len(t) > 1)
            if words:
                return (tk[i].lstrip("0") or "0"), words
    return None, frozenset()


def same_street(a, b):
    if not a or not b:
        return False
    return a == b or fuzz.ratio(" ".join(sorted(a)), " ".join(sorted(b))) >= 80


def _del1(long, short):
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def dclass(a, b):
    if a == b:
        return "SAME"
    d = int(b[:9]) - int(a[:9])
    if abs(d) == 1:
        return "pm1"
    if abs(d) == 2:
        return "pm2"
    if len(a) >= 2 and sorted(a) == sorted(b):
        return "perm"
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return "digsub"
    if abs(len(a) - len(b)) == 1 and (_del1(a, b) if len(a) > len(b) else _del1(b, a)):
        return "digindel"
    if abs(d) <= 10:
        return "even3_10" if d % 2 == 0 else "odd3_10"
    return "even_gt10" if d % 2 == 0 else "odd_gt10"


CLASSES = ["pm1", "pm2", "odd3_10", "even3_10", "digsub", "odd_gt10", "even_gt10", "digindel", "perm"]


class Parser:
    def __init__(self, name_of, addr_of):
        self.name_of, self.addr_of = name_of, addr_of
        self.c, self.s = {}, {}

    def core(self, i):
        v = self.c.get(i)
        if v is None:
            v = self.c[i] = wcore(self.name_of[i])
        return v

    def st(self, i):
        v = self.s.get(i)
        if v is None:
            v = self.s[i] = wstreet(self.addr_of[i])
        return v


def classify(P, s1s, recs):
    """-> arrays: same_core (bool), cls (str: '' if not same-core/same-street/both numbers present)"""
    n = len(s1s)
    sc = np.zeros(n, bool); cls = np.full(n, "", dtype=object); nlen = np.zeros(n, np.int8)
    for j, (s, r) in enumerate(zip(s1s, recs)):
        cs = P.core(s)
        if not cs or cs != P.core(r):
            continue
        sc[j] = True
        (na, sa), (nb, sb) = P.st(s), P.st(r)
        if na is None or nb is None or not same_street(sa, sb):
            continue
        cls[j] = dclass(na, nb); nlen[j] = min(len(na), 4)
    return sc, cls, nlen


def f05(tp, fp, fn):
    if tp == 0:
        return 0.0 if (fp + fn) else 1.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 1.25 * p * r / (0.25 * p + r)


def shape_refs(kk, ids):
    allk = np.concatenate([kk.xs(s, level="src").reindex(ids, fill_value=0).values if s in kk.index.get_level_values(1) else np.zeros(len(ids), int)
                           for s in ("S2", "S3")])
    pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); sb = kv * pk / (kv * pk).sum()
    return float(sb[1]), float(pk[0])


def shape_f(kk, s1s, recs, REP1, ADD1):
    g = pd.DataFrame(dict(s1=s1s, src=[r[:2] for r in recs])).drop_duplicates()
    if len(g) == 0:
        return None
    kv = kk.reindex(pd.MultiIndex.from_frame(g)).fillna(0).values
    p1 = float((kv == 1).mean())
    return dict(n=int(len(kv)), p1=round(p1, 4), f_p1=round((REP1 - p1) / (REP1 - ADD1), 2),
                se=round(float(np.sqrt(p1 * (1 - p1) / len(kv)) / (REP1 - ADD1)), 2))


# =========================================================== TEST (S005 accepted)
T = load("test")
s1df = T["s1"]; recdf = pd.concat([T["s2"], T["s3"]])
name_of = dict(zip(s1df.id, s1df.name)); name_of.update(zip(recdf.id, recdf.name))
addr_of = dict(zip(s1df.id, s1df.addr)); addr_of.update(zip(recdf.id, recdf.addr))
PT = Parser(name_of, addr_of)
del recdf
TEST = {}
for cc in ("France", "US", "India"):
    a = accepted(f"S005_{cc}").reset_index(drop=True)
    sc, cls, nlen = classify(PT, a.s1.values, a.rec.values)
    a["sc"], a["cls"], a["nlen"] = sc, cls, nlen
    TEST[cc] = a
    fk = a[a.kept_final]
    L(cc, "accepted", len(a), "final", len(fk), "same-core", int(sc.sum()))
    o = dict(n_accepted=len(a), n_final=int(len(fk)), n_s1_country=int((s1df.country == cc).sum()))
    o["final_counts"] = fk[fk.cls.isin(CLASSES)].cls.value_counts().to_dict()
    o["final_shift_total"] = int(fk.cls.isin(CLASSES).sum())
    o["final_shift_total_excl_digindel_perm"] = int(fk.cls.isin([c for c in CLASSES if c not in ("digindel", "perm")]).sum())
    o["rate_of_all_accepted"] = {c: round(float((a.cls == c).mean()), 5) for c in CLASSES}
    o["rate_shift_total_of_all_accepted"] = round(float(a.cls.isin(CLASSES).mean()), 5)
    o["rate_shift_total_of_final"] = round(float(fk.cls.isin(CLASSES).mean()), 5)
    o["mean_p_final"] = fk[fk.cls.isin(CLASSES)].groupby("cls").p.mean().round(4).to_dict()
    o["n_s1_with_shift_final"] = int(fk[fk.cls.isin(CLASSES)].s1.nunique())
    # house-number length composition: share of accepted same-core same-street pairs (any class incl SAME) per S1-number length
    ss = a[a.cls != ""]
    o["nlen_share_same_core_street"] = (ss.nlen.value_counts(normalize=True).sort_index().round(4)).to_dict()
    tab = ss.groupby("nlen").cls.value_counts(normalize=True).unstack(fill_value=0)
    o["class_share_by_nlen"] = {int(k): {c: round(float(v), 5) for c, v in row.items() if c != "SAME"} for k, row in tab.iterrows()}
    R[f"test_{cc}"] = o
    L(json.dumps({k: o[k] for k in ("final_counts", "final_shift_total", "rate_shift_total_of_all_accepted", "mean_p_final")}))

# S1 house-number length distribution per country (all test S1)
hl = {}
for cc in ("France", "US", "India"):
    ids = s1df.id[s1df.country == cc].values
    ln = [len(PT.st(i)[0]) if PT.st(i)[0] else 0 for i in ids]
    hl[cc] = {int(k): round(v, 4) for k, v in (pd.Series(np.minimum(ln, 4)).value_counts(normalize=True).sort_index()).items()}
R["test_s1_housenum_len_share(0=none)"] = hl
L("house-number length share:", hl)

# ---------------- shape test per class on test final pairs (replicates D's estimator with WD classes)
SH = {}
for cc in ("France", "US", "India"):
    a = TEST[cc]; fk = a[a.kept_final]
    kk = fk.assign(src=fk.rec.str[:2]).groupby(["s1", "src"]).size()
    REP1, ADD1 = shape_refs(kk, s1df.id[s1df.country == cc].values)
    SH[cc] = dict(REP_p1=round(REP1, 4), ADD_p1=round(ADD1, 4))
    for c in CLASSES + ["SAME"]:
        sel = fk[fk.cls == c]
        if len(sel) >= 100:
            SH[cc][c] = shape_f(kk, sel.s1.values, sel.rec.values, REP1, ADD1)
    grp = fk[fk.cls.isin(["pm1", "pm2", "odd3_10"])]
    SH[cc]["pm1+pm2+odd3_10"] = shape_f(kk, grp.s1.values, grp.rec.values, REP1, ADD1)
R["shape_test_test"] = SH
L("shape test (test):"); L(pd.DataFrame({cc: {c: (f"{v['f_p1']:+.2f}+-{v['se']:.2f} n={v['n']}" if isinstance(v, dict) else v) for c, v in d.items()} for cc, d in SH.items()}).to_string())

# ---------------- France sibling check (label-free): another France S1 with the record's core name at the record's number+street
fr = TEST["France"]; frk = fr[fr.kept_final & fr.cls.isin(CLASSES)].copy()
frs_ids = s1df.id[s1df.country == "France"].values
by_core = collections.defaultdict(list)
for i in frs_ids:
    by_core[PT.core(i)].append(i)


def sib_info(s, r, idx):
    cr = PT.core(r); nr, sr = PT.st(r)
    others = [o for o in idx.get(cr, []) if o != s]
    exact = any(PT.st(o)[0] == nr and same_street(PT.st(o)[1], sr) for o in others)
    return len(others), exact


si = [sib_info(s, r, by_core) for s, r in zip(frk.s1.values, frk.rec.values)]
frk["n_samename_other"] = [x for x, _ in si]; frk["sib_exact"] = [y for _, y in si]
t = frk.groupby("cls").agg(n=("p", "size"), share_any_samename_S1=("n_samename_other", lambda x: round(float((x > 0).mean()), 4)),
                           share_sib_at_rec_number=("sib_exact", "mean"), share_multi_claimed=("n_claims", lambda x: round(float((x > 1).mean()), 4)),
                           mean_p=("p", "mean")).round(4)
L("France final shift pairs: sibling check"); L(t.to_string())
R["France_sibling"] = t.to_dict(orient="index")
R["France_sibling_total"] = dict(n=int(len(frk)), any_samename=int((frk.n_samename_other > 0).sum()), sib_exact=int(frk.sib_exact.sum()))

# ---------------- ESTIMATED impact bound on France macro F0.5 (other accepted pairs of the S1 assumed correct, no FN)
nall = len(s1df); nfr = len(frs_ids)
ntot = fr[fr.kept_final].groupby("s1").size()
IMP = {}
for nm, cl in (("pm1+pm2+odd3_10", ["pm1", "pm2", "odd3_10"]), ("all_shift_classes", CLASSES)):
    m = frk[frk.cls.isin(cl)].groupby("s1").size()
    gain = sum(1.0 - f05(ntot[s] - k, k, 0) for s, k in m.items())
    loss = sum(1.0 - f05(ntot[s] - k, 0, k) for s, k in m.items())
    o = dict(n_pairs=int(m.sum()), n_s1=int(len(m)), share_France_S1=round(len(m) / nfr, 5),
             oracle_gain_if_ALL_decoy_France_pts=round(100 * gain / nfr, 3), oracle_gain_if_ALL_decoy_LB_pts=round(100 * gain / nall, 4))
    for f in (0.04, 0.2, 0.35, 0.5):
        o[f"f={f}"] = dict(oracle_France_pts=round(100 * f * gain / nfr, 3), oracle_LB_pts=round(100 * f * gain / nall, 4),
                           drop_class_France_pts=round(100 * (f * gain - (1 - f) * loss) / nfr, 3),
                           drop_class_LB_pts=round(100 * (f * gain - (1 - f) * loss) / nall, 4))
    IMP[nm] = o
R["France_impact_ESTIMATED"] = IMP
L(json.dumps(IMP, indent=1))

# examples (France)
EX = []
for c in ("pm1", "pm2", "odd3_10"):
    for r in frk[frk.cls == c].sample(4, random_state=1).itertuples():
        EX.append(f"{c} p={r.p:.3f} nclaims={r.n_claims} samename_other={r.n_samename_other} sib={r.sib_exact} | {name_of[r.s1]} | {addr_of[r.s1]}  ->  {name_of[r.rec]} | {addr_of[r.rec]}")
R["France_examples"] = EX
for e in EX:
    L(" ", e)
del T, PT, name_of, addr_of, TEST, fr, frk

# =========================================================== V1 (labelled, US + India)
Tr = load("train")
s1t = Tr["s1"]; rect = pd.concat([Tr["s2"], Tr["s3"]])
name_of = dict(zip(s1t.id, s1t.name)); name_of.update(zip(rect.id, rect.name))
addr_of = dict(zip(s1t.id, s1t.addr)); addr_of.update(zip(rect.id, rect.addr))
ctry_of_s1 = dict(zip(s1t.id, s1t.country))
del rect
owner = dict(zip(Tr["gt"].rec, Tr["gt"].s1))
PV = Parser(name_of, addr_of)
m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"]); y = m["y"].astype(int)
s1v = m["s1_ids"][m["s1idx"]]; cand = m["cand"]; ctry = m["country"][m["s1idx"]]
sc, cls, nlen = classify(PV, s1v, cand)
V = pd.DataFrame(dict(s1=s1v, rec=cand, y=y, p=p, country=ctry, sc=sc, cls=cls, nlen=nlen))
V["acc"] = V.p >= NEW_TH; V["hard"] = (V.p >= 0.001) | (V.y == 1)
L("V1 rows", len(V), "same-core", int(sc.sum()), "shift-class rows", int(V.cls.isin(CLASSES).sum()))
LAB = {}
for cc in ("US", "India"):
    g = V[(V.country == cc) & (V.cls != "")]
    rows = {}
    for c in CLASSES + ["SAME"]:
        x = g[g.cls == c]; xh = x[x.hard]; xa = x[x.acc]
        rows[c] = dict(n_all=len(x), n_hard=len(xh), n_pos=int(x.y.sum()), p_match_hard=round(float(xh.y.mean()), 4) if len(xh) else None,
                       p_match_all=round(float(x.y.mean()), 4) if len(x) else None, n_acc=len(xa),
                       prec=round(float(xa.y.mean()), 4) if len(xa) else None, FP=int((xa.y == 0).sum()), FN=int(((x.y == 1) & ~x.acc).sum()))
    LAB[cc] = rows
    L(cc); L(pd.DataFrame(rows).T.to_string())
    n_s1 = int((m["country"] == cc).sum())
    sh = g[g.cls.isin(CLASSES)]
    LAB[cc + "_total_shift"] = dict(n_V1_S1=n_s1, FP=int((sh.acc & (sh.y == 0)).sum()), FN=int(((sh.y == 1) & ~sh.acc).sum()),
                                   n_acc=int(sh.acc.sum()), FP_per_1k_S1=round(1000 * float((sh.acc & (sh.y == 0)).sum()) / n_s1, 3))
    # accepted shift pairs as share of all accepted (V1, compare to test rates)
    va = V[(V.country == cc) & V.acc]
    LAB[cc + "_rate_shift_of_accepted"] = round(float(va.cls.isin(CLASSES).mean()), 5)
R["V1_labelled"] = LAB

# where do V1 negatives in the shift classes come from? owner of the record in gt + label-free sibling in the train S1 table
by_core_tr = {cc: collections.defaultdict(list) for cc in ("US", "India")}
for i, c in zip(s1t.id.values, s1t.country.values):
    by_core_tr[c][PV.core(i)].append(i)
SRC = {}
for cc in ("US", "India"):
    g = V[(V.country == cc) & V.cls.isin(CLASSES) & V.hard].copy()
    own = [owner.get(r) for r in g.rec]
    g["owned"] = [o is not None for o in own]
    g["owner_samecore"] = [(o is not None and PV.core(o) == PV.core(r)) for o, r in zip(own, g.rec)]
    g["owner_samenum"] = [(o is not None and PV.st(o)[0] == PV.st(r)[0]) for o, r in zip(own, g.rec)]
    # label-free sibling (train S1 table of the same country)
    sib = []
    for s, r in zip(g.s1, g.rec):
        cr = PV.core(r); nr, sr = PV.st(r)
        others = [o for o in by_core_tr[cc].get(cr, []) if o != s]
        sib.append((len(others), any(PV.st(o)[0] == nr and same_street(PV.st(o)[1], sr) for o in others)))
    g["n_samename_other"] = [a for a, _ in sib]; g["sib_exact"] = [b for _, b in sib]
    neg = g[g.y == 0]
    o = dict(n_hard=len(g), n_neg=len(neg), neg_owned_by_other_S1=round(float(neg.owned.mean()), 4),
             neg_owner_samecore=round(float(neg.owner_samecore.mean()), 4), neg_owner_samecore_samenum=round(float((neg.owner_samecore & neg.owner_samenum).mean()), 4),
             neg_unowned_distractor=round(float((~neg.owned).mean()), 4),
             P_match_given_sib_exact=round(float(g[g.sib_exact].y.mean()), 4) if g.sib_exact.any() else None, n_sib_exact=int(g.sib_exact.sum()),
             P_match_given_no_sib=round(float(g[~g.sib_exact].y.mean()), 4), n_no_sib=int((~g.sib_exact).sum()),
             share_any_samename_S1=round(float((g.n_samename_other > 0).mean()), 4), share_sib_exact=round(float(g.sib_exact.mean()), 4))
    SRC[cc] = o
    L(cc, "V1 shift-class hard candidates:", json.dumps(o))
    if cc == "US":
        EXV = []
        for r in g[(g.y == 0) & g.cls.isin(["pm1", "pm2"])].sample(6, random_state=0).itertuples():
            ow = owner.get(r.rec)
            EXV.append(f"{r.cls} y=0 p={r.p:.3f} acc={r.acc} sib={r.sib_exact} | {name_of[r.s1]} | {addr_of[r.s1]} -> {name_of[r.rec]} | {addr_of[r.rec]}"
                       f"  || owner: {(name_of[ow] + ' | ' + addr_of[ow]) if ow else 'NONE (unlinked distractor)'}")
        for r in g[(g.y == 0) & g.acc].itertuples():
            ow = owner.get(r.rec)
            EXV.append(f"ACCEPTED-FP {r.cls} p={r.p:.3f} sib={r.sib_exact} | {name_of[r.s1]} | {addr_of[r.s1]} -> {name_of[r.rec]} | {addr_of[r.rec]}"
                       f"  || owner: {(name_of[ow] + ' | ' + addr_of[ow]) if ow else 'NONE'}")
        R["V1_US_examples"] = EXV
        for e in EXV:
            L(" ", e)
R["V1_negative_sources"] = SRC

# ---------------- shape-test calibration on V1 (labels known): groups = V1 accepted (p>=NEW_TH), no max-claimer
CAL = {}
for cc in ("US", "India"):
    va = V[(V.country == cc) & V.acc]
    kk = va.assign(src=va.rec.str[:2]).groupby(["s1", "src"]).size()
    REP1, ADD1 = shape_refs(kk, m["s1_ids"][m["country"] == cc])
    o = dict(REP_p1=round(REP1, 4), ADD_p1=round(ADD1, 4))
    for nm, sel in (("all_accepted_TP", va[va.y == 1]), ("all_accepted_FP", va[va.y == 0]),
                    ("shift_all_accepted", va[va.cls.isin(CLASSES)]), ("shift_TP_only", va[va.cls.isin(CLASSES) & (va.y == 1)]),
                    ("pm1+pm2+odd3_10_accepted", va[va.cls.isin(["pm1", "pm2", "odd3_10"])]),
                    ("pm1+pm2+odd3_10_TP_only", va[va.cls.isin(["pm1", "pm2", "odd3_10"]) & (va.y == 1)]),
                    ("SAME_TP", va[(va.cls == "SAME") & (va.y == 1)])):
        o[nm] = shape_f(kk, sel.s1.values, sel.rec.values, REP1, ADD1)
        if o[nm] is not None:
            o[nm]["true_FP_frac"] = round(float((sel.y == 0).mean()), 4)
    CAL[cc] = o
    L(cc, "shape calibration V1:", json.dumps(o))
R["shape_calibration_V1"] = CAL
json.dump(R, open(os.path.join(OUT, "WD_verify.json"), "w"), indent=1, default=str)
L("wrote WD_verify.json")
