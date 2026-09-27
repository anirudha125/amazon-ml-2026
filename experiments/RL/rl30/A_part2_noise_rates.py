"""RL-30 investigator A, PART 2 (READ-ONLY).
Token NOISE rates = how often the record generator adds / drops a token for the SAME business.
 (a) TEST, label-free: pseudo-positives = accepted('S005_<C>') with p>=0.99, n_claims==1, kept_final, and record house number
     + street_name tokens == S1's (street_parts).  Also a lower band 0.78<=p<0.99 (kept_final) for comparison.
 (b) TRAIN, label-backed (US/India): all GT links with the same street filter (no model selection) -> unbiased noise rates.
 (c) V1, label-backed selection-bias check: RL-27 NEW OOF p; noise among y=1 & same-street, p>=0.99 vs all p.
Typo handling: an S1-only token and a record-only token with Levenshtein sim >= 0.75 (or distance 1) are paired as a TYPO
substitution and removed from the pure add/drop counts.
Outputs: A_p2_noise_{C}_{src}.csv, A_p2_subst_{C}.csv, A_p2_summary.json
"""
import os, sys, json, time, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s)
    s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    if a.isdigit() or b.isdigit():
        return False
    return LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75


def same_street(a1, a2):
    n1, s1, _ = street_parts(a1)
    n2, s2, _ = street_parts(a2)
    return n1 is not None and n1 == n2 and bool(s1) and s1 == s2


class Acc:
    def __init__(s):
        s.s1has = Counter(); s.rechas = Counter(); s.both = Counter(); s.drop = Counter(); s.add = Counter()
        s.pdrop = Counter(); s.padd = Counter(); s.typo = Counter(); s.subst = Counter(); s.tsubst = Counter()
        s.n = 0; s.n_ident = 0; s.n_core_ident = 0; s.n_pure_zero = 0; s.n_one_sub = 0; s.n_one_sub_typo = 0
        s.n_pure_drop1 = 0; s.n_pure_add1 = 0; s.n_sets = Counter()

    def add_pair(s, A, B):
        """A = S1 tokens, B = record tokens"""
        s.n += 1
        sh, oa, ob = A & B, A - B, B - A
        for t in A: s.s1has[t] += 1
        for t in B: s.rechas[t] += 1
        for t in sh: s.both[t] += 1
        for t in oa: s.drop[t] += 1
        for t in ob: s.add[t] += 1
        if not oa and not ob: s.n_ident += 1
        ca = frozenset(t for t in A if t not in LEGAL and t not in HONOR)
        cb = frozenset(t for t in B if t not in LEGAL and t not in HONOR)
        if ca == cb: s.n_core_ident += 1
        # typo pairing (greedy by similarity)
        pa, pb = set(oa), set(ob)
        if pa and pb:
            cands = sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True)
            for _, a, b in cands:
                if a in pa and b in pb:
                    pa.discard(a); pb.discard(b); s.typo[a] += 1; s.tsubst[(a, b)] += 1
        for t in pa: s.pdrop[t] += 1
        for t in pb: s.padd[t] += 1
        if not pa and not pb: s.n_pure_zero += 1
        if len(pa) == 1 and not pb: s.n_pure_drop1 += 1
        if len(pb) == 1 and not pa: s.n_pure_add1 += 1
        if len(oa) == 1 and len(ob) == 1:
            a, b = next(iter(oa)), next(iter(ob))
            if typo(a, b): s.n_one_sub_typo += 1
            else:
                s.n_one_sub += 1; s.subst[(a, b)] += 1

    def table(s):
        toks_ = set(s.s1has) | set(s.rechas)
        rows = []
        for t in toks_:
            both, dr, ad, pdr, pad = s.both[t], s.drop[t], s.add[t], s.pdrop[t], s.padd[t]
            occ = both + dr + ad
            rows.append(dict(tok=t, legal=t in LEGAL, s1has=s.s1has[t], rechas=s.rechas[t], both=both, drop=dr, add=ad,
                             pdrop=pdr, padd=pad, typo=s.typo[t], occ=occ,
                             noise_raw=(dr + ad) / occ, noise_pure=(pdr + pad) / occ,
                             drop_rate=pdr / s.s1has[t] if s.s1has[t] else np.nan,
                             add_rel=pad / s.s1has[t] if s.s1has[t] else np.nan,
                             add_per_1k_pairs=1000 * pad / s.n, typo_rate=s.typo[t] / s.s1has[t] if s.s1has[t] else np.nan))
        return pd.DataFrame(rows)

    def summ(s):
        return dict(pairs=s.n, frac_name_identical=s.n_ident / s.n, frac_core_identical=s.n_core_ident / s.n,
                    frac_no_pure_diff=s.n_pure_zero / s.n, frac_pure_drop1=s.n_pure_drop1 / s.n, frac_pure_add1=s.n_pure_add1 / s.n,
                    frac_one_subst_word=s.n_one_sub / s.n, frac_one_subst_typo=s.n_one_sub_typo / s.n)


summary = {}
D = load("test", verbose=False)
S1 = D["s1"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
for C in ("France", "US", "India"):
    a = accepted(f"S005_{C}")
    a = a[a.kept_final]
    if C != "France":
        a = a.sample(n=min(len(a), 1_200_000), random_state=0)      # US/India: 1.2M-row random sample of kept pairs
    s1n = S1.name.reindex(a.s1.values).values; s1a = S1.addr.reindex(a.s1.values).values
    rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
    hi = ((a.p >= 0.99) & (a.n_claims == 1)).values
    AH, AL = Acc(), Acc()
    n_hi = n_hi_street = n_lo = n_lo_street = 0
    for k in range(len(a)):
        is_hi = hi[k]
        if not is_hi and a.p.values[k] >= 0.99:
            continue                                  # multi-claimed high-p: excluded from both bands
        if is_hi: n_hi += 1
        else: n_lo += 1
        if not same_street(s1a[k], ra[k]):
            continue
        A, B = ntoks(s1n[k]), ntoks(rn[k])
        if is_hi:
            n_hi_street += 1; AH.add_pair(A, B)
        else:
            n_lo_street += 1; AL.add_pair(A, B)
    th, tl = AH.table(), AL.table()
    th.to_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), index=False)
    tl.to_csv(os.path.join(OUT, f"A_p2_noise_{C}_testLO.csv"), index=False)
    sub = pd.DataFrame([dict(s1_tok=x, rec_tok=y, n=n) for (x, y), n in AH.subst.most_common(2000)])
    sub.to_csv(os.path.join(OUT, f"A_p2_subst_{C}.csv"), index=False)
    tsub = pd.DataFrame([dict(s1_tok=x, rec_tok=y, n=n) for (x, y), n in AH.tsubst.most_common(2000)])
    tsub.to_csv(os.path.join(OUT, f"A_p2_typosubst_{C}.csv"), index=False)
    summary[f"{C}_testPP"] = dict(accepted_hi=n_hi, pseudo_pos=n_hi_street, street_pass=n_hi_street / max(n_hi, 1), **AH.summ())
    summary[f"{C}_testLO"] = dict(accepted_lo=n_lo, lo_street=n_lo_street, street_pass=n_lo_street / max(n_lo, 1), **AL.summ())
    print(C, json.dumps(summary[f"{C}_testPP"]), json.dumps(summary[f"{C}_testLO"]), f"{time.time() - T0:.0f}s", flush=True)
del REC, S1, D

# ---------------- (b) TRAIN GT links, label-backed, no model selection
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
gt = DT["gt"]
gt = gt.sample(n=min(len(gt), 1_600_000), random_state=0)
ctry = S1t.country.reindex(gt.s1.values).values
g1n = S1t.name.reindex(gt.s1.values).values; g1a = S1t.addr.reindex(gt.s1.values).values
grn = RECt.name.reindex(gt.rec.values).values; gra = RECt.addr.reindex(gt.rec.values).values
for C in ("US", "India"):
    AG = Acc(); n_all = n_st = 0
    for k in np.where(ctry == C)[0]:
        if not isinstance(grn[k], str):
            continue
        n_all += 1
        if not same_street(g1a[k], gra[k]):
            continue
        n_st += 1
        AG.add_pair(ntoks(g1n[k]), ntoks(grn[k]))
    AG.table().to_csv(os.path.join(OUT, f"A_p2_noise_{C}_trainGT.csv"), index=False)
    summary[f"{C}_trainGT"] = dict(gt_links_sampled=n_all, street_pass=n_st / max(n_all, 1), **AG.summ())
    print(C, "trainGT", json.dumps(summary[f"{C}_trainGT"]), f"{time.time() - T0:.0f}s", flush=True)

# ---------------- (c) V1 selection-bias check (RL-27 NEW OOF p, labels)
m = np.load(PATHS["v1_meta"], allow_pickle=True)
p = np.load(PATHS["v1_p_new"])
y = m["y"]; s1ids = m["s1_ids"][m["s1idx"]]; cand = m["cand"]; vc = m["country"][m["s1idx"]]
sel = np.where(y == 1)[0]
v1n = S1t.name.reindex(s1ids[sel]).values; v1a = S1t.addr.reindex(s1ids[sel]).values
vrn = RECt.name.reindex(cand[sel]).values; vra = RECt.addr.reindex(cand[sel]).values
for C in ("US", "India"):
    AA, AH9 = Acc(), Acc()
    for q, k in enumerate(sel):
        if vc[k] != C or not isinstance(vrn[q], str) or not same_street(v1a[q], vra[q]):
            continue
        A, B = ntoks(v1n[q]), ntoks(vrn[q])
        AA.add_pair(A, B)
        if p[k] >= 0.99:
            AH9.add_pair(A, B)
    AA.table().to_csv(os.path.join(OUT, f"A_p2_noise_{C}_V1pos.csv"), index=False)
    AH9.table().to_csv(os.path.join(OUT, f"A_p2_noise_{C}_V1pos_p99.csv"), index=False)
    summary[f"{C}_V1pos"] = AA.summ(); summary[f"{C}_V1pos_p99"] = AH9.summ()
    print(C, "V1", json.dumps(AA.summ()), json.dumps(AH9.summ()), f"{time.time() - T0:.0f}s", flush=True)

json.dump(summary, open(os.path.join(OUT, "A_p2_summary.json"), "w"), indent=1)
print("done", f"{time.time() - T0:.0f}s")
