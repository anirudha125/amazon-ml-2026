"""RL-30 adversarial verifier WA, PART 2 (READ-ONLY): decision-relevant support of the 'filler token' signal in FRANCE.
The claim's evidence lives entirely in ACCEPTED pairs (p>=0.78). A feature that says "record-only filler is noise" can only change
a decision where a filler-bearing pair is currently REJECTED (p<0.78: potential FN) -- raising p inside the accepted band changes nothing.
Here: every same-street (house number + street tokens) (S1, candidate) pair of the France retrieval pool (P3 chunks, all 174),
classified by the name difference (accent-folded tokens, typo-paired):
  ident       no pure token difference
  legal_only  all differing tokens are legal forms
  filler_only all differing tokens are A's role-A ('filler') tokens, >=1 non-legal filler
  filler_sub  record-only tokens are all fillers (>=1 non-legal) and exactly ONE non-filler S1 token is dropped ('Club' -> 'Associes')
  other
status: NA (not accepted by S005 => p<0.78), LO (0.78<=p<0.99), HI (p>=0.99); kept_final flag.
Also: co-located ambiguity (another S1 at the same canonical address whose core covers the record's core minus fillers), per-S1 F0.5
upper/lower bounds if every NA filler pair were added (all TP vs all FP; ESTIMATE, other kept pairs assumed TP, no other misses).
Output: WA_fill_2_results.json, WA_fill_2_examples.json
"""
import os, sys, json, time, glob, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
FILLER = set(roles.index[roles.role == "A"])
FILLER_NL = FILLER - LEGAL
FR_ONLY = {"cie", "fils", "associes", "compagnie", "developpement", "et", "frs"}
L("filler set", len(FILLER), sorted(FILLER))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


def skey(addr):
    n, s, _ = street_parts(addr)
    return f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else ""


def classify(pa, pb):
    d = pa | pb
    if not d:
        return "ident"
    if d <= LEGAL:
        return "legal_only"
    if d <= FILLER:
        return "filler_only"
    if pb and pb <= FILLER and (pb - LEGAL) and len(pa - FILLER) == 1:
        return "filler_sub"
    return "other"


def f05(P, R):
    return 0.0 if P + R == 0 else 1.25 * P * R / (0.25 * P + R)


D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
REC = REC[REC.country == "France"]
N_ALL = len(D["s1"]); N_FR = len(s1)
s1_sk = pd.Series([skey(x) for x in s1.addr.values], index=s1.index)
rec_sk = pd.Series([skey(x) for x in REC.addr.values], index=REC.index)
s1_ak = s1.addr.map(akey)
s1_tok = {i: frozenset(toks(n)) for i, n in zip(s1.index, s1.name.values)}
byak = defaultdict(list)
for i, k in zip(s1_ak.index, s1_ak.values):
    if k.strip(): byak[k].append(i)
L("prep", N_FR, len(REC))

acc = accepted("S005_France")
acc_idx = pd.MultiIndex.from_arrays([acc.s1.values, acc.rec.values])
acc_p = pd.Series(acc.p.values, index=acc_idx); acc_k = pd.Series(acc.kept_final.values, index=acc_idx)
kept_per_s1 = acc[acc.kept_final].groupby("s1").size()
acc_per_s1 = acc.groupby("s1").size()

cnt = Counter(); fr_cnt = Counter(); n_pool = 0; n_street = 0; found_acc = 0
na_pairs = defaultdict(lambda: defaultdict(int))     # class -> s1 -> #NA pairs
amb = Counter(); ex = defaultdict(list); rnd = random.Random(0)
ident_acc_s1 = set()
rows_keep = []
files = sorted(glob.glob(PATHS["test_chunks"].format(country="France")))
for fi, f in enumerate(files):
    z = np.load(f, allow_pickle=True)
    sa, ca = z["s1"], z["cand"]
    n_pool += len(sa)
    k1 = s1_sk.reindex(sa).values; k2 = rec_sk.reindex(ca).values
    m = (k1 == k2) & (k1 != "") & pd.notna(k1)
    idx = np.where(m)[0]
    n_street += len(idx)
    mi = pd.MultiIndex.from_arrays([sa[idx], ca[idx]])
    pv = acc_p.reindex(mi).values; kv = acc_k.reindex(mi).values
    rn = REC.name.reindex(ca[idx]).values
    for q, j in enumerate(idx):
        s, c = sa[j], ca[j]
        A_ = s1_tok[s]; B_ = frozenset(toks(rn[q])) if isinstance(rn[q], str) else frozenset()
        pa, pb = pure(A_, B_)
        cl = classify(pa, pb)
        p = pv[q]
        stt = "NA" if not (p == p) else ("HI" if p >= 0.99 else "LO")
        kf = bool(kv[q]) if stt != "NA" else False
        cnt[(cl, stt)] += 1
        if stt != "NA": found_acc += 1
        if kf: cnt[(cl, stt + "_kept")] += 1
        fro = bool(pb & FR_ONLY)
        if cl in ("filler_only", "filler_sub") and fro:
            fr_cnt[(cl, stt)] += 1
        if cl == "ident" and stt != "NA" and kf:
            ident_acc_s1.add(s)
        if cl in ("filler_only", "filler_sub"):
            rcore = frozenset(t for t in B_ if t not in FILLER and t not in LEGAL and t not in HONOR)
            own = frozenset(t for t in A_ if t not in FILLER and t not in LEGAL and t not in HONOR)
            others = [o for o in byak.get(s1_ak[s], []) if o != s]
            cov = [o for o in others if rcore and rcore <= frozenset(t for t in s1_tok[o] if t not in FILLER and t not in LEGAL and t not in HONOR)
                   and frozenset(t for t in s1_tok[o] if t not in FILLER and t not in LEGAL and t not in HONOR) != own]
            amb[(cl, stt, "n")] += 1
            amb[(cl, stt, "has_coloc")] += bool(others)
            amb[(cl, stt, "ambiguous")] += bool(cov)
            if stt == "NA":
                na_pairs[cl][s] += 1
                if fro: na_pairs[cl + "_fr_only"][s] += 1
            if len(ex[(cl, stt)]) < 25 and rnd.random() < 0.02:
                ex[(cl, stt)].append(dict(s1=s1.name[s], rec=rn[q], p=None if stt == "NA" else round(float(p), 4), kept=kf,
                                          s1_only=sorted(pa), rec_only=sorted(pb), coloc=[s1.name[o] for o in others][:5], ambiguous=bool(cov)))
    if fi % 20 == 0:
        L(f"chunk {fi}/{len(files)} pool {n_pool:,} street {n_street:,}")

R = dict(filler_set=sorted(FILLER), n_pool_rows=n_pool, n_same_street_rows=n_street, n_accepted_total=int(len(acc)),
         n_accepted_found_on_street=found_acc)
tab = defaultdict(dict)
for (cl, stt), v in cnt.items(): tab[cl][stt] = v
R["class_by_status"] = tab
R["fr_only_adders_class_by_status"] = {f"{cl}|{stt}": v for (cl, stt), v in fr_cnt.items()}
R["ambiguity"] = {f"{cl}|{stt}|{k}": v for (cl, stt, k), v in sorted(amb.items())}
bounds = {}
for cl, d in na_pairs.items():
    gain_tp = loss_fp = 0.0; n0 = 0; n_ident = 0
    for s, k in d.items():
        mk = int(kept_per_s1.get(s, 0))
        n0 += mk == 0; n_ident += s in ident_acc_s1
        if mk > 0:
            gain_tp += 1 - f05(1.0, mk / (mk + k)); loss_fp += 1 - f05(mk / (mk + k), 1.0)
        else:
            gain_tp += 1.0; loss_fp += 1.0
    bounds[cl] = dict(n_pairs=int(sum(d.values())), n_s1=len(d), n_s1_no_kept_pair=n0, n_s1_with_kept_ident_record=n_ident,
                      dLB_pp_if_all_TP_added=round(100 * gain_tp / N_ALL, 4), dLB_pp_if_all_FP_added=round(-100 * loss_fp / N_ALL, 4),
                      dFrance_pp_if_all_TP=round(100 * gain_tp / N_FR, 3), dFrance_pp_if_all_FP=round(-100 * loss_fp / N_FR, 3),
                      breakeven_TP_fraction=round(loss_fp / (gain_tp + loss_fp), 3) if gain_tp + loss_fp else None)
R["NA_bounds"] = bounds
R["N_all_test_S1"] = N_ALL; R["N_France_S1"] = N_FR
json.dump(R, open(os.path.join(OUT, "WA_fill_2_results.json"), "w"), indent=1, ensure_ascii=False, default=str)
json.dump({f"{a}|{b}": v for (a, b), v in ex.items()}, open(os.path.join(OUT, "WA_fill_2_examples.json"), "w"), indent=1, ensure_ascii=False)
L(json.dumps(R, default=str)[:4000])
