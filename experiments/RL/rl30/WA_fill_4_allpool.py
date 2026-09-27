"""RL-30 adversarial verifier WA (filler claim), PART 4 (READ-ONLY): same as PART 2 but over the WHOLE France retrieval pool
(all 33.2M (S1, candidate) rows, any address), so the support count is not limited by the same-street filter.
Key-based, accent-folded tokens (rl30_lib.toks):
  core(x) = tokens - FILLER(A role A) - LEGAL - HONOR
  filler_only_any : core(S1) == core(rec), non-empty, and the symmetric token difference contains >=1 non-legal filler
  filler_sub_any  : core(rec) == core(S1) minus exactly ONE token, and the record carries >=1 non-legal filler absent from the S1
status: NA (not accepted by S005 => p<0.78), LO, HI; street = same house number + street tokens; exact = same canonical address.
Bounds (ESTIMATE): per-S1 F0.5 if every NA pair of the class were added, all TP vs all FP (other kept pairs assumed TP, no other misses).
Output: WA_fill_4_results.json
"""
import os, sys, json, time, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
FILLER = set(roles.index[roles.role == "A"])
FNL = FILLER - LEGAL
DROP = FILLER | LEGAL | HONOR
FR_ONLY = {"cie", "fils", "associes", "compagnie", "developpement", "et", "frs"}


def f05(P, R):
    return 0.0 if P + R == 0 else 1.25 * P * R / (0.25 * P + R)


def skey(addr):
    n, s, _ = street_parts(addr)
    return f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else ""


D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id"); REC = REC[REC.country == "France"]
N_ALL = len(D["s1"]); N_FR = len(s1)
s1_t = [frozenset(toks(n)) for n in s1.name.values]
rec_t = [frozenset(toks(n)) if isinstance(n, str) else frozenset() for n in REC.name.values]
s1_core = [" ".join(sorted(t - DROP)) for t in s1_t]
rec_core = [" ".join(sorted(t - DROP)) for t in rec_t]
vocab = pd.Index(pd.unique(np.array(s1_core + rec_core, dtype=object)))
s1_cc = pd.Series(vocab.get_indexer(s1_core), index=s1.index)
rec_cc = pd.Series(vocab.get_indexer(rec_core), index=REC.index)
empty_code = vocab.get_indexer([""])[0]
rec_hasf = pd.Series([bool(t & FNL) for t in rec_t], index=REC.index)
# minus-one keys of S1 cores
m1 = []
for sid, t in zip(s1.index, s1_t):
    c = sorted(t - DROP)
    if len(c) >= 2:
        for i in range(len(c)):
            m1.append((sid, " ".join(c[:i] + c[i + 1:])))
m1 = pd.DataFrame(m1, columns=["s1", "key"])
m1 = m1[m1.key.isin(vocab)]
s1_code_idx = pd.Series(np.arange(N_FR), index=s1.index)
M = np.int64(len(vocab) + 1)
m1_hash = np.unique(s1_code_idx.reindex(m1.s1.values).values.astype(np.int64) * M + vocab.get_indexer(m1.key.values).astype(np.int64))
s1_sk = pd.Series([skey(x) for x in s1.addr.values], index=s1.index); rec_sk = pd.Series([skey(x) for x in REC.addr.values], index=REC.index)
s1_ak = s1.addr.map(akey); rec_ak = REC.addr.map(akey)
s1_tok = dict(zip(s1.index, s1_t)); rec_tok = dict(zip(REC.index, rec_t))
L("prep", N_FR, len(REC), "vocab", len(vocab), "minus-one keys", len(m1_hash))

acc = accepted("S005_France")
acc_idx = pd.MultiIndex.from_arrays([acc.s1.values, acc.rec.values])
acc_p = pd.Series(acc.p.values, index=acc_idx)
kept_per_s1 = acc[acc.kept_final].groupby("s1").size()
cnt = Counter(); na = defaultdict(lambda: defaultdict(int)); ex = defaultdict(list)
n_pool = 0
for fi, f in enumerate(sorted(glob.glob(PATHS["test_chunks"].format(country="France")))):
    z = np.load(f, allow_pickle=True); sa, ca = z["s1"], z["cand"]; n_pool += len(sa)
    a1 = s1_cc.reindex(sa).values; a2 = rec_cc.reindex(ca).values
    fo = (a1 == a2) & (a1 != empty_code)
    hf = rec_hasf.reindex(ca).values.astype(bool)
    h = s1_code_idx.reindex(sa).values.astype(np.int64) * M + a2.astype(np.int64)
    fs = hf & (a2 != empty_code) & np.isin(h, m1_hash)
    for cl, mask in (("filler_only_any", fo), ("filler_sub_any", fs)):
        idx = np.where(mask)[0]
        if not len(idx): continue
        pv = acc_p.reindex(pd.MultiIndex.from_arrays([sa[idx], ca[idx]])).values
        for q, j in enumerate(idx):
            s, c = sa[j], ca[j]; A_, B_ = s1_tok[s], rec_tok[c]
            d = A_ ^ B_
            if cl == "filler_only_any" and not (d & FNL):
                continue
            if cl == "filler_sub_any" and not ((B_ - A_) & FNL):
                continue
            p = pv[q]; stt = "NA" if not (p == p) else ("HI" if p >= 0.99 else "LO")
            geo = "exact" if (s1_ak[s] == rec_ak[c] and s1_ak[s]) else ("street" if (s1_sk[s] and s1_sk[s] == rec_sk[c]) else "other_addr")
            fr = bool((B_ - A_) & FR_ONLY)
            cnt[(cl, stt, geo)] += 1
            if fr: cnt[(cl + "_fr_only", stt, geo)] += 1
            if stt == "NA":
                na[cl][s] += 1; na[f"{cl}|{geo}"][s] += 1
                if len(ex[(cl, geo)]) < 12 and (j % 97 == 0):
                    ex[(cl, geo)].append(dict(s1=s1.name[s], s1_addr=s1.addr[s], rec=REC.name[c], rec_addr=REC.addr[c]))
    if fi % 40 == 0:
        L(f"chunk {fi} pool {n_pool:,}", sum(v for k, v in cnt.items() if not k[0].endswith("fr_only")))
R = dict(n_pool_rows=n_pool, filler_set=sorted(FILLER))
tab = defaultdict(dict)
for (cl, stt, geo), v in sorted(cnt.items()): tab[cl][f"{stt}|{geo}"] = v
for cl in list(tab):
    tot = Counter()
    for k, v in tab[cl].items(): tot[k.split("|")[0]] += v
    tab[cl]["TOTAL"] = dict(tot)
R["class_status_geo"] = tab
bounds = {}
for cl, dd in na.items():
    g = l = 0.0; n0 = 0
    for s, k in dd.items():
        mk = int(kept_per_s1.get(s, 0)); n0 += mk == 0
        if mk > 0:
            g += 1 - f05(1.0, mk / (mk + k)); l += 1 - f05(mk / (mk + k), 1.0)
        else:
            g += 1.0; l += 1.0
    bounds[cl] = dict(n_pairs=int(sum(dd.values())), n_s1=len(dd), n_s1_no_kept_pair=n0,
                      dLB_pp_if_all_TP=round(100 * g / N_ALL, 4), dLB_pp_if_all_FP=round(-100 * l / N_ALL, 4),
                      dFrance_pp_if_all_TP=round(100 * g / N_FR, 3), breakeven_TP_fraction=round(l / (g + l), 3) if g + l else None)
R["NA_bounds"] = bounds
R["NA_examples"] = {f"{a}|{b}": v for (a, b), v in ex.items()}
json.dump(R, open(os.path.join(OUT, "WA_fill_4_results.json"), "w"), indent=1, ensure_ascii=False, default=str)
L(json.dumps({k: v for k, v in R.items() if k != "NA_examples" and k != "filler_set"}, default=str)[:5000])
