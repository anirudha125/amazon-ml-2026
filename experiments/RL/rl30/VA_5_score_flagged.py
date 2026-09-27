"""RL-30 verifier VA, step 5 (France TEST, inference only, CPU, no training, nothing written outside rl30/VA_*).
S005's saved France predictions keep only p >= 0.78, so the p of the REJECTED filler-only pairs is unknown. Re-score exactly the
S005 France stage 2 (rl27_model_NEW_s42.pkl through rl29_score's feature assembly: LF[:22] | block A | LF[22:] | rank_dense | dcos |
rrUb (top-10 by base, cached) | RL-27 test features) and keep p (and the stage-2 feature row) for every same-street flagged pair
(VA_2 flags).  Parity: recomputed p of the accepted flagged pairs must equal the stored accepted p.
Writes VA_5_scored.pkl (flagged rows: s1, rec, kind, p, base p, base rank) + VA_5_X.npy (their stage-2 feature rows) + VA_5_score.json."""
import os, sys, json, time, re, glob, pickle
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
os.environ["LGB_THREADS"] = "1"
sys.path.insert(0, RL)
import rl29_score as RS
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s if isinstance(s, str) else ""); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


R = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
R = R[R.occ >= 300]
FILL = set(R.index[R.add_LR >= 0.05]); DROP = set(R.index[R.drop_rate.fillna(0) >= 0.20]); STRIP = FILL | DROP | LEGAL | HONOR
D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].reset_index(drop=True)
rec = pd.concat([D["s2"], D["s3"]]); rec = rec[rec.country == "France"].reset_index(drop=True)
del D


def skey(a):
    n_, st, _ = street_parts(a if isinstance(a, str) else "")
    return None if (n_ is None or not st) else n_ + "|" + " ".join(sorted(st))


codes, _ = pd.factorize(pd.concat([s1.addr.map(skey), rec.addr.map(skey)], ignore_index=True), use_na_sentinel=True)
s1_code = codes[:len(s1)]; rec_code = codes[len(s1):]
s1_tok = [ntoks(x) for x in s1.name.values]; rec_name = rec.name.values
s1_ix = pd.Index(s1.id.values); rec_ix = pd.Index(rec.id.values)
M = RS.load_model(os.path.join(RL, "rl27_model_NEW_s42.pkl"))
RR = pickle.load(open(os.path.join(ROOT, "experiments", "P3", RS.RRC.format("France")), "rb"))
print("setup", f"{time.time() - T0:.0f}s", "th", M["th"], "n_feat", M["n_feat"], flush=True)


def work(path):
    z = np.load(path); cs1, cc = z["s1"], z["cand"]
    i1 = s1_ix.get_indexer(cs1); i2 = rec_ix.get_indexer(cc); ok = (i1 >= 0) & (i2 >= 0)
    c1 = np.where(ok, s1_code[np.clip(i1, 0, None)], -1); c2 = np.where(ok, rec_code[np.clip(i2, 0, None)], -2)
    tgt, kinds = [], []
    for k in np.flatnonzero((c1 >= 0) & (c1 == c2)):
        A = s1_tok[i1[k]]; B = ntoks(rec_name[i2[k]]); ob = B - A
        if not ob: continue
        pa, pb = set(A - B), set(ob)
        if pa and pb:
            for _, x, zz in sorted(((LV.normalized_similarity(x, zz), x, zz) for x in pa for zz in pb if typo(x, zz)), reverse=True):
                if x in pa and zz in pb: pa.discard(x); pb.discard(zz)
        if not pb or not all(t in FILL for t in pb): continue
        rest = [t for t in pa if t not in STRIP]
        tgt.append(k); kinds.append("F_only_st" if not rest else ("F_sub1_st" if len(rest) == 1 else "F_add_other_st"))
    if not tgt:
        return None
    tgt = np.array(tgt)
    LF, rd, dc = z["LF"], z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(cs1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb_ = M["base"].predict_proba(Xb)[:, 1]
    sel = S2.topk_mask(s1idx, pb_, 10)
    col = np.full(len(cs1), np.nan, np.float32)
    for i in np.flatnonzero(sel):
        v = RR.get((str(cs1[i]), str(cc[i])))
        if v is not None: col[i] = v
    BA = S2.block_a_vec(s1idx, pb_)
    F = np.load(os.path.join(RL, "test_feats", "France", "rl27_" + os.path.basename(path).replace(".npz", ".npy")))
    X = np.hstack([LF[tgt, :22], BA[tgt], LF[tgt, 22:], rd[tgt].astype(np.float32)[:, None], dc[tgt].astype(np.float32)[:, None],
                   col[tgt, None], F[tgt]]).astype(np.float32)
    assert X.shape[1] == M["n_feat"]
    p = M["clf"].predict_proba(X)[:, 1]
    # base rank of the target within its S1
    order = np.lexsort((-pb_, s1idx)); rank = np.empty(len(pb_), np.int32)
    g = s1idx[order]; starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, len(g)])
    rank[order] = np.arange(len(g)) - np.repeat(starts, K)
    return cs1[tgt], cc[tgt], np.array(kinds), p, pb_[tgt], rank[tgt], X


paths = sorted(glob.glob(PATHS["test_chunks"].format(country="France")))
res = []
with mp.get_context("fork").Pool(4) as pool:
    for j, r in enumerate(pool.imap(work, paths)):
        if r is not None: res.append(r)
        if j % 20 == 0: print(j, len(paths), f"{time.time() - T0:.0f}s", flush=True)
S = pd.DataFrame(dict(s1=np.concatenate([r[0] for r in res]), rec=np.concatenate([r[1] for r in res]), kind=np.concatenate([r[2] for r in res]),
                      p=np.concatenate([r[3] for r in res]), p_base=np.concatenate([r[4] for r in res]), base_rank=np.concatenate([r[5] for r in res])))
X = np.vstack([r[6] for r in res])
a = accepted("S005_France")
S = S.merge(a[["s1", "rec", "p", "kept_final"]].rename(columns={"p": "p_stored"}), on=["s1", "rec"], how="left")
m = S.p_stored.notna()
out = dict(n_flagged=int(len(S)), parity_n=int(m.sum()), parity_max_abs_diff=float((S.p[m] - S.p_stored[m]).abs().max()),
           n_recomputed_ge_th_but_not_stored=int(((S.p >= M["th"]) & ~m).sum()), n_stored_but_recomputed_lt_th=int(((S.p < M["th"]) & m).sum()))
bins = [0, 0.01, 0.05, 0.2, 0.5, 0.78, 0.9, 0.99, 1.0001]
S["pbin"] = pd.cut(S.p, bins, right=False).astype(str)
out["p_bins_by_kind"] = {kd: g.pbin.value_counts().sort_index().to_dict() for kd, g in S.groupby("kind")}
rej = S[S.p < M["th"]]
out["rejected_by_kind_base_rank_lt10"] = {kd: dict(n=int(len(g)), in_base_top10=int((g.base_rank < 10).sum()), median_p=float(g.p.median()),
                                                     median_p_base=float(g.p_base.median())) for kd, g in rej.groupby("kind")}
out["secs"] = round(time.time() - T0, 1)
S.to_pickle(os.path.join(OUT, "VA_5_scored.pkl")); np.save(os.path.join(OUT, "VA_5_X.npy"), X)
json.dump(out, open(os.path.join(OUT, "VA_5_score.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)); print("done", f"{time.time() - T0:.0f}s")
