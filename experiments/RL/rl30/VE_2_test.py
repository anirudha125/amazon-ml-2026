"""RL-30 / VE -- label-free test-side checks of E's F2 flag against what S005 (RL-27 NEW + max-claimer) already does.
For every S005 accepted pair (before max-claimer) whose S1 has a co-located near-duplicate sibling (E_roles_test.pkl), recompute
F2_ge / F2_gt, then for the flagged pairs: who else claims the record, does the sibling claim it, and with what p; for the pairs the
max-claimer keeps, fetch the RL-27 test columns of (this S1, rec) and (sibling, rec) from the P3 chunks (is the sibling even a
candidate of the record?).  READ-ONLY.  Output: rl30/VE_2_test.json"""
import os, sys, json, time, glob, pickle, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
C = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]


def main():
    t0 = time.time()
    T = load("test", verbose=False)
    s1all = T["s1"]; S1 = s1all.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb")); SIB = RO["sib"]
    need = set(SIB.keys())
    ak_all = dict(zip(s1all.id.values, (akey(a) for a in s1all.addr.values)))
    grp = collections.defaultdict(list)
    for sid, c in zip(s1all.id.values, s1all.country.values):
        k = ak_all[sid]
        if k:
            grp[c + "|" + k].append(sid)
    NS = {}
    sib_ids = {}
    for sid in need:
        c = S1.country.loc[sid]; L = grp[c + "|" + ak_all[sid]]
        A = nset(S1.name.loc[sid])
        sib_ids[sid] = [j for j in L if j != sid and near_dup(A, nset(S1.name.loc[j]), need_core=False)]
    log(f"sibling ids {len(sib_ids):,} {time.time()-t0:.0f}s")
    out = {}; ex = {}
    for c in ("France", "US", "India"):
        a = accepted(f"S005_{c}").reset_index(drop=True)
        by_rec = a.groupby("rec").indices
        m = a.s1.isin(need).values
        ii = np.flatnonzero(m)
        f2 = np.zeros(len(a), bool); f2g = np.zeros(len(a), bool); js_arr = np.full(len(a), np.nan); jr_arr = np.full(len(a), np.nan)
        for i in ii:
            A = nset(S1.name.loc[a.s1[i]]); R = nset(recs.name.loc[a.rec[i]])
            jr = jac(R, A); js = max(jac(R, B) for B in SIB[a.s1[i]])
            js_arr[i] = js; jr_arr[i] = jr
            if js > 0 and js >= jr:
                f2[i] = True
                f2g[i] = js > jr
        kept = a.kept_final.values
        rows = []
        for i in np.flatnonzero(f2):
            cl = by_rec[a.rec[i]]
            sibs = set(sib_ids[a.s1[i]])
            sib_cl = [j for j in cl if a.s1[j] in sibs]
            sib_p = max((a.p[j] for j in sib_cl), default=np.nan)
            other_cl = [j for j in cl if j != i and a.s1[j] not in sibs]
            rows.append(dict(i=int(i), s1=a.s1[i], rec=a.rec[i], p=float(a.p[i]), kept=bool(kept[i]), n_claims=int(a.n_claims[i]), f2_gt=bool(f2g[i]),
                             jr=float(jr_arr[i]), js=float(js_arr[i]), sib_claims=len(sib_cl) > 0, sib_p=float(sib_p),
                             other_claims=len(other_cl), sib_ids=list(sibs)))
        R_ = pd.DataFrame(rows)
        r = dict(accepted=int(len(a)), F2_ge=int(f2.sum()), F2_gt=int(f2g.sum()), F2_ge_s1=int(a.s1[f2].nunique()),
                 F2_ge_multi=int((a.n_claims.values[f2] > 1).sum()), F2_ge_kept=int((f2 & kept).sum()), F2_ge_kept_s1=int(a.s1[f2 & kept].nunique()),
                 F2_gt_kept=int((f2g & kept).sum()))
        if len(R_):
            r["flagged_sibling_claims"] = int(R_.sib_claims.sum())
            r["flagged_dropped_by_mc"] = int((~R_.kept).sum())
            r["dropped_and_sibling_claims"] = int(((~R_.kept) & R_.sib_claims).sum())
            k = R_[R_.kept]
            r["kept_sibling_claims_lower_p"] = int(k.sib_claims.sum())
            r["kept_sibling_no_claim"] = int((~k.sib_claims).sum())
            r["kept_single_claim"] = int((k.n_claims == 1).sum())
            r["kept_js_eq_1"] = int((k.js == 1).sum())
            r["kept_gt"] = int(k.f2_gt.sum())
            r["kept_p_quantiles"] = [round(float(q), 4) for q in np.quantile(k.p, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(k) else None
            if len(k) and k.sib_claims.any():
                kk = k[k.sib_claims]
                r["kept_sibclaim_p_minus_sibp_quantiles"] = [round(float(q), 5) for q in np.quantile(kk.p - kk.sib_p, [0.1, 0.5, 0.9])]
            # does the sibling have ANY accepted record? (sibling-with-no-claims means this pair's record may be the sibling's only one)
            acc_s1 = set(a.s1.values)
            r["kept_sibling_has_any_accepted"] = int(sum(1 for L in k.sib_ids if any(s in acc_s1 for s in L)))
        out[c] = dict(summary=r)
        log(c, r)
        if c != "France" or not len(R_):
            continue
        # ---------- RL-27 test columns of (this S1, rec) and (sibling, rec) for the France kept flagged pairs
        k = R_[R_.kept].reset_index(drop=True)
        want = {}
        for q, row in k.iterrows():
            want[(row.s1, row.rec)] = ("self", q)
            for s in row.sib_ids:
                want[(s, row.rec)] = ("sib", q)
        # also dropped flagged pairs (for comparison)
        d = R_[~R_.kept].reset_index(drop=True)
        for q, row in d.iterrows():
            want.setdefault((row.s1, row.rec), ("dself", q))
        ws1 = set(s for s, _ in want); found = {}
        files = sorted(glob.glob(PATHS["test_chunks"].format(country=c)))
        for fi, f in enumerate(files):
            z = np.load(f)
            s1c = z["s1"]
            mm = np.isin(s1c, list(ws1))
            if not mm.any():
                continue
            cc = z["cand"]
            rl = None
            for j in np.flatnonzero(mm):
                kk_ = (s1c[j], cc[j])
                if kk_ in want:
                    if rl is None:
                        rl = np.load(f.replace(os.path.join("P3", c), os.path.join("RL", "test_feats", c)).replace("chunk_", "rl27_chunk_").replace(".npz", ".npy"))
                    found[kk_] = rl[j].tolist() + [float(z["LF"][j, 0])]
        log(f"chunks scanned {len(files)}, found {len(found)}/{len(want)} {time.time()-t0:.0f}s")
        def col(key, name):
            v = found.get(key)
            return None if v is None else v[C.index(name)]
        stats = collections.Counter(); ex_rows = []
        rv_self_kept, rv_self_drop = [], []
        for q, row in k.iterrows():
            me = found.get((row.s1, row.rec))
            sib_c = [s for s in row.sib_ids if (s, row.rec) in found]
            stats["kept_self_found"] += me is not None
            stats["kept_sibling_is_candidate"] += len(sib_c) > 0
            if me is not None:
                rv_self_kept.append(me[6])
                stats["kept_self_rv_rank1"] += me[6] == 1
                stats["kept_self_rv_rank_gt1"] += me[6] > 1
            if len(ex_rows) < 40:
                ex_rows.append(dict(s1_name=S1.name.loc[row.s1], s1_addr=S1.addr.loc[row.s1], rec_name=recs.name.loc[row.rec], rec_addr=recs.addr.loc[row.rec],
                                    p=round(row.p, 4), sib_names=[S1.name.loc[s] for s in row.sib_ids], sib_claims=bool(row.sib_claims),
                                    sib_p=None if np.isnan(row.sib_p) else round(row.sib_p, 4), sibling_is_candidate=len(sib_c) > 0,
                                    self_rv_rank=None if me is None else me[6], self_rv_gap=None if me is None else round(me[9], 4) if me[9] == me[9] else None,
                                    sib_rv_rank=[found[(s, row.rec)][6] for s in sib_c], jr=round(row.jr, 3), js=round(row.js, 3), f2_gt=bool(row.f2_gt)))
        for q, row in d.iterrows():
            me = found.get((row.s1, row.rec))
            if me is not None:
                rv_self_drop.append(me[6])
        stats = dict(stats)
        stats["kept_self_rv_rank_hist"] = dict(collections.Counter(rv_self_kept))
        stats["dropped_self_rv_rank_hist"] = dict(collections.Counter(rv_self_drop))
        stats["n_kept"] = int(len(k)); stats["n_dropped"] = int(len(d))
        out[c]["kept_rl27"] = stats
        log("France kept rl27", stats)
        ex[c] = ex_rows
    json.dump(out, open(os.path.join(HERE, "VE_2_test.json"), "w"), indent=1, default=str)
    json.dump(ex, open(os.path.join(HERE, "VE_2_test_examples.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
