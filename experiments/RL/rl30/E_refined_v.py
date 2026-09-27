"""RL-30 / E (Part 6, round 2) -- typo-robust F1/F3 + F2 on V1 and V0 (both RL-27 NEW holdouts, th 0.78), pooled V0+V1,
plus a label-only F2 audit on the training pools (T0 / E014 / T2X: no NEW probabilities there, so only P(match | flag) and
who owns the record). Flags are label-free and built from the train corpus exactly as on test.
Output: rl30/E_refined_v_results.json, rl30/E_refined_v_examples.json"""
import os, sys, time, json, pickle, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import compute_flags, derive, s1_feats, rec_feats, role_lookup
from E_flags2 import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
E24 = os.path.join(ROOT, "experiments", "E024")
KEEP_BASE = ["F1_dist_s0.5", "F1_dist_s0.5_sameaddr", "F1_filler_all", "F1_filler_all_sameaddr", "F2_sib_ge", "F2_sib_gt", "F2_has_sib",
             "F3_street_mismatch", "F3_street_mismatch_numeq"]


def main():
    t0 = time.time()
    D = load("train", verbose=False)
    s1all = D["s1"]
    NV, SV = vocabs(s1all)
    log(f"vocabs {time.time()-t0:.0f}s")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_train.pkl"), "rb")); S_role, DC = role_lookup(RO["roles"])
    S1 = s1all.set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    owner = dict(zip(D["gt"].rec.values, D["gt"].s1.values))
    # sibling ids (same definition as E_roles: same country + akey, near_dup(need_core=False))
    need = set(RO["sib"].keys())
    ak_all = pd.Series([akey(a) for a in s1all.addr.values], index=s1all.id.values)
    grp = collections.defaultdict(list)
    for sid, k, c in zip(s1all.id.values, ak_all.values, s1all.country.values):
        if k:
            grp[c + "|" + k].append(sid)
    nt = {}
    sib_ids = {}
    for sid in need:
        c = S1.country.loc[sid]; L = grp[c + "|" + ak_all.loc[sid]]
        A = nset(S1.name.loc[sid])
        sib_ids[sid] = [j for j in L if j != sid and near_dup(A, nset(S1.name.loc[j]), need_core=False)]
    log(f"sibling ids for {len(sib_ids):,} S1 {time.time()-t0:.0f}s")

    sets = {}
    for vn, pp in (("V1", PATHS["v1_p_new"]), ("V0", os.path.join(RL, "cache", "rl27_p_NEW_V0_s42.npy"))):
        M = np.load(os.path.join(E24, f"{vn}_a50n10d10a", "meta.npz"))
        s1_ids = M["s1_ids"]; s1idx = M["s1idx"]; cand = M["cand"]; ps1 = s1_ids[s1idx]; pc = M["country"][s1idx]
        SS = S1.loc[s1_ids]; uc = np.unique(cand); RR = recs.loc[uc]
        S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(s1_ids, SS.name.values, SS.addr.values)}
        RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(uc, RR.name.values, RR.addr.values)}
        fl = compute_flags(ps1, cand, pc, S1F, RF, S_role, DC, RO["sib"]); F = derive(fl)
        FR = compute_refined(ps1, cand, pc, S1F, RF, S_role, NV, SV)
        flags = {k: F[k] for k in KEEP_BASE}; flags.update(FR)
        sets[vn] = dict(s1_ids=s1_ids, s1idx=s1idx, cand=cand, y=M["y"] == 1, n_gt=M["n_gt"], cs=M["country"], p=np.load(pp), flags=flags,
                        ps1=ps1, SS=SS, RR=RR)
        log(f"{vn}: flags {time.time()-t0:.0f}s")
    # pooled V0+V1
    V1, V0 = sets["V1"], sets["V0"]; off = len(V1["s1_ids"])
    P = dict(s1idx=np.concatenate([V1["s1idx"], V0["s1idx"] + off]), cand=np.concatenate([V1["cand"], V0["cand"]]),
             y=np.concatenate([V1["y"], V0["y"]]), n_gt=np.concatenate([V1["n_gt"], V0["n_gt"]]), cs=np.concatenate([V1["cs"], V0["cs"]]),
             p=np.concatenate([V1["p"], V0["p"]]), flags={k: np.concatenate([V1["flags"][k], V0["flags"][k]]) for k in V1["flags"]},
             ps1=np.concatenate([V1["ps1"], V0["ps1"]]))
    sets["V0+V1"] = P
    out = {}; ex = {}; rng = np.random.default_rng(1)
    for vn, S in sets.items():
        acc = S["p"] >= NEW_TH
        f0 = f05_per_s1(S["s1idx"], S["y"], acc, S["n_gt"]); f0_mc = f05_per_s1(S["s1idx"], S["y"], maxclaim(acc, S["p"], S["cand"]), S["n_gt"])
        res = dict(baseline=dict(n_s1=int(len(S["n_gt"])), macro=round(f0.mean() * 100, 4), macro_mc=round(f0_mc.mean() * 100, 4),
                                 accepted=int(acc.sum()), FP=int((acc & ~S["y"]).sum()), TP=int((acc & S["y"]).sum())))
        for k, f in S["flags"].items():
            res[k] = score_flag(f, S["s1idx"], S["y"], S["n_gt"], S["cs"], S["p"], S["cand"], NEW_TH, f0, f0_mc)
            if k.startswith("F2"):
                ii = np.flatnonzero(f)
                own = [owner.get(S["cand"][i]) for i in ii]
                sib_own = sum(1 for i, o in zip(ii, own) if o is not None and o in sib_ids.get(S["ps1"][i], []))
                acc_ii = [i for i in ii if acc[i]]
                res[k]["record_owner"] = dict(this_s1=int(S["y"][ii].sum()), a_sibling=int(sib_own),
                                              other_or_none=int(len(ii) - S["y"][ii].sum() - sib_own),
                                              accepted_FP_owned_by_sibling=int(sum(1 for i in acc_ii if not S["y"][i] and owner.get(S["cand"][i]) in sib_ids.get(S["ps1"][i], []))),
                                              accepted_FP=int(sum(1 for i in acc_ii if not S["y"][i])))
            if vn != "V0+V1":
                E = {}
                for lab, m in (("acc_FP", acc & f & ~S["y"]), ("acc_TP", acc & f & S["y"]), ("rej_pos_p03", f & (S["p"] >= 0.3) & ~acc & S["y"])):
                    jj = np.flatnonzero(m)
                    if len(jj):
                        jj = rng.choice(jj, size=min(5, len(jj)), replace=False)
                        E[lab] = [dict(s1=S["ps1"][i], s1_name=S["SS"].name.loc[S["ps1"][i]], s1_addr=S["SS"].addr.loc[S["ps1"][i]], rec=S["cand"][i],
                                       rec_name=S["RR"].name.loc[S["cand"][i]], rec_addr=S["RR"].addr.loc[S["cand"][i]], p=round(float(S["p"][i]), 4),
                                       owner=owner.get(S["cand"][i]), owner_name=S1.name.loc[owner[S["cand"][i]]] if owner.get(S["cand"][i]) in S1.index else None)
                                  for i in jj]
                ex[f"{vn}:{k}"] = E
            r = res[k]
            log(vn, k, {x: r[x] for x in ("pairs", "s1", "accepted", "acc_FP", "acc_precision_in_flag")}, "veto", r.get("veto", {}).get("delta_pp"),
                "oracle", r.get("veto_oracle", {}).get("delta_pp"), "rescue", r.get("rescue_p03", {}).get("delta_pp"),
                "z", (r["logit_incremental_p>=0.01"] or {}).get("z"))
        out[vn] = res
    # ---- label-only F2 audit on the training pools (no NEW p): P(match | F2) and record ownership
    aud = {}
    for tn in ("T0", "E014", "T2X"):
        M = np.load(os.path.join(E24, f"{tn}_a50n10d10a", "meta.npz"))
        s1_ids = M["s1_ids"]; ps1 = s1_ids[M["s1idx"]]
        m = np.isin(ps1, list(RO["sib"].keys()))
        ii = np.flatnonzero(m)
        if not len(ii):
            aud[tn] = dict(pairs=0); continue
        cand = M["cand"][ii]; y = M["y"][ii] == 1; p1 = ps1[ii]; pc = M["country"][M["s1idx"][ii]]
        us = np.unique(p1); uc = np.unique(cand); SS = S1.loc[us]; RR = recs.loc[uc]
        S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(us, SS.name.values, SS.addr.values)}
        RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(uc, RR.name.values, RR.addr.values)}
        fl = compute_flags(p1, cand, pc, S1F, RF, S_role, DC, RO["sib"])
        a = {}
        for k in ("f2_ge", "f2_gt", "has_sib"):
            f = fl[k]; jj = np.flatnonzero(f)
            sib_own = sum(1 for j in jj if owner.get(cand[j]) in sib_ids.get(p1[j], []))
            a[k] = dict(pairs=int(f.sum()), s1=int(np.unique(p1[f]).size), P_match=round(float(y[f].mean()), 4) if f.any() else None,
                        owner_this_s1=int(y[f].sum()), owner_sibling=int(sib_own), owner_other_or_none=int(f.sum() - y[f].sum() - sib_own))
        aud[tn] = a
        log(tn, a)
    out["F2_training_pool_audit_label_only"] = aud
    json.dump(out, open(os.path.join(HERE, "E_refined_v_results.json"), "w"), indent=1, default=str)
    json.dump(ex, open(os.path.join(HERE, "E_refined_v_examples.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
