"""RL-31 Phase 1 -- residual-error taxonomy of the strongest labelled V1 pipeline (RRL = S006's model, th .72), with
S005's model (NEW, th .78) alongside. READ-ONLY on every input; writes only experiments/RL/rl31/p1_*.

Mutually exclusive PRIMARY categories (a link / FP gets exactly one):
  lost links (GT links of V1 S1 that are not accepted):
    stage   = miss (not in the union pool) | FN (in pool, p < th)
    class   = SYM-C1s (empty addr, owner FULL name duplicated: strictly symmetric, RL-25)
            | SYM-C1l (empty addr, only the name core shared: resolvable via suffix, RL-22b)
            | SYM-C2 (substituted name at a shared address) | SYM-C3 (no house number, near-symmetric)
            | L2 empty addr + unique name | L3 Indic | L4 substituted name | L5 house number perturbed
            | L7 record without number | L6 other                      (definitions: rl21_budget_d2b.py)
  false positives:
    F-owned-*  (record belongs to another S1: owner distinguishable / SYM-C1 / SYM-C2 / SYM-C3)
    F-unlinked decoy (record has no owner, address present) | F-unlinked empty addr
"pp if fixed" = macro-F0.5 gain on V1 if exactly those errors were corrected and everything else kept (counterfactual).
Outputs: p1_links_V1.pkl, p1_fp_V1.pkl, p1_taxonomy.json
"""
import os, sys, re, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *
from rl_data import load
from rl02_error_decomp import toks, fold
from rl04_numeric_ambiguity import addr_feats
from rl06_link_annotation import name_rel
from rl21_budget_d2b import akey, words, acronym_of

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
V = load_v1(); n = len(V["s1_ids"]); s1idx = V["s1idx"]; y = V["y"]
ids_row = V["s1_ids"][s1idx]
row_of = {(a, c): i for i, (a, c) in enumerate(zip(ids_row, V["cand"]))}
V["sel"] = topk_mask(s1idx, V["pb"], 10)
V["rk_RRL"] = rank_in_s1(s1idx, V["p_RRL"]); V["rk_pb"] = rank_in_s1(s1idx, V["pb"])
log("V1 loaded", len(y))

D = load("train", verbose=False)
s1 = D["s1"].copy()
C = pd.read_pickle(os.path.join(RL, "cache", "ctx_train.pkl")); s1["core"] = C["s1"]["core"].values
assert (C["s1"]["id"].values == s1.id.values).all(); del C
s1["akey"] = s1.addr.map(akey); s1["full"] = s1.name.map(lambda s: " ".join(toks(s)))
kcore = s1.groupby(["country", "core"]).size(); kaddr = s1.groupby(["country", "akey"]).size()
kfull = s1.groupby(["country", "full"]).size()
fam = s1.groupby(["country", "core"]).id.apply(list).to_dict()
S1 = s1.set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
owner = dict(zip(D["gt"].rec, D["gt"].s1))
log("train loaded")
Wc = {}
def W(i):
    if i not in Wc: Wc[i] = words(S1.at[i, "addr"])
    return Wc[i]

def sym_class(a, r):
    c = S1.at[a, "country"]; ca = S1.at[a, "core"]; ra, rn = R.at[r, "addr"], R.at[r, "name"]
    k = kcore.get((c, ca), 1) if ca else 1
    if not ra.strip():
        return "C1" if k >= 2 else None
    nr = name_rel(rn, S1.at[a, "name"])
    if nr == "fake" and not acronym_of(rn, S1.at[a, "name"]) and kaddr.get((c, S1.at[a, "akey"]), 1) >= 2:
        return "C2"
    if k >= 2 and addr_feats(ra)[0] is None:
        wr = words(ra); ov_a = len(wr & W(a))
        fl = fam.get((c, ca), [])
        if len(fl) <= 3000:
            for b in fl:
                if b != a and len(wr & W(b)) >= ov_a:
                    return "C3"
    return None

# ------------------------------------------------------------------ links
L = pd.read_pickle(os.path.join(RL, "cache", "rl21_links_V1.pkl"))[["s1", "rec", "addr_empty", "k", "name_rel", "hn", "wov", "country", "sym", "cat"]]
L["dup_full"] = [kfull.get((S1.at[a, "country"], S1.at[a, "full"]), 1) >= 2 for a in L.s1]
L["pcat"] = L.cat
m = L.sym == "C1"; L.loc[m & L.dup_full, "pcat"] = "SYM-C1s"; L.loc[m & ~L.dup_full, "pcat"] = "SYM-C1l"
L["row"] = [row_of.get((a, r), -1) for a, r in zip(L.s1, L.rec)]
ok = L.row.values >= 0; rw = L.row.values.clip(0)
for nm in MODELS:
    p = V["p_" + nm][rw]; L["p_" + nm] = np.where(ok, p, np.nan)
    L["out_" + nm] = np.where(~ok, "miss", np.where(p >= V["th_" + nm], "TP", "FN"))
for k in ("rk_RRL", "rk_pb", "sel", "is_s2", "rank_addr", "rank_name", "rank_dense"):
    L[k] = np.where(ok, V[k][rw], -1)
for j, cn in enumerate(RL27_COLS):
    L[cn] = np.where(ok, V["rl27"][rw, j], np.nan)
L["sidx"] = pd.Series(range(n), index=V["s1_ids"]).reindex(L.s1).values
L["is_s2_rec"] = L.rec.str.startswith("S2")
log("links", len(L), L.out_RRL.value_counts().to_dict())

# ------------------------------------------------------------------ FPs of each model
fp_tabs = {}
for nm in ("RRL", "NEW"):
    acc = V["p_" + nm] >= V["th_" + nm]; fi = np.flatnonzero(acc & (y == 0)); rows = []
    for i in fi:
        a, r = ids_row[i], V["cand"][i]; o = owner.get(r); ra = R.at[r, "addr"]
        if o is None:
            fc = "F-unlinked empty addr" if not ra.strip() else "F-unlinked decoy"
        else:
            sc = sym_class(o, r)
            same_core = S1.at[o, "core"] == S1.at[a, "core"]; same_addr = S1.at[o, "akey"] == S1.at[a, "akey"]
            if sc == "C1" and same_core: fc = "F-owned SYM-C1 (same name, empty addr)"
            elif sc == "C2" and same_addr: fc = "F-owned SYM-C2 (co-located, substituted name)"
            elif sc == "C3" and same_core: fc = "F-owned SYM-C3 (same name, no number)"
            else: fc = "F-owned, owner distinguishable"
        rows.append((a, r, i, o, fc, int(V["n_gt"][s1idx[i]]) == 0, o in set(V["s1_ids"]) if o else False))
    F = pd.DataFrame(rows, columns=["s1", "rec", "row", "owner", "cat", "singleton", "owner_in_V1"])
    for k2 in MODELS:
        F["p_" + k2] = V["p_" + k2][F.row.values]
    for k in ("rk_RRL", "rk_pb", "sel", "is_s2", "rank_addr", "rank_name", "rank_dense"):
        F[k] = V[k][F.row.values]
    for j, cn in enumerate(RL27_COLS):
        F[cn] = V["rl27"][F.row.values, j]
    F["sidx"] = s1idx[F.row.values]; F["country"] = V["country"][F.sidx.values]
    fp_tabs[nm] = F
    log("FP", nm, len(F), F.cat.value_counts().to_dict())

# ------------------------------------------------------------------ budgets
def budget(nm):
    acc = V["p_" + nm] >= V["th_" + nm]; tp0, na0 = per_s1(V, acc); ng = V["n_gt"]; base = f05_vec(tp0, na0, ng)
    out = dict(macro=round(base.mean() * 100, 3), loss_pp=round((1 - base.mean()) * 100, 3), tp=int(tp0.sum()),
               fp=int(na0.sum() - tp0.sum()), fn=int(ng.sum() - tp0.sum()))
    def gain(add=None, rm_fp=None):
        tp, na = tp0.copy(), na0.copy()
        if add is not None: np.add.at(tp, add, 1); np.add.at(na, add, 1)
        if rm_fp is not None: np.add.at(na, rm_fp, -1)
        return f05_vec(tp, na, ng) - base
    Lx = L[L["out_" + nm] != "TP"]; F = fp_tabs[nm]; rows = []
    for (c, st), g in Lx.groupby(["pcat", "out_" + nm]):
        d = gain(add=g.sidx.values)
        rows.append(dict(category=c, stage="retrieval miss" if st == "miss" else "scorer FN", links=len(g), s1=int(g.sidx.nunique()),
                         pp_if_fixed=round(d.mean() * 100, 4), reducible=c not in ("SYM-C1s", "SYM-C2", "SYM-C3")))
    for c, g in F.groupby("cat"):
        d = gain(rm_fp=g.sidx.values)
        rows.append(dict(category=c, stage="scorer FP", links=len(g), s1=int(g.sidx.nunique()), pp_if_fixed=round(d.mean() * 100, 4),
                         reducible=True, singleton_s1=int(g[g.singleton].sidx.nunique())))
    out["budget"] = sorted(rows, key=lambda r: -r["pp_if_fixed"])
    # stage totals and the fully-fixed counterfactuals
    irr = Lx.pcat.isin(["SYM-C1s", "SYM-C2", "SYM-C3"])
    out["fix_all_FP"] = round(gain(rm_fp=F.sidx.values).mean() * 100, 4)
    out["fix_all_FN_inpool"] = round(gain(add=Lx[Lx["out_" + nm] == "FN"].sidx.values).mean() * 100, 4)
    out["fix_all_FN_inpool_reducible"] = round(gain(add=Lx[(Lx["out_" + nm] == "FN") & ~irr].sidx.values).mean() * 100, 4)
    out["fix_all_miss"] = round(gain(add=Lx[Lx["out_" + nm] == "miss"].sidx.values).mean() * 100, 4)
    out["fix_all_miss_reducible"] = round(gain(add=Lx[(Lx["out_" + nm] == "miss") & ~irr].sidx.values).mean() * 100, 4)
    tp, na = tp0.copy(), na0.copy(); red = Lx[~irr].sidx.values; np.add.at(tp, red, 1); np.add.at(na, red, 1); np.add.at(na, F.sidx.values, -1)
    out["fix_all_reducible"] = round((f05_vec(tp, na, ng) - base).mean() * 100, 4)
    out["ceiling_after_fix_all_reducible"] = round(f05_vec(tp, na, ng).mean() * 100, 3)
    # S1-level loss concentration
    loss = 1 - base; cnt_err = np.bincount(Lx.sidx.values, minlength=n) + np.bincount(F.sidx.values, minlength=n)
    has_fp = np.bincount(F.sidx.values, minlength=n) > 0; has_fn = np.bincount(Lx.sidx.values, minlength=n) > 0
    irr_s1 = np.bincount(Lx[irr].sidx.values, minlength=n) > 0; red_s1 = np.bincount(Lx[~irr].sidx.values, minlength=n) > 0
    out["s1_with_error"] = int((loss > 0).sum())
    out["s1_fp_only"] = int((has_fp & ~has_fn).sum()); out["s1_fn_only"] = int((has_fn & ~has_fp).sum()); out["s1_fp_and_fn"] = int((has_fp & has_fn).sum())
    out["loss_pp_fp_only_s1"] = round(loss[has_fp & ~has_fn].sum() / n * 100, 4)
    out["loss_pp_fn_only_s1"] = round(loss[has_fn & ~has_fp].sum() / n * 100, 4)
    out["loss_pp_fp_and_fn_s1"] = round(loss[has_fp & has_fn].sum() / n * 100, 4)
    out["s1_zero_score"] = int((base == 0).sum()); out["loss_pp_zero_score_s1"] = round((base == 0).sum() / n * 100, 4)
    out["s1_pred_empty_nonsingleton"] = int(((na0 == 0) & (ng > 0)).sum())
    out["singleton_fp_s1"] = int(((ng == 0) & (na0 > 0)).sum())
    out["s1_irreducible_only"] = int((irr_s1 & ~red_s1 & ~has_fp).sum())
    out["loss_pp_irreducible_only_s1"] = round(loss[irr_s1 & ~red_s1 & ~has_fp].sum() / n * 100, 4)
    # symmetric ceiling on this S1 set (RL-25 strict definition)
    nsym = np.bincount(L[L.pcat.isin(["SYM-C1s", "SYM-C2", "SYM-C3"])].sidx.values, minlength=n)
    out["ceiling_symmetric_strict"] = round(f05_vec(ng - nsym, ng - nsym, ng).mean() * 100, 3)
    out["sym_links_accepted_as_TP"] = {c: int(((L.pcat == c) & (L["out_" + nm] == "TP")).sum()) for c in ("SYM-C1s", "SYM-C1l", "SYM-C2", "SYM-C3")}
    # per-country
    c = V["country"]
    out["by_country"] = {cc: dict(macro=round(base[c == cc].mean() * 100, 3), loss_pp=round((1 - base[c == cc].mean()) * 100, 3),
                                  ceiling=round(f05_vec(ng - nsym, ng - nsym, ng)[c == cc].mean() * 100, 3)) for cc in ("US", "India")}
    # oracle decision layers on this model's ranking
    order = np.lexsort((-V["p_" + nm], s1idx)); ys = y[order]; ss = s1idx[order]
    st = np.searchsorted(ss, np.arange(n)); en = np.searchsorted(ss, np.arange(n), side="right"); best = np.zeros(n)
    for j in range(n):
        g_ = ng[j]
        if g_ == 0: best[j] = 1.0; continue
        yy = ys[st[j]:en[j]][:50]; ctp = np.cumsum(yy); ks = np.arange(1, len(yy) + 1)
        best[j] = max((np.where(ctp > 0, 1.25 * ctp / (0.25 * g_ + ks), 0.0)).max() if len(yy) else 0.0, 0.0)
    out["oracle_per_s1_topk_on_ranking"] = round(best.mean() * 100, 3)
    return out, base

res = {}
for nm in ("RRL", "NEW"):
    r, base = budget(nm); res[nm] = r
    log(nm, json.dumps({k: v for k, v in r.items() if k != "budget"}))
    for b in r["budget"]:
        log("   ", b)
L.to_pickle(os.path.join(HERE, "p1_links_V1.pkl"))
pd.concat([f.assign(model=k) for k, f in fp_tabs.items()]).to_pickle(os.path.join(HERE, "p1_fp_V1.pkl"))
json.dump(res, open(os.path.join(HERE, "p1_taxonomy.json"), "w"), indent=1, default=float)
log("done")
