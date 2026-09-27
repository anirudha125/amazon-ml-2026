"""RL-21 -- error budget of the current best system (E024 D2b_union_rrUb_big, seed 42, th .72) on V1 (20k S1) and V0,
separating irreducible (symmetric) ambiguity from resolvable error. Read-only; CPU only.

Symmetry classes (a = true S1, r = record). If r's evidence is at least as compatible with another S1 b as with a,
no model -- pairwise or global -- can give a posterior > 1/2, so the F0.5-optimal decision is to reject:
  C1  r address empty AND >=1 other S1 in the country has a's name core                               (rigorous)
  C2  r name substituted (no shared core token, not Indic/url, not an acronym) AND a's exact address
      (token multiset) is shared by >=1 other S1 (co-located)                                            (rigorous)
  C3  r address has no house number AND >=1 other S1 with a's name core overlaps r's address words at least
      as much as a does                                                                                   (near-symmetric proxy)
Outputs: rl21_budget_d2b.json (+ cache/rl21_links_V1.pkl, cache/rl21_fp_V1.pkl)
"""
import os, sys, re, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core, fold, toks, SUFFIX, is_indic
from rl04_numeric_ambiguity import addr_feats, rel, STOP
from rl06_link_annotation import name_rel
from rl17_context_rules import f05_vec, boot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
E24 = os.path.join(ROOT, "experiments", "E024")
ARM, TH = "D2b_union_rrUb_big", 0.72


def akey(a):
    return " ".join(sorted(re.findall(r"[a-z0-9]+", fold(a))))


def words(a):
    return {w for w in re.findall(r"[a-z]{3,}", fold(a)) if w not in STOP}


def acronym_of(rn, sn):
    letters = re.sub(r"[^a-z]", "", fold(rn))
    ini = "".join(t[0] for t in toks(sn) if t and t[0].isalpha())
    ini2 = "".join(t[0] for t in toks(sn) if t and t[0].isalpha() and t not in SUFFIX)
    return len(letters) >= 2 and (letters in (ini, ini2) or (len(ini2) >= 2 and letters.startswith(ini2)))


def main():
    D = load("train", verbose=False)
    s1 = D["s1"].copy()
    C = pd.read_pickle(os.path.join(OUT, "cache", "ctx_train.pkl"))
    s1["core"] = C["s1"]["core"].values
    assert (C["s1"]["id"].values == s1.id.values).all()
    s1["akey"] = s1.addr.map(akey)
    kcore = s1.groupby(["country", "core"]).size()
    kaddr = s1.groupby(["country", "akey"]).size()
    fam = s1.groupby(["country", "core"]).id.apply(list).to_dict()
    S1 = s1.set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
    owner = dict(zip(D["gt"].rec, D["gt"].s1))
    Wcache = {}

    def W(i):
        if i not in Wcache:
            Wcache[i] = words(S1.at[i, "addr"])
        return Wcache[i]

    def sym_class(a, r):
        """symmetric class of candidate r for S1 a (None if none)."""
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

    res = {}
    for vs in ("V1", "V0"):
        m = np.load(os.path.join(E24, f"{vs}_a50n10d10a", "meta.npz"), allow_pickle=True)
        p = np.load(os.path.join(E24, f"p_{ARM}_{vs}_s42.npy"))
        s1_ids, s1idx, cand, y, ngt = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(int), m["n_gt"]
        n = len(s1_ids); acc = p >= TH
        base = f05_vec(np.bincount(s1idx, weights=acc & (y == 1), minlength=n), np.bincount(s1idx, weights=acc, minlength=n), ngt)
        pos = {(s1_ids[s1idx[i]], cand[i]): (float(p[i]), bool(acc[i])) for i in np.where(y == 1)[0]}
        vset = set(s1_ids)
        G = D["gt"][D["gt"].s1.isin(vset)]
        rows = []
        for a, r in zip(G.s1, G.rec):
            pr = pos.get((a, r))
            out = "miss" if pr is None else ("TP" if pr[1] else "FN")
            rows.append((a, r, out, pr[0] if pr else np.nan))
        L = pd.DataFrame(rows, columns=["s1", "rec", "out", "p"])
        ann = pd.read_pickle(os.path.join(OUT, "cache", "gt_annot_full.pkl"))
        L = L.merge(ann[["s1", "rec", "addr_empty", "k", "name_rel", "hn", "wov", "country"]], on=["s1", "rec"], how="left")
        L["sym"] = [sym_class(a, r) for a, r in zip(L.s1, L.rec)]
        def cat(x):
            if x.sym: return "SYM-" + x.sym
            if x.addr_empty: return "L2 empty-addr, unique name"
            if x.name_rel == "indic": return "L3 Indic name"
            if x.name_rel == "fake": return "L4 substituted name"
            if x.hn in ("d<=2", "1digit", "d<=20", "far") and x.wov >= 0.99: return "L5 house-number perturbed"
            if x.hn == "rec_nonum": return "L7 record without house number"
            return "L6 other"
        L["cat"] = L.apply(cat, axis=1)
        # FPs
        fi = np.where(acc & (y == 0))[0]
        F = []
        for i in fi:
            a, r = s1_ids[s1idx[i]], cand[i]; o = owner.get(r)
            ra = R.at[r, "addr"]
            if o is None:
                fcat = "F-unlinked, empty addr" if not ra.strip() else "F-unlinked decoy"
            else:
                sc = sym_class(o, r)       # is r symmetric between its owner o and some other S1 (incl. a)?
                same_core = S1.at[o, "core"] == S1.at[a, "core"] and S1.at[a, "country"] == S1.at[o, "country"]
                same_addr = S1.at[o, "akey"] == S1.at[a, "akey"]
                if sc == "C1" and same_core: fcat = "F-owned, SYM-C1 (same name, empty addr)"
                elif sc == "C2" and same_addr: fcat = "F-owned, SYM-C2 (co-located, substituted name)"
                elif sc == "C3" and same_core: fcat = "F-owned, SYM-C3 (same name, no number)"
                else: fcat = "F-owned, owner distinguishable"
            F.append((a, r, float(p[i]), fcat, int(ngt[s1idx[i]]) == 0))
        FP = pd.DataFrame(F, columns=["s1", "rec", "p", "cat", "singleton"])
        # counterfactuals
        idx_of = {s: j for j, s in enumerate(s1_ids)}
        tp0 = np.bincount(s1idx, weights=acc & (y == 1), minlength=n); na0 = np.bincount(s1idx, weights=acc, minlength=n)
        def score_with(add_tp=None, rm_fp=None, rm_tp=None):
            tp, na = tp0.copy(), na0.copy()
            for s in (add_tp or []): tp[idx_of[s]] += 1; na[idx_of[s]] += 1
            for s in (rm_fp or []): na[idx_of[s]] -= 1
            for s in (rm_tp or []): tp[idx_of[s]] -= 1; na[idx_of[s]] -= 1
            return f05_vec(tp, na, ngt)
        out = {"macro": round(base.mean() * 100, 3), "tp": int(tp0.sum()), "fp": int(len(FP)), "n_gt": int(ngt.sum()),
               "lost_links": int((L.out != "TP").sum())}
        bud = []
        for (c, o), g in L[L.out != "TP"].groupby(["cat", "out"]):
            d = (score_with(add_tp=list(g.s1)) - base).mean() * 100
            bud.append(dict(category=c, stage="retrieval" if o == "miss" else "scorer FN", links=len(g), pp_if_fixed=round(d, 3),
                            reducible=not c.startswith("SYM")))
        for c, g in FP.groupby("cat"):
            d = (score_with(rm_fp=list(g.s1)) - base).mean() * 100
            bud.append(dict(category=c, stage="scorer FP", links=len(g), pp_if_fixed=round(d, 3), reducible=True))
        out["budget"] = sorted(bud, key=lambda x: -x["pp_if_fixed"])
        # ceilings: symmetric links lost, everything else perfect, no FP
        n_sym = L.sym.notna().groupby(L.s1).sum().reindex(s1_ids).fillna(0).values
        ceil = f05_vec(ngt - n_sym, ngt - n_sym, ngt)
        out["ceiling_symmetric_all"] = round(ceil.mean() * 100, 3)
        for cl in ("C1", "C2", "C3"):
            ns = (L.sym == cl).groupby(L.s1).sum().reindex(s1_ids).fillna(0).values
            out[f"ceiling_{cl}_only"] = round(f05_vec(ngt - ns, ngt - ns, ngt).mean() * 100, 3)
        out["sym_links"] = {c: int((L.sym == c).sum()) for c in ("C1", "C2", "C3")}
        out["sym_links_accepted_TP"] = {c: int(((L.sym == c) & (L.out == "TP")).sum()) for c in ("C1", "C2", "C3")}
        # per-S1 top-k oracle on the model ranking (upper bound of any decision layer on this ranking)
        order = np.lexsort((-p, s1idx))
        best = np.zeros(n)
        sidx_sorted = s1idx[order]; y_sorted = y[order]
        starts = np.searchsorted(sidx_sorted, np.arange(n)); ends = np.searchsorted(sidx_sorted, np.arange(n), side="right")
        for j in range(n):
            yy = y_sorted[starts[j]:ends[j]][:30]; g = ngt[j]
            if g == 0:
                best[j] = 1.0; continue
            ctp = np.cumsum(yy); ks = np.arange(1, len(yy) + 1)
            f = np.where(ctp > 0, 1.25 * ctp / (0.25 * g + ks), 0.0)
            best[j] = max(f.max() if len(f) else 0.0, 0.0)
        out["oracle_topk_on_ranking"] = round(best.mean() * 100, 3)
        # S1-level anatomy
        pred_empty = na0 == 0
        out["s1_predicted_empty_but_nonsingleton"] = int((pred_empty & (ngt > 0)).sum())
        out["pp_lost_on_those"] = round(((pred_empty & (ngt > 0)).astype(float) * (1 - base)).sum() / n * 100, 3)
        out["singleton_fp_s1"] = int(((ngt == 0) & (na0 > 0)).sum())
        res[vs] = out
        print(vs, json.dumps({k: v for k, v in out.items() if k != "budget"}), flush=True)
        for b in out["budget"]:
            print("   ", b, flush=True)
        if vs == "V1":
            L.to_pickle(os.path.join(OUT, "cache", "rl21_links_V1.pkl")); FP.to_pickle(os.path.join(OUT, "cache", "rl21_fp_V1.pkl"))
    json.dump(res, open(os.path.join(OUT, "rl21_budget_d2b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
