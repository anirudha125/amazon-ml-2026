"""RL-26 -- information value of the RECORD-CENTRIC competitor count (the normalization term of a bipartite posterior),
measured from train labels only; no model is trained.

Name-side count (empty-address records): n_compat(r) = #S1 in the country whose name-core token set CONTAINS the
record's core tokens (record noise = drops/adds suffixes, drops words -> containment is the right compatibility test).
Address-side count (substituted-name records): n_acompat(r) = #S1 in the country whose address word set contains the
record's address words AND whose house number equals the record's (when the record has one).
Measured on ALL train records (labels = GT owner):
  P(owner == the unique compatible S1 | n_compat == 1), P(linked at all | n == 0/1/2+), by class.
Then on the current best model's V1 residuals (D2b): how many L2 / L4 FNs and owned FPs have n == 1 vs >= 2.
Read-only; CPU only. Output: rl26_record_centric.json
"""
import os, sys, re, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import toks, SUFFIX, is_indic, fold
from rl04_numeric_ambiguity import addr_feats, STOP
from rl06_link_annotation import name_rel

OUT = os.path.dirname(os.path.abspath(__file__))


def ctoks(s):
    return frozenset(t for t in toks(s) if t not in SUFFIX)


def awords(a):
    return frozenset(w for w in re.findall(r"[a-z]{3,}", fold(a)) if w not in STOP)


class Inv:
    """inverted index token -> set(S1 row ids); rarest-first set intersection."""
    def __init__(self, ids, sets):
        post = collections.defaultdict(set)
        for i, s in zip(ids, sets):
            for t in s:
                post[t].add(int(i))
        self.post = dict(post)

    def compat(self, T):
        if not T:
            return None
        lists = sorted((self.post.get(t, set()) for t in T), key=len)
        cur = set(lists[0])
        for l in lists[1:]:
            if not cur:
                break
            cur &= l
        return np.fromiter(cur, dtype=np.int64, count=len(cur))


def main():
    D = load("train", verbose=False)
    s1 = D["s1"].reset_index(drop=True)
    owner = dict(zip(D["gt"].rec, D["gt"].s1))
    sid = s1.id.values
    num = s1.addr.map(lambda a: addr_feats(a)[0]).values
    res = {}
    rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    rec["owner"] = rec.id.map(owner).astype(object).where(rec.id.map(owner).notna(), None)
    for c in ("US", "India"):
        cm = (s1.country == c).values
        idx_c = np.where(cm)[0]
        inv_n = Inv(idx_c, [ctoks(x) for x in s1.name.values[idx_c]])
        inv_a = Inv(idx_c, [awords(x) for x in s1.addr.values[idx_c]])
        rc = rec[rec.country == c]
        # ---- name side: empty-address records with Latin, non-url names
        e = rc[rc.addr.str.strip() == ""]
        rows = []
        for i, nm, o in zip(e.id, e.name, e.owner):
            if is_indic(nm) or re.search(r"\.(com|in|net|org)|^[#@]", nm.lower()):
                continue
            comp = inv_n.compat(ctoks(nm))
            if comp is None:
                continue
            n = len(comp)
            hit = (o is not None) and n >= 1 and (o in set(sid[comp]) if n <= 50 else None)
            rows.append((n, o is not None, bool(n == 1 and o is not None and sid[comp[0]] == o), hit))
        X = pd.DataFrame(rows, columns=["n", "linked", "owner_is_unique", "owner_in_set"])
        tab = {}
        for lab, mk in (("n=0", X.n == 0), ("n=1", X.n == 1), ("n=2", X.n == 2), ("n=3-9", X.n.between(3, 9)), ("n>=10", X.n >= 10)):
            g = X[mk]
            tab[lab] = dict(records=len(g), share=round(len(g) / len(X), 3), P_linked=round(g.linked.mean(), 3) if len(g) else None,
                            P_owner_is_the_unique_compatible_S1=round(g.owner_is_unique.mean(), 3) if lab == "n=1" and len(g) else None,
                            P_owner_in_compatible_set=round(g.owner_in_set.dropna().mean(), 3) if len(g.owner_in_set.dropna()) else None)
        res[f"{c}_empty_address_name_side"] = tab
        # ---- address side: ALL addressed records (linked or not) whose address fits exactly one S1 a (words contained,
        #      house number equal when present) and whose name is substituted relative to a -> posterior P(owner == a)
        sub = rc[rc.addr.str.strip() != ""].sample(min(150000, int((rc.addr.str.strip() != "").sum())), random_state=0)
        s1_name = s1.name.values
        rows = []
        for nm, ad, o in zip(sub.name, sub.addr, sub.owner):
            W = awords(ad); hn = addr_feats(ad)[0]
            comp = inv_a.compat(W)
            if comp is None or len(comp) == 0:
                continue
            if hn:
                comp = comp[num[comp] == hn]
            n = len(comp)
            if n == 0:
                continue
            fake_vs = [name_rel(nm, s1_name[j]) == "fake" for j in comp[:20]]
            if not all(fake_vs):
                continue          # name is informative for at least one compatible S1 -> not the substituted-name regime
            rows.append((n, o is not None, bool(o is not None and o in set(sid[comp]))))
        Y = pd.DataFrame(rows, columns=["n", "linked", "owner_in_set"])
        tab = {}
        for lab, mk in (("n=1", Y.n == 1), ("n=2", Y.n == 2), ("n>=3", Y.n >= 3)):
            g = Y[mk]
            tab[lab] = dict(records=len(g), P_linked_to_a_compatible_S1=round(g.owner_in_set.mean(), 3) if len(g) else None,
                            P_unlinked=round(1 - g.linked.mean(), 3) if len(g) else None)
        res[f"{c}_substituted_name_address_side"] = tab
        print(c, json.dumps({k: v for k, v in res.items() if k.startswith(c)}, indent=1), flush=True)
    json.dump(res, open(os.path.join(OUT, "rl26_record_centric.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
