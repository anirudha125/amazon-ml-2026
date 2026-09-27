"""RL-03 -- information-theoretic ceiling: how much of the loss is provably irreducible?

Symmetry argument: if a GT record r carries no address and its name is equally compatible with k S1 records
(same name core = same non-suffix token multiset, suffixes/honorifics being generator noise), no classifier can
give its true owner posterior > 1/k. For k >= 2 the F0.5-optimal decision is to reject -> r is an irreducible FN.

Computed on the FULL train GT (7.6M links) for rates, and on the 2,001 val S1 for the macro-F0.5 ceiling.
Read-only inputs. Output: printed tables + experiments/RL/rl03_ceiling.json
"""
import sys, os, re, json, pickle, unicodedata, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core, f05, toks, is_indic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    D = load("train", verbose=False)
    s1, gt = D["s1"], D["gt"]
    rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    print("computing cores ...", flush=True)
    s1["core"] = s1.name.map(core)
    rec["core"] = rec.name.map(core)
    rec["addr_empty"] = rec.addr.str.strip().eq("")
    rec["indic"] = rec.name.map(is_indic)
    cf = s1.groupby(["country", "core"]).size().rename("k").reset_index()
    s1 = s1.merge(cf, on=["country", "core"], how="left")
    g = gt.merge(s1[["id", "country", "core", "k"]].rename(columns={"id": "s1", "core": "s1_core"}), on="s1") \
          .merge(rec[["id", "core", "addr_empty", "indic"]].rename(columns={"id": "rec", "core": "rec_core"}), on="rec")
    n = len(g)
    print(f"GT links {n:,}")
    ae = g.addr_empty
    print(f"record address empty: {ae.mean()*100:.2f}% of links")
    for lo, hi in [(1, 1), (2, 2), (3, 9), (10, 99), (100, 10**9)]:
        m = ae & g.k.between(lo, hi)
        print(f"  addr-empty & S1-core-freq in [{lo},{hi}]: {m.mean()*100:.3f}% of links")
    # k=1 empty-address: what else carries the same record core? (distractors / other owners)
    owner_of = dict(zip(gt.rec, gt.s1))
    e = rec[rec.addr_empty].copy()
    e["owner"] = e.id.map(owner_of)
    grp = e.groupby("core").agg(n=("id", "size"), n_owned=("owner", lambda x: x.notna().sum()),
                                n_owners=("owner", lambda x: x.dropna().nunique()))
    print(f"\nempty-address records: {len(e):,}; linked {e.owner.notna().mean()*100:.1f}% ; unlinked {e.owner.isna().mean()*100:.1f}%")
    # Bayes posterior for an empty-address record given only its exact core, among S1 with that core (k) :
    # P(true owner) <= 1/k ; with unlinked same-core records the posterior of 'is a match of this S1' shrinks further.
    # ---------------- val ceiling ----------------
    d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
    val = sorted({m[0] for m in d["val_meta"]})
    G = g[g.s1.isin(set(val))]
    Gs = G.groupby("s1")
    irr = G[G.addr_empty & (G.k >= 2)]
    irr_set = set(zip(irr.s1, irr.rec))
    print(f"\nval GT links {len(G):,}; provably-irreducible (addr empty & k>=2): {len(irr):,} ({len(irr)/len(G)*100:.2f}%)")
    Gd = G.groupby("s1").rec.apply(set).to_dict()
    ceil = []
    for a in val:
        Ga = Gd.get(a, set())
        A = {r for r in Ga if (a, r) not in irr_set}
        ceil.append(f05(Ga, A))
    c1 = 100 * np.mean(ceil)
    print(f"CEILING-1 (perfect on everything except provably-irreducible empty-address records): {c1:.2f}")
    # alternative: 'accept them anyway' is worse in expectation (posterior <= 1/k); report k>=2 split
    res = dict(ceiling_addr_empty_k2=c1, val_links=len(G), irr_links=len(irr),
               train_addr_empty_rate=float(ae.mean()), train_addr_empty_k2_rate=float((ae & (g.k >= 2)).mean()))
    # --- what fraction of val loss (E018C) sits in this class? ---
    E = pd.read_pickle(os.path.join(OUT, "rl02_errors.pkl"))
    E["irr"] = [(a, c) in irr_set for a, c in zip(E.s1, E.cand)]
    print("E018C errors inside provably-irreducible class:", E.groupby("kind").irr.sum().to_dict(),
          "of", E.kind.value_counts().to_dict())
    json.dump(res, open(os.path.join(OUT, "rl03_ceiling.json"), "w"), indent=1)
    g.to_pickle(os.path.join(OUT, "cache", "gt_annot.pkl"))


if __name__ == "__main__":
    main()
