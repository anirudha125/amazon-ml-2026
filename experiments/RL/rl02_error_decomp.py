"""RL-02 -- error decomposition of E018C (val 2,001 S1, th 0.72) against the FULL train ground truth.

Read-only inputs: experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl (val_meta, p_va, th), dataset via rl_data.
Outputs: experiments/RL/rl02_errors.pkl (per-error table), printed summary.
"""
import sys, os, re, pickle, unicodedata, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))


def f05(G, A):
    if not G:
        return 1.0 if not A else 0.0
    tp = len(G & A)
    if tp == 0:
        return 0.0
    p, r = tp / len(A), tp / len(G)
    return 1.25 * p * r / (0.25 * p + r)


SUFFIX = set("llc inc co corp corporation company ltd limited private pvt pc plc lp llp group the center "
             "services service enterprises smt sri shri m s dba doing business as and".split())


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", fold(s))


def core(s):
    return " ".join(sorted(t for t in toks(s) if t not in SUFFIX))


def is_indic(s):
    return any(0x0900 <= ord(ch) <= 0x0DFF for ch in s)


def tags(name, addr):
    n = name.strip()
    return dict(
        addr_empty=(addr.strip() == ""),
        domain=bool(re.search(r"\.(com|in|net|org|co)\b", n.lower())) and " " not in n.strip(),
        hashtag=n.startswith("#"),
        indic=is_indic(n),
        dba=bool(re.search(r"\b(dba|doing business as|d/b/a|aka|t/a|trading as)\b", n.lower())),
        null_tok=("<NULL>" in addr) or ("<NULL>" in n),
    )


def tokset_sim(a, b):
    A, B = set(toks(a)), set(toks(b))
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def main():
    d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
    meta, p, th = d["val_meta"], d["p_va"], d["th"]
    D = load("train", verbose=False)
    s1 = D["s1"].set_index("id"); gt = D["gt"]
    rec = pd.concat([D["s2"], D["s3"]]).set_index("id")
    owner = dict(zip(gt.rec, gt.s1))
    G = gt.groupby("s1").rec.apply(set).to_dict()
    val = sorted({m[0] for m in meta})
    print("val S1 in meta:", len(val))
    pool = collections.defaultdict(dict)
    for (a, c, y), pp in zip(meta, p):
        pool[a][c] = float(pp)
    # name frequency (core) per country over all train S1
    s1c = D["s1"].copy(); s1c["core"] = s1c.name.map(core)
    core_freq = s1c.groupby(["country", "core"]).size().to_dict()

    rows = []
    sc_now, sc_noFP, sc_noFN, sc_noRet, sc_orc, sc_all = [], [], [], [], [], []
    for a in val:
        Ga = G.get(a, set()); Pa = set(pool[a]); Aa = {c for c, q in pool[a].items() if q >= th}
        ctry = s1.at[a, "country"]
        miss = Ga - Pa; fn = (Ga & Pa) - Aa; fp = Aa - Ga
        sc_now.append(f05(Ga, Aa))
        sc_noFP.append(f05(Ga, Aa - fp))
        sc_noFN.append(f05(Ga, Aa | fn))
        sc_noRet.append(f05(Ga, Aa | miss))
        sc_orc.append(f05(Ga, Ga & Pa))
        for kind, S in (("miss", miss), ("fn", fn), ("fp", fp)):
            for c in S:
                r = rec.loc[c]
                t = tags(r["name"], r["addr"])
                o = owner.get(c)
                rows.append(dict(s1=a, cand=c, kind=kind, country=ctry, src=c[:2], p=pool[a].get(c, np.nan),
                                 n_gt=len(Ga), s1_name=s1.at[a, "name"], s1_addr=s1.at[a, "addr"],
                                 name=r["name"], addr=r["addr"], owner=o,
                                 owner_name=(s1.at[o, "name"] if o else None), owner_addr=(s1.at[o, "addr"] if o else None),
                                 name_sim=tokset_sim(r["name"], s1.at[a, "name"]),
                                 core_eq=(core(r["name"]) == core(s1.at[a, "name"])),
                                 s1_core_freq=core_freq.get((ctry, core(s1.at[a, "name"])), 0), **t))
    E = pd.DataFrame(rows)
    E.to_pickle(os.path.join(OUT, "rl02_errors.pkl"))
    m = lambda v: 100 * np.mean(v)
    print(f"\nmacro F0.5 now {m(sc_now):.2f} | remove all FP {m(sc_noFP):.2f} | add all scorer-FN {m(sc_noFN):.2f} | "
          f"add all retrieval misses {m(sc_noRet):.2f} | pool oracle {m(sc_orc):.2f}")
    print("error counts:", E.kind.value_counts().to_dict())
    for k in ("miss", "fn", "fp"):
        e = E[E.kind == k]
        print(f"\n== {k}: n={len(e)}  by country {e.country.value_counts().to_dict()}  by src {e.src.value_counts().to_dict()}")
        for t in ("addr_empty", "domain", "hashtag", "indic", "dba", "null_tok", "core_eq"):
            print(f"   {t:<10} {e[t].sum():4d}  ({100*e[t].mean():.1f}%)")
        print(f"   name_sim>=0.9 {np.sum(e.name_sim>=0.9)}  name_sim<0.5 {np.sum(e.name_sim<0.5)}")
        print(f"   S1 core freq: ==1 {np.sum(e.s1_core_freq==1)}  2-9 {np.sum((e.s1_core_freq>=2)&(e.s1_core_freq<10))}  >=10 {np.sum(e.s1_core_freq>=10)}")
        if k == "fp":
            print(f"   owned by another S1: {e.owner.notna().sum()}  distractor (unlinked): {e.owner.isna().sum()}")
    # base rates of the tags among ALL val GT records (to compare)
    allg = [c for a in val for c in G.get(a, set())]
    bt = pd.DataFrame([tags(rec.at[c, "name"], rec.at[c, "addr"]) for c in allg])
    print("\nbase rates among all val GT records (n=%d):" % len(allg), {k: round(100 * bt[k].mean(), 2) for k in bt.columns})


if __name__ == "__main__":
    main()
