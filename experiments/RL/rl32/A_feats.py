"""RL-32 / Investigator A, step 0: S1-ONLY strata features for every train and test S1 (label-free for test).
Features (all computed within (split, country) S1 population, no records / no labels used):
  core      folded name tokens minus legal tokens (dots removed first so L.L.C. -> llc), sorted
  k_core    #S1 sharing core
  k_addr    #S1 sharing the exact normalised address (comma components folded + abbrev-mapped, order-invariant)
  k_cs      #S1 sharing (core, address without digits)   -> same-street same-core sibling if >=2
  k_ce      #S1 sharing (core, exact address)             -> exact duplicate S1
  hasnum    address contains a digit
  ntok      #core tokens
  city      most frequent (country-wide) digit-free component that is not the state (= most frequent component)
  k_city    #S1 in that city
Output: A_feats.pkl (one row per S1: split, id, country, feature columns)."""
import re, unicodedata, pickle, os, time, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
LEGAL = set("llc inc corp corporation ltd limited pvt private llp co company lp pc pllc plc sarl sas sasu eurl sa sci snc selarl scop".split())
ABBR = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard",
        "bd": "boulevard", "pl": "place", "r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route",
        "hwy": "highway", "pkwy": "parkway", "cir": "circle", "sq": "square", "fg": "faubourg"}


def fold(s):
    s = unicodedata.normalize("NFKD", (s or "").lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def one(args):
    n, a = args
    toks = [w for w in fold((n or "").replace(".", "")).split() if w not in LEGAL]
    core = " ".join(sorted(toks))
    comps = []
    for c in (a or "").split(","):
        t = [ABBR.get(w, w) for w in fold(c).split()]
        if t: comps.append(" ".join(t))
    ex = "|".join(sorted(comps))
    nn = "|".join(sorted(x for x in (" ".join(w for w in c.split() if not w.isdigit()) for c in comps) if x))
    hasnum = any(ch.isdigit() for ch in (a or ""))
    nodig = [c for c in comps if not any(ch.isdigit() for ch in c)]
    return core, len(toks), ex, nn, hasnum, "\t".join(nodig)


def feats(df):
    with Pool(6) as P:
        R = P.map(one, list(zip(df.name.values, df.addr.values)), chunksize=20000)
    F = pd.DataFrame(R, columns=["core", "ntok", "ax", "an", "hasnum", "nodig"])
    F["id"] = df.id.values; F["country"] = df.country.values
    out = []
    for c, g in F.groupby("country"):
        g = g.copy()
        g["k_core"] = g.groupby("core").core.transform("size")
        g["k_addr"] = g.groupby("ax").ax.transform("size")
        g["k_cs"] = g.groupby(["core", "an"]).core.transform("size")
        g["k_ce"] = g.groupby(["core", "ax"]).core.transform("size")
        comps = g.nodig.str.split("\t")
        ex = comps.explode(); vc = ex[ex != ""].value_counts()
        def city(lst):
            lst = [x for x in lst if x]
            if not lst: return ""
            lst = sorted(lst, key=lambda x: -vc.get(x, 0))
            return lst[1] if len(lst) >= 2 else ""   # [0] = state/region (most frequent component)
        g["city"] = [city(l) for l in comps.values]
        g["k_city"] = g.groupby("city").city.transform("size").where(g.city != "", 0)
        out.append(g)
    return pd.concat(out).drop(columns=["nodig"])


if __name__ == "__main__":
    t0 = time.time(); res = []
    for split in ("train", "test"):
        d = pickle.load(open(os.path.join(RL, "cache", f"{split}.pkl"), "rb"))
        F = feats(d["s1"]); F["split"] = split
        if split == "train":
            ng = d["gt"].groupby("s1").size(); F["n"] = F.id.map(ng).fillna(0).astype(np.int16)
        res.append(F); print(split, len(F), round(time.time() - t0, 1), flush=True)
    F = pd.concat(res, ignore_index=True)
    F.to_pickle(os.path.join(HERE, "A_feats.pkl"))
    for (s, c), g in F.groupby(["split", "country"]):
        print(s, c, len(g), "top cities:", g.city.value_counts().head(6).to_dict())
    print("secs", round(time.time() - t0, 1))
