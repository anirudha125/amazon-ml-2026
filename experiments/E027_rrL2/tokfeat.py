"""E027 section-3 diff-token block (5 cols, per-S1 top-10 only, NaN elsewhere):
  kind   : 0 names differ by more than one token or are equal, 1 one-token substitution, 2 one-token add/drop
  addreq : normalized addresses equal (incl. house number) and non-empty
  ldf_a  : log(df/N) of the S1-side differing token (sub) or the added/dropped token (add/drop); NaN if kind 0
  ldf_b  : log(df/N) of the candidate-side token (sub); NaN otherwise
  role   : 1 if every differing token is on the generic-role/filler list, else 0; NaN if kind 0
DF = document frequency over the same split's S2+S3 names of the same country (train split for T/V, test split for test)."""
import os, sys, pickle, math, unicodedata, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import e021_foundation as F
from recon05_baseline_scorer import normalize
ROLE = set("""club comite centre center amicale amis ami groupement groupe group societe society association assoc asso cercle union
foyer federation cooperative coop maison atelier cabinet agence company co cie compagnie inc llc corp corporation ltd limited
services service solutions enterprises enterprise industries international pvt private sarl sas sa eurl sci sasu fils associes
associates developpement development france india usa us holdings holding trust foundation institute store shop the and et de des
du la le les of""".split())
G = {}


def nrm(x):
    x = normalize(x or "")
    return "".join(ch for ch in unicodedata.normalize("NFKD", x) if not unicodedata.combining(ch))


def df_table(split, country):
    cnt = collections.Counter(); n = 0
    for src in "23":
        _, nm, _, _ = F.read_table(split, src, country)
        for x in nm:
            cnt.update(set(nrm(x).split())); n += 1
    return {t: math.log(c / n) for t, c in cnt.items()}, math.log(0.5 / n)


def feat(n1, a1, n2, a2, ldf, floor):
    t1, t2 = n1.split(), n2.split(); s1, s2 = set(t1), set(t2)
    ae = float(a1 == a2 and bool(a1))
    d1, d2 = s1 - s2, s2 - s1
    if len(t1) == len(t2) and len(d1) == 1 and len(d2) == 1:
        a, b = next(iter(d1)), next(iter(d2))
        return (1.0, ae, ldf.get(a, floor), ldf.get(b, floor), float(a in ROLE and b in ROLE))
    if abs(len(t1) - len(t2)) == 1 and len(d1) + len(d2) == 1:
        a = next(iter(d1 or d2))
        return (2.0, ae, ldf.get(a, floor), np.nan, float(a in ROLE))
    return (0.0, ae, np.nan, np.nan, np.nan)


def _work(chunk):
    tx, ldf, floor = G["tx"], G["ldf"], G["floor"]
    out = np.empty((len(chunk), 5), np.float32)
    for k, (s, c) in enumerate(chunk):
        (n1, a1), (n2, a2) = tx[s], tx[c]; out[k] = feat(n1, a1, n2, a2, ldf[G["ct"][s]], floor[G["ct"][s]])
    return out


def compute(pairs, tx, ct, ldf, floor, workers=12):
    import multiprocessing as mp
    G.update(tx=tx, ct=ct, ldf=ldf, floor=floor)
    ch = [pairs[i:i + 50000] for i in range(0, len(pairs), 50000)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_work, ch)
    return np.vstack(res) if res else np.zeros((0, 5), np.float32)


def load_trainval():
    p = os.path.join(HERE, "tokfeat_trainval.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    P = pickle.load(open(os.path.join(ROOT, "experiments", "E026_rrL", "cache", "top10_pairs.pkl"), "rb"))["all"]
    s1 = pickle.load(open(os.path.join(ROOT, "experiments", "E021", "s1.pkl"), "rb"))
    need = {x for pp in P for x in pp}; tx, ct = {}, {}
    for s in need:
        if s.startswith("S1-"): tx[s] = (nrm(s1[s]["raw_name"]), nrm(s1[s]["raw_addr"])); ct[s] = s1[s]["country"]
    for src in "23":
        i_, nm, ad, _ = F.read_table("train", src)
        for a, b, c in zip(i_, nm, ad):
            if a in need: tx[a] = (nrm(b), nrm(c))
    ldf, floor = {}, {}
    for c in ["US", "India"]: ldf[c], floor[c] = df_table("train", c)
    X = compute(P, tx, ct, ldf, floor); d = dict(zip(P, map(tuple, X)))
    pickle.dump(d, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL); return d


def load_test(country):
    p = os.path.join(HERE, f"tokfeat_test_{country}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    P = pickle.load(open(os.path.join(ROOT, "experiments", "P3_rrL", f"top10_{country}.pkl"), "rb"))
    tx = {}
    for src in "123":
        i_, nm, ad, _ = F.read_table("test", src, country)
        for a, b, c in zip(i_, nm, ad): tx[a] = (nrm(b), nrm(c))
    ldf, floor = df_table("test", country)
    ct = collections.defaultdict(lambda: country)
    X = compute(P, tx, ct, {country: ldf}, {country: floor}); d = dict(zip(P, map(tuple, X)))
    pickle.dump(d, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL); return d


def cols(S, sel, d):
    out = np.full((len(S["y"]) if "y" in S else len(sel), 5), np.nan, np.float32)
    ids = S["s1_ids"][S["s1idx"]]
    for i in np.flatnonzero(sel):
        out[i] = d[(ids[i], S["cand"][i])]
    return out


if __name__ == "__main__":
    import time
    t = time.time(); d = load_trainval(); print("trainval", len(d), round(time.time() - t))
    X = np.array(list(d.values())); print("kind share", [(k, round(float((X[:, 0] == k).mean()), 4)) for k in (0, 1, 2)], "role|kind>0", float(np.nanmean(X[X[:, 0] > 0, 4])))
    for c in sys.argv[1:]:
        t = time.time(); d = load_test(c); print("test", c, len(d), round(time.time() - t))
