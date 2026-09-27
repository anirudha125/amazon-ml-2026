"""E027_FR step 1: parse France names/addresses, scan the FULL France candidate pool (experiments/P3/France/chunk_*.npz),
keep pairs whose street remainder is equal, and record house-number delta + one-token name edit.
Output: pairs_streq.npz (s1 idx, cand idx, delta, edit type, tok_a, tok_b) + ent.pkl.  CPU only, <=12 procs."""
import os, re, sys, pickle, unicodedata, glob, time, collections
import numpy as np
from multiprocessing import Pool

ROOT = "/teamspace/studios/this_studio/amazon-ml-challenge-2026"
OUT = os.path.join(ROOT, "experiments", "E027_FR")
TEST = os.path.join(ROOT, "student_resource", "dataset", "test")

NAME_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "the", "of", "a", "au", "aux", "en"}
STREET_STOP = NAME_STOP | {"no", "n", "bis", "ter", "quater"}
ABBR = {"r": "rue", "av": "avenue", "ave": "avenue", "avn": "avenue", "bd": "boulevard", "blvd": "boulevard", "boul": "boulevard",
        "bld": "boulevard", "pl": "place", "ch": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
        "sq": "square", "qu": "quai", "crs": "cours", "fg": "faubourg", "fbg": "faubourg", "pas": "passage", "pass": "passage",
        "st": "saint", "ste": "sainte", "res": "residence", "lot": "lotissement", "prom": "promenade", "sent": "sentier",
        "rpt": "rondpoint", "esp": "esplanade", "mte": "montee", "hle": "halle", "zi": "zone", "za": "zone", "zac": "zone"}
TOK = re.compile(r"[a-z0-9]+")


def fold(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def name_toks(s):
    s = fold(s).replace(".", "")
    return tuple(sorted(t for t in TOK.findall(s) if t not in NAME_STOP))


def addr_parse(s):
    """first house number (int) and normalized street remainder of the comma-component that holds it."""
    for comp in fold(s).split(","):
        toks = TOK.findall(comp)
        for i, t in enumerate(toks):
            m = re.fullmatch(r"(\d+)[a-z]?", t)
            if m and len(m.group(1)) <= 5:
                rest = [ABBR.get(x, x) for j, x in enumerate(toks) if j != i]
                rest = [x for x in rest if x not in STREET_STOP and not x.isdigit()]
                return int(m.group(1)), " ".join(rest)
    return -1, ""


def load_entities():
    names, num, street, ids = [], [], [], []
    for src in (1, 2, 3):
        with open(os.path.join(TEST, f"test_source{src}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4 or p[3] != "France":
                    continue
                ids.append(p[0]); names.append(name_toks(p[1])); n, st = addr_parse(p[2]); num.append(n); street.append(st)
    return ids, names, np.array(num, np.int64), street


def init():
    pass


def edit(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    da, db = list((ca - cb).elements()), list((cb - ca).elements())
    if not da and not db:
        return 0, "", ""
    if len(da) == 1 and len(db) == 1:
        return 1, da[0], db[0]          # substitution a->b
    if not da and len(db) == 1:
        return 2, "", db[0]             # add (cand has one extra token)
    if len(da) == 1 and not db:
        return 3, da[0], ""             # drop (s1 has one extra token)
    return 4, "", ""                    # other


def work(path):
    z = np.load(path)
    s1 = z["s1"]; cd = z["cand"]
    i1 = np.fromiter((IDX[x] for x in s1), np.int64, len(s1))
    i2 = np.fromiter((IDX.get(x, -1) for x in cd), np.int64, len(cd))
    ok = (i2 >= 0) & (SID[i1] == SID[np.maximum(i2, 0)]) & (SID[i1] > 0) & (NUM[i1] >= 0) & (NUM[np.maximum(i2, 0)] >= 0)
    a, b = i1[ok], i2[ok]
    et = np.empty(len(a), np.int8); ta = []; tb = []
    for k in range(len(a)):
        e, x, y = edit(NAMES[a[k]], NAMES[b[k]]); et[k] = e; ta.append(x); tb.append(y)
    return len(s1), int((i2 < 0).sum()), a, b, NUM[b] - NUM[a], et, ta, tb


if __name__ == "__main__":
    t0 = time.time()
    ids, NAMES, NUM, street = load_entities()
    IDX = {x: i for i, x in enumerate(ids)}
    svocab = {}
    SID = np.array([svocab.setdefault(s, len(svocab)) if s else 0 for s in street], np.int64)
    if "" not in svocab:
        pass
    # ensure empty street -> 0 and no real street got id 0
    SID = np.array([(svocab[s] + 1) if s else 0 for s in street], np.int64)
    print("entities", len(ids), "no-number", int((NUM < 0).sum()), "streets", len(svocab), f"{time.time()-t0:.0f}s", flush=True)
    files = sorted(glob.glob(os.path.join(ROOT, "experiments", "P3", "France", "chunk_*.npz")))
    tot = miss = 0; A = []; B = []; D = []; E = []; TA = []; TB = []
    with Pool(12) as pool:
        for r in pool.imap_unordered(work, files, chunksize=1):
            n, m, a, b, d, e, ta, tb = r; tot += n; miss += m
            A.append(a); B.append(b); D.append(d); E.append(e); TA += ta; TB += tb
    A = np.concatenate(A); B = np.concatenate(B); D = np.concatenate(D); E = np.concatenate(E)
    print("pool pairs", tot, "cand-missing", miss, "street-equal pairs", len(A), f"{time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "pairs_streq.npz"), a=A, b=B, d=D, e=E, ta=np.array(TA), tb=np.array(TB))
    pickle.dump(dict(ids=ids, names=NAMES, num=NUM, street=street, pool_pairs=tot), open(os.path.join(OUT, "ent.pkl"), "wb"))
    print("done", f"{time.time()-t0:.0f}s")
