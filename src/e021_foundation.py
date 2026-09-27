"""
E021 -- data foundation for the A100 session (label-free retrieval; labels only attached as targets / for evaluation).

1. sample : disjoint S1 sets drawn from train (random.Random(2109) shuffle of all eligible train S1):
             V1 = 20,000 (new, larger validation), T2X = 40,000 (stage-2 expansion), TR = 100,000 (reranker training),
             TD = 400,000 (bi-encoder training). Excluded from all sets: the 3,995 RECON sample (V0 + original 1,994 train),
             the E014 10,000, and every S1 of the REDTEAM r06 dense slices (Jaipur, Oregon).
2. retrieve: deep lexical retrieval with the production engine (RECON-04 config: char-4-gram BM25, k1 1.5, b .75,
             max_df 5000; normalize() text) for Q = sample 3,995 + E014 10k + V1 + T2X + TR, per country and source:
             n4addr top-200 and n4name top-50. Any (k_addr <= 200, k_name <= 50) pool is a prefix slice; (50, 10) must
             reproduce the production/frozen pool (checked by `verify`).
3. tdpairs: raw texts for TD S1 and their ground-truth S2/S3 records (bi-encoder positives).
Outputs: experiments/E021/{sets.json, s1.pkl, deep_<country>_S<src>.pkl, td_<country>.pkl, verify.json}
"""
import os, sys, json, time, pickle, random, gc, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from recon05_baseline_scorer import normalize

OUT = os.path.join(H.ROOT, "experiments", "E021"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(H.ROOT, "student_resource", "dataset")
SIZES = [("V1", 20000), ("T2X", 40000), ("TR", 100000), ("TD", 400000)]
K_ADDR, K_NAME = 200, 50
TAB, NL, CR = chr(9), chr(10), chr(13)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_table(split, src, country=None):
    """(ids, raw names, raw addrs, countries) of one source table, optionally one country."""
    pre = "train" if split == "train" else "test"
    ids, nm, ad, ct = [], [], [], []
    with open(os.path.join(DATA, split, f"{pre}_source{src}.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(CR + NL).split(TAB); p += [""] * (4 - len(p))
            if country is not None and p[3] != country:
                continue
            ids.append(p[0]); nm.append(p[1]); ad.append(p[2]); ct.append(p[3])
    return ids, nm, ad, ct


def read_gt(want=None):
    gt = {}
    with open(os.path.join(DATA, "train", "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, _, rest = line.rstrip(CR + NL).partition(TAB)
            if want is None or s in want:
                gt[s] = {x for x in rest.split(",") if x}
    return gt


def excluded():
    D = H.load_e008(verbose=False)
    ex = {"sample3995": set(D["s1_dict"])}
    C14 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E014", "e014_feats_10000_translit.pkl"), "rb"))
    ex["e014"] = set(C14["new_ids"]); del C14
    r6 = set()
    for t in ["India_jaipur", "US_OR"]:
        r6 |= set(pickle.load(open(os.path.join(H.ROOT, "experiments", "REDTEAM", "r06", f"preds_{t}.pkl"), "rb"))["ids"])
    ex["r06_slices"] = r6
    return ex


def sample():
    ex = excluded(); allex = set().union(*ex.values())
    ids, nm, ad, ct = read_table("train", "1")
    elig = [s for s in ids if s not in allex]
    rng = random.Random(2109); rng.shuffle(elig)
    sets, o = {}, 0
    for name, n in SIZES:
        sets[name] = sorted(elig[o:o + n]); o += n
    D = H.load_e008(verbose=False)
    sets["V0"] = sorted(D["val_s1_ids"]); sets["T0"] = sorted(D["train_s1_ids"]); sets["E014"] = sorted(ex["e014"])
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (set(sets[names[i]]) & set(sets[names[j]])), (names[i], names[j])
    assert not (set().union(*[set(sets[k]) for k in ("V1", "T2X", "TR", "TD")]) & ex["r06_slices"])
    json.dump(dict(seed=2109, sizes=dict(SIZES), excluded={k: len(v) for k, v in ex.items()},
                   n_eligible=len(elig), sets=sets), open(os.path.join(OUT, "sets.json"), "w"))
    q = set().union(*[set(sets[k]) for k in ("V0", "T0", "E014", "V1", "T2X", "TR", "TD")])
    gt = read_gt(q)
    s1 = {}
    for s, n_, a_, c_ in zip(ids, nm, ad, ct):
        if s in q:
            s1[s] = dict(raw_name=n_, raw_addr=a_, country=c_, name_r=normalize(n_), addr_r=normalize(a_), gt=gt.get(s, set()))
    pickle.dump(s1, open(os.path.join(OUT, "s1.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    for k in sets:
        cc = collections.Counter(s1[s]["country"] for s in sets[k])
        log(f"{k:5} {len(sets[k]):7,} US {cc['US']:,} India {cc['India']:,} singletons {sum(1 for s in sets[k] if not s1[s]['gt']):,} "
            f"gt pairs {sum(len(s1[s]['gt']) for s in sets[k]):,}")


def _retrieve_one(args):
    country, src, q_ids, q_names, q_addrs = args
    import retrieval_engine as RE
    t0 = time.time()
    ids, nm, ad, _ = read_table("train", src, country)
    names = [normalize(x) for x in nm]; addrs = [normalize(x) for x in ad]; del nm, ad
    t_load = time.time() - t0
    out = dict(country=country, src=src, q_ids=q_ids, ids=ids, timings=dict(load=t_load))
    qa = [(n + " " + a) if a else n for n, a in zip(q_names, q_addrs)]
    for kind, texts, qt, k in [("addr", [(n + " " + a) if a else n for n, a in zip(names, addrs)], qa, K_ADDR),
                               ("name", names, q_names, K_NAME)]:
        tb = time.time(); eng = RE.BM25Engine().build(texts, verbose=False); out["timings"][f"{kind}_build"] = time.time() - tb
        tq = time.time(); out[kind] = np.vstack([eng.query(qt[i:i + 20000], k) for i in range(0, len(qt), 20000)])
        out["timings"][f"{kind}_query"] = time.time() - tq
        del eng, texts; gc.collect()
    pickle.dump(out, open(os.path.join(OUT, f"deep_{country}_S{src}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return country, src, out["timings"], len(ids)


def retrieve():
    import multiprocessing as mp
    sets = json.load(open(os.path.join(OUT, "sets.json")))["sets"]
    s1 = pickle.load(open(os.path.join(OUT, "s1.pkl"), "rb"))
    D = H.load_e008(verbose=False)
    q_all = sets["V0"] + sets["T0"] + sets["E014"] + sets["V1"] + sets["T2X"] + sets["TR"]
    jobs = []
    for c in ["US", "India"]:
        q = [s for s in q_all if s1[s]["country"] == c]
        # sample S1 use the golden normalized text (identical to normalize(raw); asserted)
        for s in q[:50]:
            if s in D["s1_dict"]:
                assert D["s1_dict"][s]["name"] == s1[s]["name_r"] and D["s1_dict"][s]["addr"] == s1[s]["addr_r"], s
        for src in ["2", "3"]:
            jobs.append((c, src, q, [s1[s]["name_r"] for s in q], [s1[s]["addr_r"] for s in q]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        for c, src, tm, n in pool.imap_unordered(_retrieve_one, jobs):
            log(f"deep {c} S{src}: {n:,} docs", {k: round(v) for k, v in tm.items()})
    log(f"retrieval total {time.time()-t0:.0f}s")


def load_deep(country):
    return {src: pickle.load(open(os.path.join(OUT, f"deep_{country}_S{src}.pkl"), "rb")) for src in ["2", "3"]}


def pool_ids(deep, qi, k_addr, k_name):
    """Candidate dict {cand: {is_s2, rank_addr, rank_name}} for query index qi (same semantics as BP.to_pool_dicts)."""
    d = {}
    for src, is_s2 in [("2", 1), ("3", 0)]:
        P = deep[src]; ids = P["ids"]
        for r, j in enumerate(P["addr"][qi, :k_addr]):
            if j < 0: break
            d[ids[j]] = {"is_s2": is_s2, "rank_addr": r + 1, "rank_name": 999}
        for r, j in enumerate(P["name"][qi, :k_name]):
            if j < 0: break
            c = ids[j]
            if c in d: d[c]["rank_name"] = r + 1
            else: d[c] = {"is_s2": is_s2, "rank_addr": 999, "rank_name": r + 1}
    return d


def verify():
    """(50,10) slice == frozen pool for the 3,995 sample and == E014 engine pool for the E014 10k."""
    import build_pools as BP
    D = H.load_e008(verbose=False)
    sets = json.load(open(os.path.join(OUT, "sets.json")))["sets"]
    res = {}
    for c in ["US", "India"]:
        deep = load_deep(c); qpos = {s: i for i, s in enumerate(deep["2"]["q_ids"])}
        E = pickle.load(open(os.path.join(BP.POOL_DIR, f"train_{c}_e014_10000.pkl"), "rb"))
        e14 = BP.to_pool_dicts(E, subset=set(sets["E014"]))
        jac_f, jac_e, same_rank = [], [], 0
        for s in D["s1_dict"]:
            if s in qpos:
                p = pool_ids(deep, qpos[s], 50, 10); a, b = set(D["cands"][s]), set(p)
                jac_f.append(len(a & b) / max(len(a | b), 1))
        for s, pe in e14.items():
            p = pool_ids(deep, qpos[s], 50, 10); a, b = set(pe), set(p)
            jac_e.append(len(a & b) / max(len(a | b), 1))
            same_rank += all(pe[x]["rank_addr"] == p[x]["rank_addr"] and pe[x]["rank_name"] == p[x]["rank_name"] for x in a & b)
        res[c] = dict(frozen_mean_jaccard=float(np.mean(jac_f)), frozen_identical=float(np.mean(np.array(jac_f) == 1)), n_sample=len(jac_f),
                      e014_mean_jaccard=float(np.mean(jac_e)), e014_identical=float(np.mean(np.array(jac_e) == 1)),
                      e014_same_ranks=same_rank / max(len(e14), 1), n_e014=len(jac_e))
        log(c, res[c])
    json.dump(res, open(os.path.join(OUT, "verify.json"), "w"), indent=1)


def tdpairs():
    sets = json.load(open(os.path.join(OUT, "sets.json")))["sets"]
    s1 = pickle.load(open(os.path.join(OUT, "s1.pkl"), "rb"))
    td = sets["TD"]; need = set().union(*[s1[s]["gt"] for s in td])
    texts = {}
    for src in ["2", "3"]:
        ids, nm, ad, ct = read_table("train", src)
        for i, n_, a_ in zip(ids, nm, ad):
            if i in need:
                texts[i] = (n_, a_)
        del ids, nm, ad, ct
    for c in ["US", "India"]:
        q = [s for s in td if s1[s]["country"] == c]
        out = dict(s1_ids=q, s1_text=[(s1[s]["raw_name"], s1[s]["raw_addr"]) for s in q], gt=[sorted(s1[s]["gt"]) for s in q],
                   texts={g: texts[g] for s in q for g in s1[s]["gt"]})
        pickle.dump(out, open(os.path.join(OUT, f"td_{c}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        log(f"TD {c}: {len(q):,} S1, {sum(len(g) for g in out['gt']):,} positive pairs")


if __name__ == "__main__":
    {"sample": sample, "retrieve": retrieve, "verify": verify, "tdpairs": tdpairs}[sys.argv[1]]()
