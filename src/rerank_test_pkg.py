"""
rerank_test_pkg.py -- build per-country reranker inputs for TEST and write reranker scores back into shard files.

pkg  <country>  : pairs = base-model top-10 per test S1 (from pred_<model>_<country>_full.pkl 'base_top10' or
                  basetop10_<model>_<country>_full.pkl); raw texts streamed from the test source files.
                  -> experiments/TEST_PIPELINE/rerank_pkg/<country>_pairs.tsv.gz, <country>_texts.tsv.gz
attach <country>: scores (float32, aligned with pairs) -> per-shard rr_XXXX.pkl files next to shard_XXXX.pkl
Label-free (test data only).
"""
import os, sys, gzip, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

T = os.path.join(H.ROOT, "experiments", "TEST_PIPELINE"); PK = os.path.join(T, "rerank_pkg"); os.makedirs(PK, exist_ok=True)
TEST = os.path.join(H.ROOT, "student_resource", "dataset", "test")
MODEL = "E014B_s42"


def top10(country):
    p = os.path.join(T, f"basetop10_{MODEL}_{country}_full.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    d = pickle.load(open(os.path.join(T, f"pred_{MODEL}_{country}_full.pkl"), "rb"))
    assert d.get("base_top10"), f"no base_top10 for {country}; run predict_test.py basetop"
    return d["base_top10"]


def pkg(country):
    tt = top10(country)
    need = set(tt)
    with gzip.open(os.path.join(PK, f"{country}_pairs.tsv.gz"), "wt", encoding="utf-8") as f:
        f.write("s1\tcand\n")
        for s, L in tt.items():
            for c in L:
                f.write(f"{s}\t{c}\n"); need.add(c)
    n = 0
    with gzip.open(os.path.join(PK, f"{country}_texts.tsv.gz"), "wt", encoding="utf-8") as out:
        out.write("id\tname\taddr\tcountry\n")
        for src in "123":
            with open(os.path.join(TEST, f"test_source{src}.tsv"), encoding="utf-8") as f:
                f.readline()
                for line in f:
                    i = line.find("\t")
                    if line[:i] in need:
                        p = line.rstrip("\r\n").split("\t"); p += [""] * (4 - len(p))
                        out.write("\t".join(x.replace("\t", " ") for x in p[:4]) + "\n"); n += 1
    print(country, "S1", len(tt), "pairs", sum(len(v) for v in tt.values()), "texts", n, "needed", len(need), flush=True)


def attach(country, scores_path):
    sc = np.load(scores_path).astype(np.float32)
    pairs = []
    with gzip.open(os.path.join(PK, f"{country}_pairs.tsv.gz"), "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, c = line.rstrip("\n").split("\t"); pairs.append((s, c))
    assert len(pairs) == len(sc), (len(pairs), len(sc))
    by = {}
    for (s, c), v in zip(pairs, sc):
        by.setdefault(s, {})[c] = float(v)
    info = json.load(open(os.path.join(T, f"shards_{country}_full", "prepare_info.json")))
    for sp in info["shards"]:
        d = pickle.load(open(sp, "rb"))
        pickle.dump({s: by.get(s, {}) for s in d["s1"]}, open(sp.replace("shard_", "rr_"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(country, "attached", len(sc), "scores to", len(info["shards"]), "shards", flush=True)


if __name__ == "__main__":
    {"pkg": lambda: pkg(sys.argv[2]), "attach": lambda: attach(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
