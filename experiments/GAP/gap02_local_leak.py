"""GAP-02: local (non-linear) ID / row-order leakage test on TRAIN. RL-01 only tested global correlation.
For S1 with >=2 records in the same source: are sibling records closer in file row / id number than random pairs?"""
import numpy as np, pandas as pd, pickle, os
d = pickle.load(open(os.path.join("..", "RL", "cache", "train.pkl"), "rb"))
gt = d["gt"]; rng = np.random.default_rng(0)
out = {}
for src in ("s2", "s3"):
    df = d[src]; row = pd.Series(np.arange(len(df)), index=df["id"].values)
    num = pd.Series(df["id"].str[3:].astype(np.int64).values, index=df["id"].values)
    g = gt[gt.rec.str.startswith(src.upper())].copy()
    g["row"] = row.reindex(g.rec).values; g["num"] = num.reindex(g.rec).values
    g = g.sort_values(["s1", "row"])
    first = g.groupby("s1").head(2)
    sz = first.groupby("s1").size(); first = first[first.s1.isin(sz[sz == 2].index)]
    a = first.groupby("s1").agg(r0=("row", "min"), r1=("row", "max"), n0=("num", "min"), n1=("num", "max"))
    sib_row = (a.r1 - a.r0).values; sib_num = (a.n1 - a.n0).values
    # null: random pairs of linked records
    k = len(a); i = rng.integers(0, len(g), k); j = rng.integers(0, len(g), k)
    null_row = np.abs(g.row.values[i] - g.row.values[j]); null_num = np.abs(g.num.values[i] - g.num.values[j])
    q = [0.001, 0.01, 0.05, 0.5]
    out[src] = dict(n_sibling_pairs=int(k),
                    sib_row_q=np.quantile(sib_row, q).tolist(), null_row_q=np.quantile(null_row, q).tolist(),
                    sib_num_q=np.quantile(sib_num, q).tolist(), null_num_q=np.quantile(null_num, q).tolist(),
                    frac_sib_row_gap_lt_1000=float((sib_row < 1000).mean()), frac_null_row_gap_lt_1000=float((null_row < 1000).mean()))
    print(src, out[src])
# S1 row vs record row: are records of S1 at file position x located near a fixed map of x? (binned mutual information)
s1row = pd.Series(np.arange(len(d["s1"])), index=d["s1"]["id"].values)
for src in ("s2", "s3"):
    df = d[src]; row = pd.Series(np.arange(len(df)), index=df["id"].values)
    g = gt[gt.rec.str.startswith(src.upper())]
    x = s1row.reindex(g.s1).values; y = row.reindex(g.rec).values
    B = 200; hx = (x * B // (len(s1row))).astype(int); hy = (y * B // len(df)).astype(int)
    H = np.zeros((B, B)); np.add.at(H, (hx, hy), 1); P = H / H.sum(); px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    nz = P > 0; mi = float((P[nz] * np.log2(P[nz] / (px @ py)[nz])).sum())
    # permutation null
    yp = rng.permutation(hy); H2 = np.zeros((B, B)); np.add.at(H2, (hx, yp), 1); P2 = H2 / H2.sum(); nz2 = P2 > 0
    mi0 = float((P2[nz2] * np.log2(P2[nz2] / (P2.sum(1, keepdims=True) @ P2.sum(0, keepdims=True))[nz2])).sum())
    out[src + "_MI_bits_200bins"] = [mi, mi0]; print(src, "MI bits (actual, permuted):", mi, mi0)
import json; json.dump(out, open("gap02_local_leak.json", "w"), indent=1)
