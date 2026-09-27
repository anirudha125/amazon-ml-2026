"""WD part 5: refined label-free token classes on the model-free same-address pool frames, then the labelled check.
Per frame (country pool), label-free token stats at same address: n_add (pure ADD1), n_sw (swap-to), S1 df.
  appended(t)      : n_add >= 20 and n_add / (n_add + n_sw) >= 0.05      ('is sometimes appended purely' -- D's filler notion, model-free)
  distinguishing(t): n_sw >= 10, n_add / (n_add + n_sw) < 0.01, len >= 4, S1 df >= 20   (never appended, used as a swap target)
  D_ratio(t)       : (n_add + n_sw) / S1_df  (D's added_vs_in_names, here on the model-free pool)
Checks: (1) labelled P(match) of appended tokens as pure adds (are all 'appended' tokens filler?), (2) P(match) of same-address swaps
to distinguishing tokens, (3) V1 model errors in the class, (4) D_ratio-based classes vs labels, (5) France class sizes / acceptance / S1 support.
Writes rl30/WD_5_results.json"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH
from WD_common import s1_df

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test", verbose=False); Tr = load("train", verbose=False)


def tokstats(d, df):
    add = collections.Counter(d.b[d.kind == "ADD1"]); sw = collections.Counter(d.b[d.kind == "SWAP1"])
    toks_ = set(add) | set(sw)
    t = pd.DataFrame(dict(tok=list(toks_)))
    t["n_add"] = t.tok.map(add).fillna(0).astype(int); t["n_sw"] = t.tok.map(sw).fillna(0).astype(int)
    t["S1_df"] = t.tok.map(lambda x: df.get(x, 0))
    t["add_share"] = t.n_add / (t.n_add + t.n_sw)
    t["D_ratio"] = (t.n_add + t.n_sw) / t.S1_df.clip(lower=1)
    t["appended"] = (t.n_add >= 20) & (t.add_share >= 0.05)
    t["distinguishing"] = (t.n_sw >= 10) & (t.add_share < 0.01) & (t.tok.str.len() >= 4) & (t.S1_df >= 20)
    return t.set_index("tok")


def classes(d, ts, df):
    app = set(ts.index[ts.appended]); dis = set(ts.index[ts.distinguishing])
    dfa = d.a.map(lambda x: df.get(x, 0))
    c = np.full(len(d), "", dtype=object)
    add = (d.kind == "ADD1").values; sw = (d.kind == "SWAP1").values
    b_app = d.b.isin(app).values; b_dis = d.b.isin(dis).values
    a_ok = ((dfa >= 20) & (d.a.str.len() >= 4)).values
    c[add & b_app] = "ADD_appended"; c[add & ~b_app] = "ADD_other"
    c[sw & b_app] = "SWAP_to_appended"; c[sw & b_dis & a_ok] = "SWAP_word_to_distinguishing"
    c[sw & ~b_app & ~(b_dis & a_ok)] = "SWAP_other"
    return c


# ---------------- labelled train pools
d = pd.read_pickle(os.path.join(OUT, "WD_2_pairs_TRAIN.pkl"))
d["acc"] = d.p >= NEW_TH
meta = json.load(open(os.path.join(OUT, "WD_2_meta_TRAIN.json")))
m = np.load(PATHS["v1_meta"]); pv = np.load(PATHS["v1_p_new"]); yv = m["y"]; cv = m["country"][m["s1idx"]]
out = {}
for cc in ("US", "India"):
    df = s1_df(Tr["s1"].name[Tr["s1"].country == cc].values)
    x = d[d.country == cc].copy()
    ts = tokstats(x, df)
    # label per token (for evaluation only)
    tadd = x[x.kind == "ADD1"].groupby("b").y.agg(["size", "mean"]); tsw = x[x.kind == "SWAP1"].groupby("b").y.agg(["size", "mean"])
    ts["p_match_add"] = tadd["mean"].reindex(ts.index); ts["p_match_sw"] = tsw["mean"].reindex(ts.index)
    app = ts[ts.appended].sort_values("n_add", ascending=False)
    L(f"==== TRAIN {cc}: 'appended' tokens (label-free) with their labelled P(match) as pure adds"); L(app.round(3).to_string())
    x["cls"] = classes(x, ts, df)
    n_s1 = sum(meta[f"TRAIN_{k}"]["n_s1_by_country"].get(cc, 0) for k in ("V1", "T2X", "E014", "T0"))
    g = x[x.cls != ""].groupby("cls").agg(n=("y", "size"), n_pos=("y", "sum"), p_match=("y", "mean"))
    g["per_10k_S1"] = 1e4 * g.n / n_s1
    L(g.round(4).to_string())
    v1 = x[x.pool == "V1"]
    cls = {}
    for k, gg in v1[v1.cls != ""].groupby("cls"):
        cls[k] = dict(n=int(len(gg)), n_pos=int(gg.y.sum()), n_acc=int(gg.acc.sum()), FP=int((gg.acc & (gg.y == 0)).sum()), FN=int((~gg.acc & (gg.y == 1)).sum()))
    tot = dict(FP=int(((pv >= NEW_TH) & (yv == 0) & (cv == cc)).sum()), FN=int(((pv < NEW_TH) & (yv == 1) & (cv == cc)).sum()))
    L("V1 model in class", json.dumps(cls), "| all V1", tot)
    # D_ratio-based split of pure adds + swaps (D's notion: high ratio = filler, low ratio = distinguishing)
    x["D_ratio_b"] = x.b.map(ts.D_ratio)
    q = x[x.kind.isin(["ADD1", "SWAP1"]) & (x.b.map(lambda t: df.get(t, 0)) >= 20)]
    q = q.assign(Dbin=pd.cut(q.D_ratio_b, [0, 0.01, 0.03, 0.1, 0.3, 1, 1e9]))
    gq = q.groupby(["kind", "Dbin"], observed=True).y.agg(["size", "mean"])
    L("P(match) by D_ratio bin of the record-only token (vocab tokens, S1 df>=20):"); L(gq.round(3).to_string())
    sw_dis = x[x.cls == "SWAP_word_to_distinguishing"]
    out[cc] = dict(n_s1=int(n_s1), appended_tokens=app.reset_index()[["tok", "S1_df", "n_add", "n_sw", "add_share", "D_ratio", "p_match_add"]].round(4).values.tolist(),
                   by_class=g.round(5).reset_index().values.tolist(), V1_model_in_class=cls, V1_total=tot,
                   D_ratio_bins=[[str(k[0]), str(k[1]), int(r["size"]), round(float(r["mean"]), 4)] for k, r in gq.iterrows()],
                   dis_swap_top=sw_dis.groupby(["a", "b"]).y.agg(["size", "sum"]).sort_values("size", ascending=False).head(15).reset_index().values.tolist())
    L("distinguishing swaps top pairs:", out[cc]["dis_swap_top"])
    s1n = Tr["s1"].set_index("id"); rn = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
    ex = x[(x.cls == "ADD_appended") & (x.y == 0)].sample(frac=1, random_state=0).head(6)
    out[cc]["appended_but_false_examples"] = [f"{s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in ex.itertuples()]
    ex = sw_dis.sample(frac=1, random_state=0).head(8)
    out[cc]["dis_swap_examples"] = [f"y={r.y} p={r.p:.3f} | {s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in ex.itertuples()]
    for e in out[cc]["appended_but_false_examples"] + ["---"] + out[cc]["dis_swap_examples"]:
        L("   ", e)
R["train"] = out
del d

# ---------------- test frames
for j, cc in (("FR", "France"), ("US", "US"), ("IN", "India")):
    d = pd.read_pickle(os.path.join(OUT, f"WD_2_pairs_{j}.pkl"))
    df = s1_df(T["s1"].name[T["s1"].country == cc].values)
    ts = tokstats(d, df)
    ts["acc_rate_add"] = d[d.kind == "ADD1"].groupby("b").acc.mean().reindex(ts.index)
    ts["acc_rate_sw"] = d[d.kind == "SWAP1"].groupby("b").acc.mean().reindex(ts.index)
    d["cls"] = classes(d, ts, df)
    n_s1 = json.load(open(os.path.join(OUT, "WD_2_meta_FR_US_IN.json")))[j]["n_s1"]
    g = d[d.cls != ""].groupby("cls").agg(n=("acc", "size"), acc_rate=("acc", "mean"), n_final=("kept_final", "sum"), n_s1_final=("s1", lambda s: 0))
    g["n_s1_final"] = [d[(d.cls == k) & d.kept_final].s1.nunique() for k in g.index]
    g["per_10k_S1"] = 1e4 * g.n / n_s1; g["S1_share_final_%"] = 100 * g.n_s1_final / n_s1
    app = ts[ts.appended].sort_values("n_add", ascending=False)
    L(f"==== TEST {cc} (S1 in frame {n_s1:,})"); L(g.round(4).to_string()); L("appended tokens:"); L(app.head(30).round(3).to_string())
    R[f"test_{cc}"] = dict(n_s1=int(n_s1), by_class=g.round(5).reset_index().values.tolist(),
                           appended_tokens=app.head(40).reset_index()[["tok", "S1_df", "n_add", "n_sw", "add_share", "D_ratio", "acc_rate_add", "acc_rate_sw"]].round(4).values.tolist(),
                           n_distinguishing_tokens=int(ts.distinguishing.sum()))
    if cc == "France":
        s1n = T["s1"].set_index("id"); rn = pd.concat([T["s2"], T["s3"]]).set_index("id")
        ex = d[(d.kind == "ADD1") & d.b.isin(["groupe", "holding", "distribution", "international", "participations"])].sample(frac=1, random_state=0).head(8)
        R["France_appended_rejected_examples"] = [f"acc={r.acc} | {s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in ex.itertuples()]
        for e in R["France_appended_rejected_examples"]:
            L("   ", e)
    del d
json.dump(R, open(os.path.join(OUT, "WD_5_results.json"), "w"), indent=1, default=str)
L("wrote WD_5_results.json")
