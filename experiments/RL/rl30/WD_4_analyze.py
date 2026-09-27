"""WD part 4: analysis of the model-free same-address pool frames (WD_2_pairs_*.pkl).
A. circularity: pure-ADD1 / swap-to counts per token in the MODEL-FREE pool vs among accepted pairs (acceptance rate per token)
B. label-free token score from the pool (no model, no labels): filler(t) := pool pure-ADD1(t) >= 20 and ADD1(t)/S1_df(t) >= 0.02
C. labelled check on the train pools (y): P(match) of same-address single-token swaps / adds by record-token class;
   V1 only: RL-27 NEW accept / FP / FN in each class vs all V1 errors; support per 10k S1 (train pools vs France test pool)
Writes rl30/WD_4_results.json"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH
from WD_common import s1_df, core

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test"); Tr = load("train")
DF = {cc: s1_df(T["s1"].name[T["s1"].country == cc].values) for cc in ("France", "US", "India")}
DFtr = {cc: s1_df(Tr["s1"].name[Tr["s1"].country == cc].values) for cc in ("US", "India")}
meta = {}
for f in ("WD_2_meta_FR_US_IN.json", "WD_2_meta_TRAIN.json"):
    if os.path.exists(os.path.join(OUT, f)):
        meta.update(json.load(open(os.path.join(OUT, f))))
R["meta"] = meta
BW = ["club", "comite", "amis", "centre", "amicale", "societe", "ecole", "maison", "pharmacie", "federation", "association", "union", "sportive"]
FI = ["fils", "services", "associes", "developpement", "france", "cie", "compagnie"]


def token_table(d, df):
    add = collections.Counter(d.b[d.kind == "ADD1"]); add_acc = collections.Counter(d.b[(d.kind == "ADD1") & d.acc])
    sw = collections.Counter(d.b[d.kind == "SWAP1"]); sw_acc = collections.Counter(d.b[(d.kind == "SWAP1") & d.acc])
    rows = []
    for t in set(add) | set(sw):
        rows.append(dict(tok=t, S1_df=df.get(t, 0), pool_add=add[t], acc_add=add_acc[t], pool_swapto=sw[t], acc_swapto=sw_acc[t]))
    tt = pd.DataFrame(rows)
    tt["pool_add_ratio"] = tt.pool_add / tt.S1_df.clip(lower=1)
    tt["acc_rate_add"] = tt.acc_add / tt.pool_add.clip(lower=1); tt["acc_rate_swapto"] = tt.acc_swapto / tt.pool_swapto.clip(lower=1)
    return tt


def classify_b(d, df, filler):
    return np.where(d.kind.isin(["SWAP1", "ADD1"]) == False, "",
                    np.where(d.b.isin(filler), "filler", np.where(d.b.map(lambda t: df.get(t, 0)) >= 20, "vocab_nonfiller", "rare_or_garble")))


# ---------------- A/B test frames
FILL = {}
for j, cc in (("FR", "France"), ("US", "US"), ("IN", "India")):
    p = os.path.join(OUT, f"WD_2_pairs_{j}.pkl")
    if not os.path.exists(p):
        continue
    d = pd.read_pickle(p)
    n_s1 = meta.get(j, {}).get("n_s1", d.s1.nunique())
    tt = token_table(d, DF[cc])
    filler = set(tt[(tt.pool_add >= 20) & (tt.pool_add_ratio >= 0.02) & (tt.S1_df >= 1)].tok) | set(tt[(tt.pool_add >= 20) & (tt.S1_df == 0)].tok)
    FILL[cc] = filler
    tt.sort_values("pool_swapto", ascending=False).to_csv(os.path.join(OUT, f"WD_4_pooltokens_{cc}.csv"), index=False)
    sel = tt[tt.tok.isin(BW + FI)].set_index("tok").round(4)
    L(f"==== {cc} pool frame: same-address rows {len(d):,}, S1 {n_s1:,}"); L(sel.to_string())
    d["bcls"] = classify_b(d, DF[cc], filler)
    g = d[d.kind.isin(["SWAP1", "ADD1"])].groupby(["kind", "bcls"]).agg(n=("acc", "size"), acc_rate=("acc", "mean"), final_rate=("kept_final", "mean"))
    g["per_10k_S1"] = 1e4 * g.n / n_s1
    L(g.round(4).to_string())
    R[f"test_{cc}"] = dict(n_same_addr=int(len(d)), n_s1=int(n_s1), kind_share=d.kind.value_counts(normalize=True).round(4).to_dict(),
                            by_class=g.round(5).reset_index().values.tolist(), selected_tokens=sel.reset_index().values.tolist(),
                            n_filler_tokens=len(filler), fillers_top=tt[tt.tok.isin(filler)].sort_values("pool_add", ascending=False).head(25)[["tok", "S1_df", "pool_add", "pool_add_ratio", "acc_rate_add"]].round(4).values.tolist())
    # circularity: business words in the pool as pure adds -- how often do they occur and how often does the model accept them?
    bw_add = tt[(tt.S1_df >= 200) & ~tt.tok.isin(filler)]
    R[f"test_{cc}"]["nonfiller_vocab_pool_add_total"] = int(bw_add.pool_add.sum())
    R[f"test_{cc}"]["nonfiller_vocab_acc_add_total"] = int(bw_add.acc_add.sum())
    R[f"test_{cc}"]["nonfiller_vocab_pool_swapto_total"] = int(bw_add.pool_swapto.sum())
    R[f"test_{cc}"]["nonfiller_vocab_acc_swapto_total"] = int(bw_add.acc_swapto.sum())
    L("non-filler vocab (S1_df>=200): pool adds", int(bw_add.pool_add.sum()), "accepted", int(bw_add.acc_add.sum()),
      "| pool swap-to", int(bw_add.pool_swapto.sum()), "accepted", int(bw_add.acc_swapto.sum()))
    # examples of pool pure-adds of business words (France) with acceptance flag
    if cc == "France":
        s1n = T["s1"].set_index("id"); rn = pd.concat([T["s2"], T["s3"]]).set_index("id")
        ex = d[(d.kind == "ADD1") & d.b.isin(["club", "comite", "amicale", "centre", "amis", "ecole", "federation", "association"])].sample(frac=1, random_state=0).head(12)
        R["France_pool_business_add_examples"] = [f"acc={r.acc} p={r.p} | {s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in ex.itertuples()]
        ex = d[(d.kind == "SWAP1") & (d.bcls == "vocab_nonfiller") & ~d.acc].sample(frac=1, random_state=0).head(10)
        R["France_pool_vocab_swap_REJECTED_examples"] = [f"p={r.p} | {s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in ex.itertuples()]
        for e in R["France_pool_business_add_examples"] + R["France_pool_vocab_swap_REJECTED_examples"]:
            L("   ", e)
    del d

# ---------------- C labelled train pools
p = os.path.join(OUT, "WD_2_pairs_TRAIN.pkl")
if os.path.exists(p):
    d = pd.read_pickle(p)
    d["acc"] = d.p >= NEW_TH
    out = {}
    for cc in ("US", "India"):
        x = d[d.country == cc].copy()
        tt = token_table(x.assign(acc=x.y == 1), DFtr[cc])          # acc column reused as 'is true' here (label, for the TRUE-rate columns)
        filler = set(tt[(tt.pool_add >= 20) & (tt.pool_add_ratio >= 0.02) & (tt.S1_df >= 1)].tok) | set(tt[(tt.pool_add >= 20) & (tt.S1_df == 0)].tok)
        tt = tt.rename(columns={"acc_add": "true_add", "acc_swapto": "true_swapto", "acc_rate_add": "p_match_add", "acc_rate_swapto": "p_match_swapto"})
        tt.sort_values("pool_swapto", ascending=False).to_csv(os.path.join(OUT, f"WD_4_traintokens_{cc}.csv"), index=False)
        x["bcls"] = classify_b(x, DFtr[cc], filler)
        n_s1 = sum(meta.get(f"TRAIN_{k}", {}).get("n_s1_by_country", {}).get(cc, 0) for k in ("V1", "T2X", "E014", "T0"))
        g = x[x.kind.isin(["SWAP1", "ADD1"])].groupby(["kind", "bcls"]).agg(n=("y", "size"), n_pos=("y", "sum"), p_match=("y", "mean"))
        g["per_10k_S1"] = 1e4 * g.n / max(1, n_s1)
        L(f"==== TRAIN pools {cc}: same-address rows {len(x):,}, S1 {n_s1:,}, filler tokens {len(filler)}"); L(g.round(4).to_string())
        o = dict(n_s1=int(n_s1), by_class=g.round(5).reset_index().values.tolist(),
                 fillers_top=tt[tt.tok.isin(filler)].sort_values("pool_add", ascending=False).head(25)[["tok", "S1_df", "pool_add", "pool_add_ratio", "p_match_add"]].round(4).values.tolist())
        # organisation words that are pure-appended in US/India accepted test pairs: are they true?
        org = ["board", "trust", "commission", "federation", "society", "council", "association", "foundation", "center", "services", "partners"]
        o["org_word_adds"] = tt[tt.tok.isin(org)][["tok", "S1_df", "pool_add", "true_add", "p_match_add", "pool_swapto", "true_swapto", "p_match_swapto"]].round(4).values.tolist()
        L(tt[tt.tok.isin(org)].round(4).to_string())
        # vocab (non-filler) swaps: top pairs with labels
        vs = x[(x.kind == "SWAP1") & (x.bcls == "vocab_nonfiller")]
        o["vocab_swap_top_pairs"] = vs.groupby(["a", "b"]).y.agg(["size", "sum"]).sort_values("size", ascending=False).head(20).reset_index().values.tolist()
        # V1: model behaviour inside the class
        v1 = x[x.pool == "V1"]
        m = np.load(PATHS["v1_meta"]); pv = np.load(PATHS["v1_p_new"]); yv = m["y"]; cv = m["country"][m["s1idx"]]
        tot_fp = int(((pv >= NEW_TH) & (yv == 0) & (cv == cc)).sum()); tot_fn = int(((pv < NEW_TH) & (yv == 1) & (cv == cc)).sum())
        cls = {}
        for (k, b), gg in v1[v1.kind.isin(["SWAP1", "ADD1"])].groupby(["kind", "bcls"]):
            cls[f"{k}|{b}"] = dict(n=int(len(gg)), n_pos=int(gg.y.sum()), n_acc=int(gg.acc.sum()), FP=int((gg.acc & (gg.y == 0)).sum()),
                                   FN=int((~gg.acc & (gg.y == 1)).sum()))
        o["V1_model_in_class"] = cls; o["V1_total_FP"] = tot_fp; o["V1_total_FN"] = tot_fn
        L("V1 model per class:", json.dumps(cls), "| total V1 FP", tot_fp, "FN", tot_fn)
        # label-free token score as a same-address feature: AUC of 'record-only token is non-filler vocab' among same-address SWAP1+ADD1
        z = x[x.kind.isin(["SWAP1", "ADD1"]) & (x.bcls != "rare_or_garble")]
        if len(z) and z.y.nunique() == 2:
            from sklearn.metrics import roc_auc_score
            o["auc_flag_vocab_nonfiller"] = round(float(roc_auc_score(z.y, (z.bcls != "vocab_nonfiller").astype(int))), 4)
            o["n_auc_rows"] = int(len(z))
        # examples of TRUE same-address vocab swaps (the generator does produce content-word swaps on true records?)
        tr = load("train", verbose=False)
        s1n = tr["s1"].set_index("id"); rn = pd.concat([tr["s2"], tr["s3"]]).set_index("id")
        exs = vs[vs.y == 1].sample(frac=1, random_state=0).head(12)
        o["true_vocab_swap_examples"] = [f"{s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in exs.itertuples()]
        exs = vs[vs.y == 0].sample(frac=1, random_state=0).head(8)
        o["false_vocab_swap_examples"] = [f"{s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {rn.at[r.rec, 'name']} | {rn.at[r.rec, 'addr']}" for r in exs.itertuples()]
        for e in o["true_vocab_swap_examples"][:8] + ["--- false"] + o["false_vocab_swap_examples"][:6]:
            L("   ", e)
        out[cc] = o
    R["train_pools"] = out
json.dump(R, open(os.path.join(OUT, "WD_4_results.json"), "w"), indent=1, default=str)
L("wrote WD_4_results.json")
