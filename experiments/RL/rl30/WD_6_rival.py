"""WD part 6: does the India labelled analog (same-address swap to a 'distinguishing' token, P(match)=0) transfer to France?
For every same-address swap-to-distinguishing pair (WD_5 class definition, re-implemented here), is there ANOTHER S1 of the same
split/country whose core name tokens equal the record's core tokens (a rival owner)? anywhere / at the same address.
Acceptance (test) or P(match) (train) by rival status; France acceptance by target token.
Writes rl30/WD_6_results.json"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, akey, NEW_TH
from WD_common import s1_df, core, addr_key

L = lambda *a: print(*a, flush=True)
R = {}


def dis_tokens(d, df):
    add = collections.Counter(d.b[d.kind == "ADD1"]); sw = collections.Counter(d.b[d.kind == "SWAP1"])
    out = set()
    for t, n in sw.items():
        a = add.get(t, 0)
        if n >= 10 and a / (a + n) < 0.01 and len(t) >= 4 and df.get(t, 0) >= 20:
            out.add(t)
    return out


def rival_index(s1):
    """core-token-set -> list of (s1 id, akey, addr_key)"""
    idx = collections.defaultdict(list)
    for i, n, a in zip(s1.id.values, s1.name.values, s1.addr.values):
        idx[core(n)].append((i, akey(a), addr_key(a)))
    return idx


def rival_flags(sel, idx, recdf):
    any_, same = [], []
    for s, r in zip(sel.s1.values, sel.rec.values):
        c = core(recdf.at[r, "name"]); ra = recdf.at[r, "addr"]; rk, rkey = akey(ra), addr_key(ra)
        lst = [x for x in idx.get(c, []) if x[0] != s]
        any_.append(len(lst) > 0)
        same.append(any((x[1] == rk and rk) or (x[2] is not None and x[2] == rkey) for x in lst))
    return np.array(any_), np.array(same)


def summarize(sel, lab):
    g = sel.groupby(["rival_any", "rival_same_addr"])[lab].agg(["size", "mean"])
    return [[bool(k[0]), bool(k[1]), int(r["size"]), round(float(r["mean"]), 4)] for k, r in g.iterrows()]


T = load("test", verbose=False)
recdf = pd.concat([T["s2"], T["s3"]]).set_index("id")
for j, cc in (("FR", "France"), ("IN", "India"), ("US", "US")):
    d = pd.read_pickle(os.path.join(OUT, f"WD_2_pairs_{j}.pkl"))
    s1c = T["s1"][T["s1"].country == cc]
    df = s1_df(s1c.name.values)
    dis = dis_tokens(d, df)
    dfa = d.a.map(lambda x: df.get(x, 0))
    sel = d[(d.kind == "SWAP1") & d.b.isin(dis) & (dfa >= 20) & (d.a.str.len() >= 4)].copy()
    idx = rival_index(s1c)
    sel["rival_any"], sel["rival_same_addr"] = rival_flags(sel, idx, recdf)
    o = dict(n=int(len(sel)), acc_rate=round(float(sel.acc.mean()), 4), rival_any_share=round(float(sel.rival_any.mean()), 4),
             rival_same_addr_share=round(float(sel.rival_same_addr.mean()), 4), acc_by_rival=summarize(sel, "acc"))
    if cc == "France":
        bt = sel.groupby("b").acc.agg(["size", "mean"]).sort_values("size", ascending=False).head(25)
        o["acc_by_target_token"] = [[k, int(r["size"]), round(float(r["mean"]), 3)] for k, r in bt.iterrows()]
        fin = sel[sel.kept_final]
        o["final_n"] = int(len(fin)); o["final_n_s1"] = int(fin.s1.nunique())
        o["final_rival_any_share"] = round(float(fin.rival_any.mean()), 4); o["final_rival_same_addr_share"] = round(float(fin.rival_same_addr.mean()), 4)
        o["final_p_quantiles"] = fin.p.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict()
        s1n = T["s1"].set_index("id")
        ex = fin[~fin.rival_any].sample(frac=1, random_state=0).head(8)
        o["final_orphan_examples"] = [f"p={r.p:.3f} | {s1n.at[r.s1, 'name']} | {s1n.at[r.s1, 'addr']}  ->  {recdf.at[r.rec, 'name']} | {recdf.at[r.rec, 'addr']}" for r in ex.itertuples()]
    L(f"==== TEST {cc}:", json.dumps({k: v for k, v in o.items() if k not in ("final_orphan_examples",)}, default=str))
    for e in o.get("final_orphan_examples", []):
        L("   ", e)
    R[f"test_{cc}"] = o
    del d
del T, recdf

Tr = load("train", verbose=False)
recdf = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
d = pd.read_pickle(os.path.join(OUT, "WD_2_pairs_TRAIN.pkl"))
for cc in ("India", "US"):
    x = d[d.country == cc]
    s1c = Tr["s1"][Tr["s1"].country == cc]
    df = s1_df(s1c.name.values)
    dis = dis_tokens(x, df)
    dfa = x.a.map(lambda t: df.get(t, 0))
    sel = x[(x.kind == "SWAP1") & x.b.isin(dis) & (dfa >= 20) & (x.a.str.len() >= 4)].copy()
    if not len(sel):
        R[f"train_{cc}"] = dict(n=0); L(f"==== TRAIN {cc}: no rows"); continue
    idx = rival_index(s1c)
    sel["rival_any"], sel["rival_same_addr"] = rival_flags(sel, idx, recdf)
    # also: is the record TRUE for the rival? (gt link rival->rec)
    gt = set(zip(Tr["gt"].s1.values, Tr["gt"].rec.values))
    o = dict(n=int(len(sel)), p_match=round(float(sel.y.mean()), 4), rival_any_share=round(float(sel.rival_any.mean()), 4),
             rival_same_addr_share=round(float(sel.rival_same_addr.mean()), 4), p_match_by_rival=summarize(sel, "y"))
    owners = Tr["gt"][Tr["gt"].rec.isin(set(sel.rec))]
    o["rec_owned_by_some_train_S1"] = round(float(sel.rec.isin(set(owners.rec)).mean()), 4)
    L(f"==== TRAIN {cc}:", json.dumps(o))
    R[f"train_{cc}"] = o
json.dump(R, open(os.path.join(OUT, "WD_6_results.json"), "w"), indent=1, default=str)
L("wrote WD_6_results.json")
