"""VD (adversarial verifier of D's 'France net record deficit'): the France-vs-US post-max-claimer count gap vs the
max-claimer drop count, S004 vs S005 (LB France unchanged), and concrete dropped-claim examples.
Writes VD_mc.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, akey, core_tokens
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test", verbose=False)
s1 = T["s1"].set_index("id"); rec = pd.concat([T["s2"], T["s3"]]).set_index("id")
fr_ids = T["s1"].id[T["s1"].country == "France"].values
for tag in ("S004_France", "S005_France"):
    a = accepted(tag)
    L(tag, "columns", list(a.columns), len(a))
    if "kept_final" not in a.columns:
        # reconstruct max-claimer: keep the highest-p claimant of each record
        a = a.sort_values(["rec", "p"], ascending=[True, False]); a["kept_final"] = ~a.duplicated("rec")
    pre = a.groupby("s1").size().reindex(fr_ids, fill_value=0); post = a[a.kept_final].groupby("s1").size().reindex(fr_ids, fill_value=0)
    nc = a.groupby("rec").size()
    R[tag] = dict(pre_per_s1=float(pre.mean()), post_per_s1=float(post.mean()), drop_per_s1=float((pre - post).mean()),
                  multi_claimed_recs=int((nc > 1).sum()), unique_recs=int(len(nc)), s1_with_drop=float(((pre - post) > 0).mean()),
                  p1_post=float((post == 1).mean()), p0_post=float((post == 0).mean()))
    L(tag, R[tag])
# S005 dropped claims: relationship between the losing S1 and the winning S1
a = accepted("S005_France")
w = a[a.kept_final][["rec", "s1", "p"]].rename(columns={"s1": "win", "p": "pw"})
d = a[~a.kept_final].merge(w, on="rec", how="left")
d["lose_name"] = d.s1.map(s1.name); d["win_name"] = d.win.map(s1.name)
d["lose_addr"] = d.s1.map(s1.addr); d["win_addr"] = d.win.map(s1.addr)
d["same_akey"] = d.lose_addr.map(akey) == d.win_addr.map(akey)
d["same_name"] = d.lose_name.str.lower() == d.win_name.str.lower()
lc = d.lose_name.map(core_tokens); wc = d.win_name.map(core_tokens)
d["core_share"] = [len(x & y) > 0 for x, y in zip(lc, wc)]
d["p_gap"] = d.pw - d.p
R["drop_relation"] = dict(n=len(d), same_akey=float(d.same_akey.mean()), same_name=float(d.same_name.mean()),
                          core_share=float(d.core_share.mean()), median_p_lose=float(d.p.median()), median_p_win=float(d.pw.median()),
                          frac_gap_lt_002=float((d.p_gap < 0.02).mean()))
L("drop relation", R["drop_relation"])
rng = np.random.default_rng(0)
ex = d.iloc[rng.choice(len(d), 12, replace=False)]
exl = []
for _, r in ex.iterrows():
    exl.append(dict(rec=r.rec, rec_name=rec.name.get(r.rec), rec_addr=rec.addr.get(r.rec), loser=r.s1, loser_name=r.lose_name, loser_addr=r.lose_addr,
                    p_lose=round(float(r.p), 4), winner=r.win, winner_name=r.win_name, winner_addr=r.win_addr, p_win=round(float(r.pw), 4)))
    L(exl[-1])
R["examples"] = exl
json.dump(R, open(os.path.join(OUT, "VD_mc.json"), "w"), indent=1, default=float)
L("wrote VD_mc.json")
