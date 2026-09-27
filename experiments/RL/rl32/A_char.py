"""RL-32 / A step 3: characterise France crowd strata (k_core>=31, 6-30) vs k_core=1 and vs US/India, label-free.
(a) contested records (accepted by 2+ S1, S006 th .72): per S1 rate, winner-loser p margin, address class of winner vs loser.
(b) rejected band pairs p6 in [0.2,.72): share whose record is accepted (post max-claimer) by ANOTHER S1; address/name class."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(RL))
sys.path.insert(0, os.path.join(ROOT, "experiments", "GAP")); sys.path.insert(0, os.path.join(RL, "rl31"))
import gap_lib as G
from gap03_band_classes import classify
te = G.records("test"); rng = np.random.default_rng(0)
S = pd.read_pickle(os.path.join(HERE, "A_s1table.pkl")).set_index("id")
out = {}
def crowd(k): return np.where(k >= 31, "31+", np.where(k >= 6, "6-30", np.where(k >= 2, "2-5", "1")))
for c in ("US", "India", "France"):
    z = np.load(os.path.join(RL, "rl31", "test_scores", f"{c}.npz"), allow_pickle=True)
    p = z["p6"].astype(np.float64); s1 = z["s1"]; cand = z["cand"]
    cr = crowd(S.loc[s1, "k_core"].values)
    acc = p >= .72
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": p, "cr": cr, "acc": acc, "rrl": z["rrl"], "rk": z["rk"]})
    A = d[d.acc].copy(); A["nclaim"] = A.groupby("cand").cand.transform("size")
    A["pmax"] = A.groupby("cand").p.transform("max"); A["win"] = A.p == A.pmax
    win_s1 = A[A.win].drop_duplicates("cand").set_index("cand").s1
    nS1 = pd.Series(cr).groupby(S.loc[z["u"], "k_core"].pipe(lambda k: crowd(k.values))).size() if False else pd.Series(crowd(S.loc[z["u"], "k_core"].values)).value_counts()
    res = {}
    for g in ("1", "2-5", "6-30", "31+"):
        Ag = A[A.cr == g]; con = Ag[Ag.nclaim >= 2]
        r = dict(n_S1=int(nS1.get(g, 0)), contested_claims_perS1=round(len(con) / max(1, nS1.get(g, 0)), 4),
                 losers_perS1=round(float((~con.win).sum()) / max(1, nS1.get(g, 0)), 4))
        if len(con):
            L = con[~con.win].merge(A[A.win][["cand", "p"]].drop_duplicates("cand").rename(columns={"p": "pw"}), on="cand")
            r["loser_margin_q"] = [round(float(x), 4) for x in np.percentile(L.pw - L.p, [10, 25, 50, 75, 90])]
            smp = con.sample(min(len(con), 20000), random_state=0)
            cl = [classify(te[a][0], te[a][1], te[b][0], te[b][1]) for a, b in zip(smp.s1, smp.cand)]
            smp = smp.assign(ac=[x[0] for x in cl], nc=[x[1] for x in cl])
            r["winner_addr_class"] = smp[smp.win].ac.value_counts(normalize=True).round(3).to_dict()
            r["loser_addr_class"] = smp[~smp.win].ac.value_counts(normalize=True).round(3).to_dict()
            r["loser_name_class"] = smp[~smp.win].nc.value_counts(normalize=True).head(5).round(3).to_dict()
        # (b) rejected band
        B = d[(d.cr == g) & (d.p >= 0.2) & (~d.acc)].copy()
        ws = B.cand.map(win_s1); B["claimed_else"] = ws.notna() & (ws.values != B.s1.values)
        r["band_pairs_perS1"] = round(len(B) / max(1, nS1.get(g, 0)), 4)
        r["band_mass_perS1"] = round(float(B.p.sum()) / max(1, nS1.get(g, 0)), 4)
        r["band_claimed_elsewhere_share"] = round(float(B.claimed_else.mean()), 4) if len(B) else None
        r["band_unclaimed_mass_perS1"] = round(float(B.p[~B.claimed_else].sum()) / max(1, nS1.get(g, 0)), 4)
        if len(B):
            smp = B.sample(min(len(B), 15000), random_state=0)
            cl = [classify(te[a][0], te[a][1], te[b][0], te[b][1]) for a, b in zip(smp.s1, smp.cand)]
            smp = smp.assign(ac=[x[0] for x in cl], nc=[x[1] for x in cl])
            r["band_unclaimed_class"] = (smp[~smp.claimed_else].ac + "|" + smp[~smp.claimed_else].nc).value_counts(normalize=True).head(6).round(3).to_dict()
            r["band_claimed_class"] = (smp[smp.claimed_else].ac + "|" + smp[smp.claimed_else].nc).value_counts(normalize=True).head(4).round(3).to_dict()
        # accepted-after-MC classes
        W = Ag[Ag.win].sample(min(int(Ag.win.sum()), 15000), random_state=0)
        cl = [classify(te[a][0], te[a][1], te[b][0], te[b][1]) for a, b in zip(W.s1, W.cand)]
        r["accepted_mc_addr_class"] = pd.Series([x[0] for x in cl]).value_counts(normalize=True).round(3).to_dict()
        r["accepted_mc_p_lt_.95_perS1"] = round(float(((Ag.win) & (Ag.p < .95)).sum()) / max(1, nS1.get(g, 0)), 4)
        res[g] = r; print(c, g, json.dumps(r), flush=True)
    out[c] = res
json.dump(out, open(os.path.join(HERE, "A_char.json"), "w"), indent=1)
