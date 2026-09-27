"""RL-30 adversarial verifier WA, step 3 (READ-ONLY). Expected LB value of removing the France flagged pairs (WA_1 re-derived set),
as a function of the false-match share f, for the full set and for subsets (without 'france' substitutions, without morphological
variants, LO band only). Same assumptions as A_part2h (other accepted pairs of the S1 are correct, no misses; ESTIMATE).
Output: WA_3_results.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

F = pd.read_pickle(os.path.join(OUT, "WA_1_flagged_France.pkl"))
a = accepted("S005_France"); a = a[a.kept_final]
tot = a.groupby("s1").size()
NTOT = 1_732_544; NFR = 259_452


def f05(P, Rr):
    return 0.0 if P + Rr == 0 else 1.25 * P * Rr / (0.25 * P + Rr)


def pref(x, y):
    n = 0
    for c1, c2 in zip(x, y):
        if c1 != c2: break
        n += 1
    return n


F["france"] = (F.y == "france") | (F.x == "france")
F["morph"] = [(pref(x, y) >= 4) or (x in y) or (y in x) for x, y in zip(F.x, F.y)]
S = {}
subsets = {"all": np.ones(len(F), bool), "LO": (F.p < 0.99).values, "HI": (F.p >= 0.99).values,
           "no_france_no_morph": (~F.france & ~F.morph).values, "LO_no_france_no_morph": ((F.p < 0.99) & ~F.france & ~F.morph).values,
           "france_only": F.france.values, "morph_only": F.morph.values}
for nm, msk in subsets.items():
    g = F[msk]; cnt = g.groupby("s1").size()
    gain = loss = 0.0
    for s, kf in cnt.items():
        m = int(tot[s]) - int(kf)
        gain += 1.0 - (f05(m / (m + kf), 1.0) if m > 0 else 0.0)
        loss += 1.0 - (f05(1.0, m / (m + kf)) if m > 0 else 0.0)
    ev = lambda f: 100 * (f * gain - (1 - f) * loss) / NTOT
    S[nm] = dict(pairs=int(len(g)), S1=int(len(cnt)), mean_p=round(float(g.p.mean()), 4), model_implied_f=round(float((1 - g.p).mean()), 4),
                 dLB_allFP_pp=round(ev(1.0), 4), dLB_allTP_pp=round(ev(0.0), 4), breakeven_f=round(loss / (gain + loss), 3),
                 dLB_at_model_implied_pp=round(ev(float((1 - g.p).mean())), 4), dLB_at_f_0035_pp=round(ev(0.035), 4),
                 dLB_at_f_022_pp=round(ev(0.22), 4), dFrance_allFP_pp=round(100 * gain / NFR, 3), dFrance_allTP_pp=round(-100 * loss / NFR, 3))
    print(nm, S[nm], flush=True)
json.dump(S, open(os.path.join(OUT, "WA_3_results.json"), "w"), indent=1)
