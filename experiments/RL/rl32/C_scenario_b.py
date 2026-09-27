"""RL-32 C: France S006 veto vs same-band rr++ vs threshold raise under uniform error-odds multipliers k (V1+T cell x band error rates).
Reads C_analyze.json; writes C_scenario_b.json."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); A = json.load(open(os.path.join(H, 'C_analyze.json')))
CELLS = ["rr++", "rrL+ rrUb-", "rrL- rrUb+", "rr--", "not top10"]; B = [".99-1", ".95-.99", ".90-.95", "th-.90"]
comp = A['part2']['S006']['France']['_cell_band_per_s1']; err = {}
for nm in CELLS:
    for bn in B:
        n = sum(A['part1'][f'{s}_S006']['cells'][nm]['by_band'][bn][0] for s in ('V1', 'T')); fp = sum(A['part1'][f'{s}_S006']['cells'][nm]['by_band'][bn][1] for s in ('V1', 'T'))
        err[(nm, bn)] = (fp + .5) / (n + 1)
def ev(k, vc, vb):
    e = {x: (k * v / (1 - v)) / (1 + k * v / (1 - v)) for x, v in err.items()}
    vol = sum(comp[a][b] for a in vc for b in vb); fp = sum(comp[a][b] * e[(a, b)] for a in vc for b in vb)
    return dict(France_fp_total=round(float(sum(comp[a][b] * e[(a, b)] for (a, b) in e)), 4), vol=round(vol, 4), fp=round(float(fp), 4),
                fp_rate=round(float(fp / vol), 3), dF_pp=round(float(22.1 * fp - 9.2 * (vol - fp)), 3))
out = {f"k={k:.2f}": {"veto any rr<=0 & p<.95": ev(k, CELLS[1:4], B[2:]), "rr++ & p<.95": ev(k, ["rr++"], B[2:]), "threshold .95": ev(k, CELLS, B[2:])}
       for k in (1.0, float(np.exp(.52)), 6.77, 7.54, 20.6)}
json.dump(out, open(os.path.join(H, 'C_scenario_b.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
