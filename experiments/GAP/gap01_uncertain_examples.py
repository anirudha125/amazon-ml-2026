"""GAP-01: side-by-side anatomy of the uncertain band, France test (S006 probs) vs V1 (RRL probs, labelled)."""
import numpy as np, sys
import gap_lib as G
rng = np.random.default_rng(7)
lo, hi = 0.25, 0.75
# ---------- France (test, no labels) ----------
F = G.france_scores("France"); te = G.records("test")
p = F["p6"]; band = (p >= lo) & (p <= hi)
print(f"France kept pairs {len(p):,}; in [{lo},{hi}] {band.sum():,}; per S1 {band.sum()/len(F['u']):.4f}")
# competitor: best other S1 claiming same record
order = np.argsort(-p); best = {}
for i in order:
    c = F["cand"][i]
    best.setdefault(c, []).append(i)
idx = rng.choice(np.flatnonzero(band), 40, replace=False)
for i in idx:
    s, c = F["s1"][i], F["cand"][i]
    comp = [j for j in best[c] if j != i][:1]
    sn, sa, _ = te[s]; cn, ca, _ = te[c]
    line = f"p={p[i]:.2f} rrl={F['rrl'][i]:+.1f} rk={F['rk'][i]}\n   S1 : {sn} | {sa}\n   REC: {cn} | {ca}"
    if comp:
        j = comp[0]; on, oa, _ = te[F['s1'][j]]
        line += f"\n   RIVAL p={p[j]:.2f}: {on} | {oa}"
    print(line)
# ---------- V1 (labelled) ----------
V = L = None
import rl31_lib as L
V = L.load_v1(); tr = G.records("train")
pv = V["p_RRL"]; bv = (pv >= lo) & (pv <= hi)
print(f"\nV1 pairs {len(pv):,}; in band {bv.sum():,}; per S1 {bv.sum()/len(V['s1_ids']):.4f}; band precision {V['y'][bv].mean():.3f}")
for c in ("US", "India"):
    m = bv & (V["country"][V["s1idx"]] == c)
    print(f"  {c}: in band {m.sum():,} per S1 {m.sum()/(V['country']==c).sum():.4f} prec {V['y'][m].mean():.3f}")
idx = rng.choice(np.flatnonzero(bv), 30, replace=False)
for i in idx:
    s = V["s1_ids"][V["s1idx"][i]]; c = V["cand"][i]
    sn, sa, cc = tr[s]; cn, ca, _ = tr[c]
    print(f"y={V['y'][i]} p={pv[i]:.2f} [{cc}]\n   S1 : {sn} | {sa}\n   REC: {cn} | {ca}")
