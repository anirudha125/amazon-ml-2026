"""RL-30 investigator A, PART 2f (READ-ONLY, reads A_* csv only). Structure of France content-for-content substitutions:
are they concentrated on specific directed pairs (synonym-like noise operator) or spread like independent draws (decoy-like)?
lift(a->b) = n(a->b) / (N * P_out(a) * P_in(b)), marginals from the same band's content->content substitutions.
Reference: co-located S1-S1 one-token swaps (Part 1, different businesses by construction).
Output: A_p2f_summary.json"""
import os, json, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
content = set(roles.index[(roles.add_LR < 0.05) & (roles.occ >= 300) & (~roles.legal.astype(bool))])
S = {}
def lift_table(df, a="s1_tok", b="rec_tok"):
    d = df[df[a].isin(content) & df[b].isin(content) & (df[a] != df[b])].copy()
    N = d.n.sum(); po = d.groupby(a).n.sum() / N; pi = d.groupby(b).n.sum() / N
    d["exp"] = [N * po[x] * pi[y] for x, y in zip(d[a], d[b])]
    d["lift"] = d.n / d.exp
    return d.sort_values("n", ascending=False), N
for band in ("HI", "LO"):
    d, N = lift_table(pd.read_csv(os.path.join(OUT, f"A_p2b_subst_France_{band}.csv"), keep_default_na=False))
    top = d.head(40)
    S[band] = dict(N_content_subs_in_top3000=int(N), frac_on_pairs_lift_gt3=round(float(d[d.lift > 3].n.sum() / N), 4),
                   weighted_mean_log2_lift=round(float((d.n * np.log2(d.lift)).sum() / N), 3),
                   top=[(x, y, int(n), round(float(l), 2)) for x, y, n, l in top[["s1_tok", "rec_tok", "n", "lift"]].values[:25]],
                   top_by_lift_n_ge_15=[(x, y, int(n), round(float(l), 2)) for x, y, n, l in d[d.n >= 15].sort_values("lift", ascending=False)[["s1_tok", "rec_tok", "n", "lift"]].values[:15]])
sw = pd.read_csv(os.path.join(OUT, "A_p1_swaps_France.csv"), keep_default_na=False)
sw2 = pd.concat([sw.rename(columns={"a": "s1_tok", "b": "rec_tok"}), sw.rename(columns={"b": "s1_tok", "a": "rec_tok"})])
d, N = lift_table(sw2)
S["COLOC_S1_swaps"] = dict(N=int(N), frac_on_pairs_lift_gt3=round(float(d[d.lift > 3].n.sum() / N), 4),
                           weighted_mean_log2_lift=round(float((d.n * np.log2(d.lift)).sum() / N), 3))
# asymmetry: for the top HI pairs, n(a->b) vs n(b->a)
hi = pd.read_csv(os.path.join(OUT, "A_p2b_subst_France_HI.csv"), keep_default_na=False).set_index(["s1_tok", "rec_tok"]).n
S["HI_direction"] = [(x, y, int(hi.get((x, y), 0)), int(hi.get((y, x), 0))) for x, y, _, _ in S["HI"]["top"][:15]]
json.dump(S, open(os.path.join(OUT, "A_p2f_summary.json"), "w"), indent=1)
print(json.dumps(S, indent=0)[:6000])
