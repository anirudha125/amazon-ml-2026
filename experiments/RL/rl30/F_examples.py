"""RL-30 Part 7 (investigator F) follow-up: concrete examples + self-reference share for pseudo-label statistics. READ-ONLY.
Output: experiments/RL/rl30/F_examples.json
"""
import sys, os, json, collections, itertools, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

R = {}
TE = load("test", verbose=False)
fs1 = TE["s1"][TE["s1"].country == "France"].reset_index(drop=True)

# (1) self-reference: France S1 token document frequency -> share of occurrences on rare tokens
df = collections.Counter()
for nm in fs1.name.values:
    df.update(set(toks(nm)))
occ = collections.Counter()
for nm in fs1.name.values:
    occ.update(toks(nm))
tot = sum(occ.values())
R["france_token_df_occurrence_share"] = {f"df<={k}": round(sum(n for t, n in occ.items() if df[t] <= k) / tot, 4) for k in (1, 2, 5, 20, 100)}
R["france_s1_with_a_df1_token"] = round(sum(1 for nm in fs1.name.values if any(df[t] == 1 for t in toks(nm))) / len(fs1), 4)

# (2) near-duplicate co-located France S1 pairs: examples + do the two siblings claim the same records in S005 (pre max-claimer)?
A = accepted("S005_France")
claims = A.groupby("s1").rec.apply(set).to_dict()
kept = A[A.kept_final].groupby("s1").rec.apply(set).to_dict()
ak = [akey(a) for a in fs1.addr.values]
grp = collections.defaultdict(list)
for i, a in enumerate(ak):
    if a:
        grp[a].append(i)
pairs = []
for a, idx in grp.items():
    if len(idx) < 2 or len(idx) > 200:
        continue
    ts = [set(toks(fs1.name.values[i])) for i in idx]
    for x, y in itertools.combinations(range(len(idx)), 2):
        Aa, Bb = ts[x], ts[y]
        if (Aa & Bb) and Aa != Bb and len(Aa - Bb) <= 1 and len(Bb - Aa) <= 1:
            pairs.append((idx[x], idx[y]))
st = collections.Counter(); ex = []
diff_tok = collections.Counter()
for i, j in pairs:
    a, b = fs1.id.values[i], fs1.id.values[j]
    ca, cb = claims.get(a, set()), claims.get(b, set())
    ka, kb = kept.get(a, set()), kept.get(b, set())
    st["pairs"] += 1
    st["both_claim_something"] += bool(ca) and bool(cb)
    st["share_any_claimed_record"] += bool(ca & cb)
    st["both_nonempty_final"] += bool(ka) and bool(kb)
    st["one_empty_final"] += bool(ka) != bool(kb)
    st["both_empty_final"] += (not ka) and (not kb)
    Ta, Tb = set(toks(fs1.name.values[i])), set(toks(fs1.name.values[j]))
    for t in (Ta ^ Tb):
        diff_tok[t] += 1
    if len(ex) < 12:
        ex.append(dict(a=[a, fs1.name.values[i]], b=[b, fs1.name.values[j]], addr=fs1.addr.values[i],
                       claims_a=len(ca), claims_b=len(cb), shared_claims=len(ca & cb), final_a=len(ka), final_b=len(kb)))
R["france_neardup_coloc"] = dict(stats=dict(st), top_differing_tokens=diff_tok.most_common(15), examples=ex)
del A

# (3) department-vs-region examples: S2/S3 records whose admin unit is a department while S1 uses the region
ex = []
for src in ("s2", "s3"):
    d = TE[src][TE[src].country == "France"]
    for nm, ad in zip(d.name.values[:200000], d.addr.values[:200000]):
        comps = [c.strip() for c in fold(ad).split(",")]
        if any(c in ("nord", "gironde", "loire-atlantique", "pas-de-calais") for c in comps):
            ex.append([src, nm, ad])
            if len(ex) % 3 == 0:
                break
R["dept_examples"] = ex[:6]
R["s1_region_examples"] = [[n, a] for n, a in zip(fs1.name.values[:5], fs1.addr.values[:5])]

# (4) V1 same-address one-token-substitution pairs (label-backed): examples of positives and negatives
TR = load("train", verbose=False)
m = np.load(PATHS["v1_meta"], allow_pickle=True)
s1n = pd.Series(TR["s1"].name.values, index=TR["s1"].id.values); s1a = pd.Series(TR["s1"].addr.values, index=TR["s1"].id.values)
recn = pd.concat([pd.Series(TR[s].name.values, index=TR[s].id.values) for s in ("s2", "s3")])
reca = pd.concat([pd.Series(TR[s].addr.values, index=TR[s].id.values) for s in ("s2", "s3")])
sid = m["s1_ids"][m["s1idx"]]
N1 = s1n.reindex(sid).values; A1 = s1a.reindex(sid).values; CN = recn.reindex(m["cand"]).values; CA = reca.reindex(m["cand"]).values
pos, neg = [], []
for i in range(len(sid)):
    if not (isinstance(CA[i], str) and isinstance(CN[i], str)):
        continue
    if akey(CA[i]) != akey(A1[i]) or not akey(A1[i]):
        continue
    X, Y = core_tokens(N1[i]), core_tokens(CN[i])
    if (X & Y) and len(X - Y) == 1 and len(Y - X) == 1:
        rec = [N1[i], CN[i], sorted(X - Y), sorted(Y - X), A1[i]]
        (pos if m["y"][i] else neg).append(rec)
R["V1_same_addr_sub1_examples"] = dict(n_pos=len(pos), n_neg=len(neg), pos=pos[:8], neg=neg[:8])
json.dump(R, open(os.path.join(OUT, "F_examples.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str)[:12000])
