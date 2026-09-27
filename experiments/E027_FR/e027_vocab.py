"""E027_FR step 2: token document frequency and most frequent substituted / added / dropped tokens in the France pool
(street-equal pairs, all deltas pooled; no S006 match information used). Used to write the pre-registered lists."""
import pickle, collections, numpy as np, json
E = pickle.load(open("ent.pkl", "rb")); z = np.load("pairs_streq.npz")
df = collections.Counter(t for n in E["names"] for t in set(n))
e, ta, tb, d = z["e"], z["ta"], z["tb"], z["d"]
m0 = d == 0
print("edit type counts (all d):", collections.Counter(e.tolist())); print("edit type counts d==0:", collections.Counter(e[m0].tolist()))
sub = e == 1
pc = collections.Counter(tuple(sorted(p)) for p in zip(ta[sub & m0], tb[sub & m0]))
print("top 120 substituted pairs at d==0 (unordered):")
for p, c in pc.most_common(120): print(c, p, df[p[0]], df[p[1]])
tc = collections.Counter(np.concatenate([ta[sub & m0], tb[sub & m0]]).tolist())
print("top 80 tokens in substitutions d==0:", [(t, c, df[t]) for t, c in tc.most_common(80)])
ad = collections.Counter(np.concatenate([tb[(e == 2) & m0], ta[(e == 3) & m0]]).tolist())
print("top 60 add/drop tokens d==0:", [(t, c, df[t]) for t, c in ad.most_common(60)])
print("DF top 150:", df.most_common(150))
json.dump(dict(df_top=df.most_common(3000)), open("name_token_df.json", "w"))
