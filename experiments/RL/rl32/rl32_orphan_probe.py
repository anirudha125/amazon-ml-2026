"""RL-32 quick probe: do UNCLAIMED test records form sibling clusters (orphan entities), unlike train decoys?
Key = (country, folded name core w/o legal tokens, first house number, longest address word>=4). Address-less records skipped.
Train: linked vs unlinked (GT). Test: claimed (accepted by S005 model p5>=.78) vs unclaimed (max p5 over S1 < 0.05 or absent)."""
import re, unicodedata, pickle, numpy as np, pandas as pd, json
LEGAL = set("llc inc corp corporation ltd limited pvt private llp co company lp pc pllc plc sarl sas sasu eurl sa sci snc".split())
def fold(s):
    s = unicodedata.normalize("NFKD", (s or "").lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
def key(n, a, c):
    fa = fold(a)
    if not fa: return None
    m = re.search(r"\d+", fa); num = m.group(0).lstrip("0") if m else ""
    w = [t for t in fa.split() if len(t) >= 4 and not t.isdigit()]
    core = " ".join(sorted(t for t in fold(n).split() if t not in LEGAL))
    if not core: return None
    return f"{c}|{core}|{num}|{max(w, key=len) if w else ''}"
def keys(df):
    return pd.Series([key(n, a, c) for n, a, c in zip(df.name.values, df.addr.values, df.country.values)], index=df.id.values)
out = {}
tr = pickle.load(open("../cache/train.pkl", "rb"))
rec = pd.concat([tr["s2"], tr["s3"]]); K = keys(rec)
linked = set(tr["gt"].rec.values)
df = pd.DataFrame({"k": K.values, "lk": [i in linked for i in K.index], "c": rec.country.values}).dropna()
for c in ("US", "India"):
    d = df[df.c == c]
    for nm, sub in (("linked", d[d.lk]), ("unlinked", d[~d.lk])):
        vc = sub.k.value_counts(); tw = sub.k.map(vc) >= 2
        out[f"train_{c}_{nm}"] = dict(n=int(len(sub)), share_with_twin_in_same_class=round(float(tw.mean()), 4))
        print(c, "train", nm, out[f"train_{c}_{nm}"])
    # cross: unlinked records whose key twin is a linked record
    lk = set(d[d.lk].k); u = d[~d.lk]
    out[f"train_{c}_unlinked_twin_of_linked"] = round(float(u.k.isin(lk).mean()), 4); print(c, "train unlinked with linked twin", out[f"train_{c}_unlinked_twin_of_linked"])
te = pickle.load(open("../cache/test.pkl", "rb"))
rec = pd.concat([te["s2"], te["s3"]]); K = keys(rec)
for c in ("US", "India", "France"):
    z = np.load(f"../rl31/test_scores/{c}.npz", allow_pickle=True)
    mp = pd.Series(z["p5"]).groupby(z["cand"]).max()
    claimed = set(mp.index[mp.values >= 0.78]); weak = set(mp.index[mp.values >= 0.05])
    m = rec.country.values == c
    d = pd.DataFrame({"k": K.values[m], "id": K.index[m]}).dropna()
    d["cl"] = d.id.isin(claimed); d["un"] = ~d.id.isin(weak)
    for nm, sub in (("claimed", d[d.cl]), ("unclaimed", d[d.un])):
        vc = sub.k.value_counts(); tw = sub.k.map(vc) >= 2
        out[f"test_{c}_{nm}"] = dict(n=int(len(sub)), share_with_twin_in_same_class=round(float(tw.mean()), 4),
                                     n_groups_ge2=int((vc >= 2).sum()), size_hist={int(k): int(v) for k, v in vc.value_counts().sort_index().items() if k <= 8})
        print(c, "test", nm, {k: v for k, v in out[f"test_{c}_{nm}"].items() if k != "size_hist"}, out[f"test_{c}_{nm}"]["size_hist"])
    ck = set(d[d.cl].k); u = d[d.un]
    out[f"test_{c}_unclaimed_twin_of_claimed"] = round(float(u.k.isin(ck).mean()), 4); print(c, "test unclaimed with claimed twin", out[f"test_{c}_unclaimed_twin_of_claimed"])
json.dump(out, open("rl32_orphan_probe.json", "w"), indent=1)
