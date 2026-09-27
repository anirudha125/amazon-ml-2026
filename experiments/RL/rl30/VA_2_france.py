"""RL-30 adversarial verifier VA, step 2: France TEST (label-free). Where do A's 'filler-only' France pairs sit relative to
S005's decisions?  A feature that raises p on filler-noise pairs can change a decision only for pairs that are (a) in the
candidate pool with p < 0.78 (possible FN recovery), or (b) accepted but lost the max-claimer (kept_final False).
Pairs already accepted and kept (the 0.78-0.99 band A highlights) cannot change the submission.
Filler / droppable sets = A's France role-A thresholds (A_roles_France.csv: occ >= 300 and add_LR >= 0.05 -> filler;
drop_rate >= 0.20 -> droppable).  Flags are identical to VA_1 (same-street variants only, since the pool scan filters on
street key first).  READ-ONLY; writes VA_2_france.json."""
import os, sys, json, time, re, glob
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


R = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
R = R[R.occ >= 300]
FILL = set(R.index[R.add_LR >= 0.05]); DROP = set(R.index[R.drop_rate.fillna(0) >= 0.20])
print("France fillers", len(FILL), sorted(FILL), "droppable", len(DROP), sorted(DROP), flush=True)

D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].reset_index(drop=True)
rec = pd.concat([D["s2"], D["s3"]]); rec = rec[rec.country == "France"].reset_index(drop=True)
del D


def skey(a):
    n_, st, _ = street_parts(a if isinstance(a, str) else "")
    return None if (n_ is None or not st) else n_ + "|" + " ".join(sorted(st))


s1_sk = s1.addr.map(skey); rec_sk = rec.addr.map(skey)
codes, uniq = pd.factorize(pd.concat([s1_sk, rec_sk], ignore_index=True), use_na_sentinel=True)
s1_code = codes[:len(s1)]; rec_code = codes[len(s1):]
print("street keys", len(uniq), "S1 with key", float((s1_code >= 0).mean()), "rec with key", float((rec_code >= 0).mean()), f"{time.time() - T0:.0f}s", flush=True)
s1_ix = pd.Index(s1.id.values); rec_ix = pd.Index(rec.id.values)
s1_name = s1.name.values; rec_name = rec.name.values; s1_addr = s1.addr.values; rec_addr = rec.addr.values

a = accepted("S005_France")
akey_ = a.s1.values.astype(object) + "|" + a.rec.values.astype(object)
ACC = dict(zip(akey_, zip(a.p.values, a.kept_final.values, a.n_claims.values)))
rec_kept = set(a.rec.values[a.kept_final.values])          # records assigned to some S1 in the final S005 France output
rec_claimed = set(a.rec.values)

cache = {}


def nt(s):
    v = cache.get(s)
    if v is None:
        v = cache[s] = ntoks(s if isinstance(s, str) else "")
    return v


cnt = Counter(); ex = {}
rng = np.random.default_rng(0)
tot_pairs = tot_st = 0
paths = sorted(glob.glob(PATHS["test_chunks"].format(country="France")))
for ci, path in enumerate(paths):
    z = np.load(path); cs1 = z["s1"]; cc = z["cand"]
    i1 = s1_ix.get_indexer(cs1); i2 = rec_ix.get_indexer(cc)
    ok = (i1 >= 0) & (i2 >= 0)
    c1 = np.where(ok, s1_code[np.clip(i1, 0, None)], -1); c2 = np.where(ok, rec_code[np.clip(i2, 0, None)], -2)
    st = np.flatnonzero((c1 >= 0) & (c1 == c2))
    tot_pairs += len(cs1); tot_st += len(st)
    for k in st:
        A, B = nt(s1_name[i1[k]]), nt(rec_name[i2[k]])
        oa, ob = A - B, B - A
        if not ob:
            continue
        pa, pb = set(oa), set(ob)
        if pa and pb:
            for _, x, zz in sorted(((LV.normalized_similarity(x, zz), x, zz) for x in pa for zz in pb if typo(x, zz)), reverse=True):
                if x in pa and zz in pb: pa.discard(x); pb.discard(zz)
        if not pb or not all(t in FILL for t in pb):
            continue
        rest = [t for t in pa if t not in FILL and t not in DROP and t not in LEGAL and t not in HONOR]
        flags = ["F_add_st"] + (["F_only_st"] if not rest else (["F_sub1_st"] if len(rest) == 1 else []))
        key = f"{cs1[k]}|{cc[k]}"
        hit = ACC.get(key)
        if hit is None:
            state = "rejected_rec_kept_by_other_S1" if cc[k] in rec_kept else ("rejected_rec_claimed_but_lost" if cc[k] in rec_claimed else "rejected_rec_unclaimed")
        else:
            pv, kept, ncl = hit
            state = ("acc_kept_p>=0.99" if pv >= 0.99 else "acc_kept_p0.78-0.99") if kept else "acc_lost_maxclaim"
        for f in flags:
            cnt[f"{f}|{state}"] += 1; cnt[f"{f}|n"] += 1
            exl = ex.setdefault(f"{f}|{state}", [])
            if len(exl) < 10 and rng.random() < 0.02:
                exl.append(dict(s1=str(s1_name[i1[k]]), rec=str(rec_name[i2[k]]), s1_addr=str(s1_addr[i1[k]]), rec_addr=str(rec_addr[i2[k]]),
                                p=None if hit is None else round(float(hit[0]), 4)))
    if ci % 20 == 0:
        print(ci, len(paths), tot_pairs, tot_st, dict(cnt.most_common(6)), f"{time.time() - T0:.0f}s", flush=True)

out = dict(pool_pairs=tot_pairs, same_street_pool_pairs=tot_st, fillers=sorted(FILL), droppable=sorted(DROP), counts=dict(sorted(cnt.items())))
for f in ("F_add_st", "F_only_st", "F_sub1_st"):
    nn = cnt[f"{f}|n"]
    out[f"{f}_shares"] = {k.split("|")[1]: round(v / max(nn, 1), 4) for k, v in cnt.items() if k.startswith(f + "|") and not k.endswith("|n")}
out["examples"] = ex
out["secs"] = round(time.time() - T0, 1)
json.dump(out, open(os.path.join(OUT, "VA_2_france.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
print("done", f"{time.time() - T0:.0f}s")
