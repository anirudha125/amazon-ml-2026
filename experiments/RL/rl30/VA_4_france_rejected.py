"""RL-30 verifier VA, step 4 (France TEST, label-free). The only France pairs a filler feature could flip are the same-street
filler-only pool pairs that S005 REJECTS (VA_2: F_only_st 1,818 of 37,441; F_add_st 11,540 of 71,001). Why are they rejected?
For each flagged same-street pool pair (accepted-kept vs rejected) measure, label-free:
  sib_same_core : # OTHER France S1 at the same street key whose filler-stripped name key equals this S1's (ambiguous sibling)
  sib_rec_exact : # OTHER France S1 at the same street key whose raw name-token set equals the RECORD's (record = another S1's name)
  sib_rec_core  : # OTHER France S1 at the same street key whose stripped key equals the RECORD's stripped key
  s1_kept       : # records S005 finally assigned to this S1
  legal_swap    : S1 and record carry DIFFERENT legal forms (both non-empty)
Stripped key = name tokens minus A's France filler / droppable sets minus LEGAL / HONOR.  READ-ONLY; writes VA_4_france_rejected.json."""
import os, sys, json, time, re, glob
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s if isinstance(s, str) else ""); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


R = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
R = R[R.occ >= 300]
FILL = set(R.index[R.add_LR >= 0.05]); DROP = set(R.index[R.drop_rate.fillna(0) >= 0.20])
STRIP = FILL | DROP | LEGAL | HONOR
D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].reset_index(drop=True)
rec = pd.concat([D["s2"], D["s3"]]); rec = rec[rec.country == "France"].reset_index(drop=True)
del D


def skey(a):
    n_, st, _ = street_parts(a if isinstance(a, str) else "")
    return None if (n_ is None or not st) else n_ + "|" + " ".join(sorted(st))


s1_sk = s1.addr.map(skey).values; rec_sk_s = rec.addr.map(skey)
codes, uniq = pd.factorize(pd.concat([pd.Series(s1_sk), rec_sk_s], ignore_index=True), use_na_sentinel=True)
s1_code = codes[:len(s1)]; rec_code = codes[len(s1):]
s1_tok = [ntoks(x) for x in s1.name.values]
s1_core = [frozenset(t for t in T if t not in STRIP) for T in s1_tok]
by_st_core = Counter(); by_st_tok = Counter()
for i in range(len(s1)):
    if s1_code[i] >= 0:
        by_st_core[(s1_code[i], s1_core[i])] += 1; by_st_tok[(s1_code[i], s1_tok[i])] += 1
print("indexes built", f"{time.time() - T0:.0f}s", flush=True)
s1_ix = pd.Index(s1.id.values); rec_ix = pd.Index(rec.id.values)
a = accepted("S005_France")
ACC = dict(zip(a.s1.values.astype(object) + "|" + a.rec.values.astype(object), zip(a.p.values, a.kept_final.values)))
kept = a[a.kept_final]; s1_kept = kept.s1.value_counts().to_dict(); rec_kept = set(kept.rec.values)
rcache = {}
rows = []
for path in sorted(glob.glob(PATHS["test_chunks"].format(country="France"))):
    z = np.load(path); cs1 = z["s1"]; cc = z["cand"]
    i1 = s1_ix.get_indexer(cs1); i2 = rec_ix.get_indexer(cc)
    ok = (i1 >= 0) & (i2 >= 0)
    c1 = np.where(ok, s1_code[np.clip(i1, 0, None)], -1); c2 = np.where(ok, rec_code[np.clip(i2, 0, None)], -2)
    for k in np.flatnonzero((c1 >= 0) & (c1 == c2)):
        j1, j2 = i1[k], i2[k]
        A = s1_tok[j1]; B = rcache.get(j2)
        if B is None: B = rcache[j2] = ntoks(rec.name.values[j2])
        ob = B - A
        if not ob: continue
        pa, pb = set(A - B), set(ob)
        if pa and pb:
            for _, x, zz in sorted(((LV.normalized_similarity(x, zz), x, zz) for x in pa for zz in pb if typo(x, zz)), reverse=True):
                if x in pa and zz in pb: pa.discard(x); pb.discard(zz)
        if not pb or not all(t in FILL for t in pb): continue
        rest = [t for t in pa if t not in STRIP]
        kind = "F_only_st" if not rest else ("F_sub1_st" if len(rest) == 1 else "F_add_other_st")
        hit = ACC.get(f"{cs1[k]}|{cc[k]}")
        state = ("kept" if hit[1] else "lost_mc") if hit is not None else ("rej_rec_kept_other" if cc[k] in rec_kept else "rej_rec_unassigned")
        sc = c1[k]; rcore = frozenset(t for t in B if t not in STRIP)
        la = {t for t in A if t in LEGAL}; lb = {t for t in B if t in LEGAL}
        rows.append((kind, state, by_st_core[(sc, s1_core[j1])] - 1, by_st_tok[(sc, B)] - (1 if B == A else 0),
                     by_st_core[(sc, rcore)] - (1 if rcore == s1_core[j1] else 0), s1_kept.get(cs1[k], 0),
                     bool(la and lb and la != lb), None if hit is None else float(hit[0]), s1.name.values[j1], rec.name.values[j2]))
Df = pd.DataFrame(rows, columns=["kind", "state", "sib_same_core", "sib_rec_exact", "sib_rec_core", "s1_kept", "legal_swap", "p", "s1", "rec"])
print("rows", len(Df), f"{time.time() - T0:.0f}s", flush=True)
out = {}
for (kd, stt), g in Df.groupby(["kind", "state"]):
    out[f"{kd}|{stt}"] = dict(n=int(len(g)), sib_same_core_any=round(float((g.sib_same_core > 0).mean()), 4),
                              sib_rec_exact_any=round(float((g.sib_rec_exact > 0).mean()), 4),
                              sib_rec_core_any=round(float((g.sib_rec_core > 0).mean()), 4),
                              legal_swap=round(float(g.legal_swap.mean()), 4), s1_kept0=round(float((g.s1_kept == 0).mean()), 4),
                              s1_kept_median=float(g.s1_kept.median()),
                              no_sibling_and_no_legal_swap=int(((g.sib_same_core == 0) & (g.sib_rec_exact == 0) & (g.sib_rec_core == 0) & ~g.legal_swap).sum()))
    print(kd, stt, out[f"{kd}|{stt}"], flush=True)
rej = Df[(Df.kind == "F_only_st") & Df.state.str.startswith("rej")]
clean = rej[(rej.sib_same_core == 0) & (rej.sib_rec_exact == 0) & (rej.sib_rec_core == 0) & ~rej.legal_swap]
out["F_only_rejected_clean_examples"] = clean.sample(min(25, len(clean)), random_state=0)[["s1", "rec", "s1_kept"]].to_dict("records")
out["F_only_rejected_with_sibling_examples"] = rej[rej.sib_same_core > 0].sample(min(15, int((rej.sib_same_core > 0).sum())), random_state=0)[["s1", "rec", "sib_same_core", "s1_kept"]].to_dict("records")
out["F_only_rejected_clean_s1_kept_dist"] = clean.s1_kept.clip(upper=5).value_counts().sort_index().to_dict()
out["F_only_rejected_rec_only_tokens"] = Counter(t for r_, s_ in zip(rej.rec, rej.s1) for t in (ntoks(r_) - ntoks(s_))).most_common(20)
out["F_only_kept_rec_only_tokens"] = Counter(t for r_, s_ in zip(Df[(Df.kind == "F_only_st") & (Df.state == "kept")].rec, Df[(Df.kind == "F_only_st") & (Df.state == "kept")].s1)
                                             for t in (ntoks(r_) - ntoks(s_))).most_common(20)
out["secs"] = round(time.time() - T0, 1)
Df.to_pickle(os.path.join(OUT, "VA_4_france_flagged_pairs.pkl"))
json.dump(out, open(os.path.join(OUT, "VA_4_france_rejected.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in out.items() if "examples" not in k}, indent=0, ensure_ascii=False, default=str)[:5000])
print("done", f"{time.time() - T0:.0f}s")
