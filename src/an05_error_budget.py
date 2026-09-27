"""
AN05 -- error budget + FP/FN taxonomy for an E024 arm on V1 (20,000 S1) [diagnostic; labels used for analysis only].
Loss decomposition: macro F0.5 gain if exactly one error class were fixed (others unchanged):
  retrieval FN (GT not in pool), scorer FN (in-pool GT rejected), scorer FP (non-GT accepted; split singleton / non-singleton).
Taxonomy per error pair: channel (lexical-only / dense-only / both), candidate name Indic, candidate address missing,
exact name, S1 name frequency, FP owned by another S1 (conflict-resolvable), probability band.
Usage: python src/an05_error_budget.py <arm> [seed] [pooltag]
"""
import os, sys, json, pickle, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e021_foundation as F
import e023_stage2 as S2
from translit import has_indic
from recon05_baseline_scorer import normalize

arm = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42; pooltag = sys.argv[3] if len(sys.argv) > 3 else "a50n10d10a"
E24 = S2.E24
R = pickle.load(open(os.path.join(E24, f"arm_{arm}.pkl"), "rb")); th = R["seeds"][seed]["th_oof"]
V = S2.load_set("V1", pooltag); p = np.load(os.path.join(E24, f"p_{arm}_V1_s{seed}.npy"))
s1 = pickle.load(open(os.path.join(F.OUT, "s1.pkl"), "rb"))
ids = V["s1_ids"]; n = len(ids); si = V["s1idx"]; y = V["y"]; acc = p >= th
gt_n = V["n_gt"].astype(float)
tp = np.bincount(si, weights=acc & (y == 1), minlength=n); fp = np.bincount(si, weights=acc & (y == 0), minlength=n)
inpool = np.bincount(si, weights=(y == 1), minlength=n); fn_s = inpool - tp; fn_r = gt_n - inpool


def f05(tp, fp, g):
    npred = tp + fp
    return np.where(g == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * g + npred, 1e-9), 0.0))


base = f05(tp, fp, gt_n)
sing = gt_n == 0
out = dict(arm=arm, seed=seed, th=th, macro=base.mean() * 100)
out["fix_retrieval_FN"] = (f05(tp + fn_r, fp, gt_n).mean() - base.mean()) * 100      # pool-missed GT -> accepted
out["fix_scorer_FN"] = (f05(tp + fn_s, fp, gt_n).mean() - base.mean()) * 100
out["fix_scorer_FP"] = (f05(tp, 0 * fp, gt_n).mean() - base.mean()) * 100
out["fix_FP_singletons_only"] = (f05(tp, np.where(sing, 0, fp), gt_n).mean() - base.mean()) * 100
out["oracle_on_pool"] = f05(inpool, 0 * fp, gt_n).mean() * 100
out["counts"] = dict(retrieval_FN=int(fn_r.sum()), scorer_FN=int(fn_s.sum()), FP=int(fp.sum()), TP=int(tp.sum()),
                     singleton_FP_entities=int(((fp > 0) & sing).sum()), n_singletons=int(sing.sum()))

# taxonomy
docs = {}
for c in ["US", "India"]:
    for src in "23":
        i2, n2, a2, _ = F.read_table("train", src, c)
        docs.update(zip(i2, zip(n2, a2)))
owner = {}
for s, r in s1.items():
    for g in r["gt"]:
        owner[g] = s
lex = (V["rank_addr"] < 999) | (V["rank_name"] < 999); den = V["rank_dense"] < 999
def cat(i):
    s = ids[si[i]]; c = V["cand"][i]; nm, ad = docs[c]; r = s1[s]
    ch = "both" if lex[i] and den[i] else ("lex_only" if lex[i] else "dense_only")
    return dict(channel=ch, indic_name=has_indic(nm), cand_addr_missing=not ad.strip(),
                exact_name=normalize(nm) == r["name_r"], country=r["country"],
                band=("<th+.1" if p[i] < th + 0.1 else ("<.95" if p[i] < 0.95 else ">=.95")) if acc[i] else ("<.3" if p[i] < 0.3 else ("<th-.1" if p[i] < th - 0.1 else ">=th-.1")))
tax = {}
for name, mask in [("FP", acc & (y == 0)), ("scorer_FN", ~acc & (y == 1)), ("TP", acc & (y == 1))]:
    idx = np.flatnonzero(mask); C = collections.defaultdict(collections.Counter)
    for i in idx:
        for k, v in cat(i).items():
            C[k][str(v)] += 1
        if name == "FP":
            C["owned_by_other_S1"][str(V["cand"][i] in owner)] += 1
            C["singleton_S1"][str(bool(sing[si[i]]))] += 1
    tax[name] = {k: dict(v) for k, v in C.items()}; tax[name]["n"] = len(idx)
# retrieval misses
miss = collections.Counter(); inpool_set = collections.defaultdict(set)
for i in np.flatnonzero(y == 1):
    inpool_set[ids[si[i]]].add(V["cand"][i])
examples = []
for s in ids:
    for g in s1[s]["gt"] - inpool_set[s]:
        nm, ad = docs[g]; miss["indic_name"] += has_indic(nm); miss["addr_missing"] += not ad.strip(); miss["n"] += 1
        if len(examples) < 12:
            examples.append(dict(s1=(s1[s]["raw_name"], s1[s]["raw_addr"]), gt=(nm, ad)))
tax["retrieval_FN"] = dict(miss); tax["retrieval_FN_examples"] = examples
# FP / FN examples
ex = {"FP": [], "scorer_FN": []}
for name, mask in [("FP", acc & (y == 0)), ("scorer_FN", ~acc & (y == 1))]:
    for i in np.random.default_rng(0).choice(np.flatnonzero(mask), 15, replace=False):
        s = ids[si[i]]; ex[name].append(dict(p=round(float(p[i]), 3), s1=(s1[s]["raw_name"], s1[s]["raw_addr"]), cand=docs[V["cand"][i]],
                                             ch=cat(i)["channel"], singleton=bool(sing[si[i]])))
out["taxonomy"] = tax; out["examples"] = ex
json.dump(out, open(os.path.join(E24, f"an05_{arm}_s{seed}.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("taxonomy", "examples")}, indent=1, default=str))
for k in ["FP", "scorer_FN", "retrieval_FN"]:
    print(k, json.dumps(tax[k], default=str))
