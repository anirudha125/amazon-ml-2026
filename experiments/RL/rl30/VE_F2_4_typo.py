"""RL-30 / VE_F2 -- what are the F2_ge ties that survive the S005 max-claimer?  Label-free character check.
For each F2_ge-flagged accepted pair: record-only tokens u (R - A) are compared with this S1's missing tokens (A - R) and with
the best sibling's missing tokens (B - R) by max(Levenshtein normalized similarity, same on sorted letters).
  self_typo : best sim to this S1's token >= 0.6 and > best sim to the sibling's token  (record = misspelt copy of THIS S1)
  sib_typo  : the mirror image                                                          (record = misspelt copy of the SIBLING)
  neither   : pure drop / whole-word substitution
Applied to France test (kept / dropped by S005's max-claimer) and to the labelled V1 accepted flagged pairs (E_v1_flags.npz).
READ-ONLY.  Output: rl30/VE_F2_4_typo.json"""
import os, sys, json, time, pickle, collections
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as L
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from VE_F2_3_norm import ntoks

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def sim(a, b):
    return max(L.normalized_similarity(a, b), L.normalized_similarity("".join(sorted(a)), "".join(sorted(b))))


def classify(A, R, Bs):
    """A: this S1 token set, R: record token set, Bs: sibling token sets -> (class, best sibling jac)"""
    u = R - A
    jb = [jac(R, B) for B in Bs]; B = Bs[int(np.argmax(jb))]
    a, b = A - R, B - R
    sa = max((sim(x, t) for x in u for t in a), default=0.0)
    sb = max((sim(x, t) for x in u for t in b), default=0.0)
    if sa >= 0.6 and sa > sb:
        return "self_typo"
    if sb >= 0.6 and sb > sa:
        return "sib_typo"
    return "neither"


def main():
    t0 = time.time(); out = {}
    # ---------------- France test
    T = load("test", verbose=False)
    s1all = T["s1"]; S1 = s1all.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb")); SIB = RO["sib"]; need = set(SIB)
    a = accepted("S005_France").reset_index(drop=True)
    cnt = collections.Counter(); ex = collections.defaultdict(list)
    for i in np.flatnonzero(a.s1.isin(need).values):
        s = a.s1[i]; A = nset(S1.name.loc[s]); R = nset(recs.name.loc[a.rec[i]])
        jr = jac(R, A); js = max(jac(R, B) for B in SIB[s])
        if not (js > 0 and js >= jr):
            continue
        An, Rn = ntoks(S1.name.loc[s]), ntoks(recs.name.loc[a.rec[i]])
        kind = "gt" if js > jr else "tie"
        cl = classify(A, R, SIB[s])
        st = "kept" if a.kept_final[i] else "dropped"
        cnt[f"{st}|{kind}|{cl}"] += 1
        if st == "kept" and len(ex[f"{kind}|{cl}"]) < 6:
            ex[f"{kind}|{cl}"].append(f"{S1.name.loc[s]}  <>  {recs.name.loc[a.rec[i]]}  (p={a.p[i]:.4f})")
    agg = collections.Counter()
    for k, v in cnt.items():
        st, kind, cl = k.split("|"); agg[f"{st}|{cl}"] += v
    out["France"] = dict(detail=dict(sorted(cnt.items())), by_state=dict(sorted(agg.items())), kept_examples=dict(ex))
    log("France", out["France"]["detail"], out["France"]["by_state"])
    # ---------------- V1 labelled (accepted flagged pairs)
    D = load("train", verbose=False); S1t = D["s1"].set_index("id"); rt = pd.concat([D["s2"], D["s3"]]).set_index("id")
    ROt = pickle.load(open(os.path.join(HERE, "E_roles_train.pkl"), "rb"))
    M = np.load(PATHS["v1_meta"]); ps1 = M["s1_ids"][M["s1idx"]]; cand = M["cand"]; y = M["y"] == 1
    p = np.load(PATHS["v1_p_new"]); Fz = np.load(os.path.join(HERE, "E_v1_flags.npz")); f2 = Fz["f2_ge"]
    rows = []
    for i in np.flatnonzero(f2 & (p >= NEW_TH)):
        A = nset(S1t.name.loc[ps1[i]]); R = nset(rt.name.loc[cand[i]])
        rows.append(dict(s1=S1t.name.loc[ps1[i]], rec=rt.name.loc[cand[i]], y=bool(y[i]), cls=classify(A, R, ROt["sib"][ps1[i]]),
                         strict=bool(Fz["f2_gt"][i])))
    out["V1_accepted_flagged"] = rows
    for r in rows:
        log("V1", r)
    json.dump(out, open(os.path.join(HERE, "VE_F2_4_typo.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
