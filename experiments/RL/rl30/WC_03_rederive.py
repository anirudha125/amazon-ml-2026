"""WC step 3 (adversarial verifier of investigator C, lens: statistical artefact / support).  READ-ONLY; writes only rl30/WC_*.
Independent re-derivation of the 'France KEY2 decoy' support counts from the raw test text + S005 accepted tables:
  full   = |addr_words(S1) & addr_words(rec)| / min(|.|,|.|)                       (re-implemented here, same definition as C)
  rsC    = C's typo-robust street similarity (street_sig + rmatch from C_street2.py, recomputed from scratch here)
  rsB    = an ALTERNATIVE street definition: rl30_lib.street_parts (first comma component with a digit) street tokens,
           rapidfuzz token_set_ratio of the joined tokens                              (sensitivity check of the parser)
  hnC/hnB house-number agreement under each parser
  KEY2x  = full >= 0.8 & rsx < 0.6
Also: KEY2 rate by name genericity (dupf_core = #other S1 of the country with the same core-token set), per-S1 exposure (t = other
kept records of that S1), ESTIMATED macro-F0.5 stakes with a break-even FP fraction.
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import street_sig, rmatch
from rapidfuzz import fuzz

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

def addr_ok(a):
    return bool(a) and a.strip().lower() not in ("null", "<null>")

def feats(sa, ra):
    u, inv = np.unique(np.concatenate([sa, ra]), return_inverse=True)
    W = [addr_words(a) if addr_ok(a) else frozenset() for a in u]
    SP = [street_parts(a) if addr_ok(a) else (None, frozenset(), frozenset()) for a in u]
    SJ = [" ".join(sorted(s[1])) for s in SP]
    SG = [street_sig(a) for a in u]
    n = len(sa); i1 = inv[:n]; i2 = inv[n:]
    full = np.full(n, np.nan, np.float32); rsC = np.full(n, np.nan, np.float32); hnC = np.full(n, -1, np.int8)
    rsB = np.full(n, np.nan, np.float32); hnB = np.full(n, -1, np.int8)
    for k in range(n):
        a, b = i1[k], i2[k]
        if W[a] and W[b]:
            full[k] = len(W[a] & W[b]) / min(len(W[a]), len(W[b]))
        r, _, h = rmatch(SG[a], SG[b]); rsC[k] = r; hnC[k] = h
        if SP[a][1] and SP[b][1]:
            rsB[k] = 1.0 if SP[a][1] & SP[b][1] == SP[a][1] or SP[a][1] & SP[b][1] == SP[b][1] else fuzz.token_set_ratio(SJ[a], SJ[b]) / 100.0
        if SP[a][0] is not None and SP[b][0] is not None:
            hnB[k] = 1 if SP[a][0] == SP[b][0] else 0
    return full, rsC, hnC, rsB, hnB

TD = load("test", verbose=False)
s1df = TD["s1"]; s1 = s1df.set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
core = pd.Series([" ".join(sorted(core_tokens(x))) for x in s1df.name.values], index=s1df.id.values)
fulln = pd.Series([" ".join(name_tokens(x)) for x in s1df.name.values], index=s1df.id.values)
dupf_core = (pd.DataFrame({"c": s1df.country.values, "k": core.values}).groupby(["c", "k"]).k.transform("size") - 1).values
dupf_full = (pd.DataFrame({"c": s1df.country.values, "k": fulln.values}).groupby(["c", "k"]).k.transform("size") - 1).values
DC = pd.Series(dupf_core, index=s1df.id.values); DF = pd.Series(dupf_full, index=s1df.id.values)
N_S1 = s1df.country.value_counts().to_dict(); N_ALL = len(s1df)
RES = dict(n_s1=N_S1)
BUCK = [(0, 0), (1, 2), (3, 9), (10, 10 ** 9)]
for c in ("France", "US", "India"):
    A = accepted(f"S005_{c}")
    sid = A.s1.values; cid = A.rec.values; p = A.p.values; kept = A.kept_final.values.astype(bool); ncl = A.n_claims.values
    sa = s1.addr.reindex(sid).fillna("").values.astype(object); ra = rec.addr.reindex(cid).fillna("").values.astype(object)
    full, rsC, hnC, rsB, hnB = feats(sa, ra)
    log(c, "features done", len(p))
    K2C = (full >= 0.8) & (rsC < 0.6); K2B = (full >= 0.8) & (rsB < 0.6)
    DIFFC = ~np.isnan(full) & (rsC < 0.6)
    dc = DC.reindex(sid).values; df = DF.reindex(sid).values
    R = dict(n_accepted=int(len(p)), n_kept=int(kept.sum()), n_s1_country=int(N_S1[c]))
    def blk(m):
        return dict(n=int(m.sum()), share_acc=round(float(m.mean()), 5), kept=int((m & kept).sum()), share_kept=round(float((m & kept).sum() / kept.sum()), 5),
                    s1_kept=int(len(np.unique(sid[m & kept]))), mean_p_kept=round(float(p[m & kept].mean()), 4) if (m & kept).any() else None,
                    multi_claim=round(float((ncl[m] > 1).mean()), 4) if m.any() else None)
    for nm, m in [("KEY2_C", K2C), ("KEY2_C&hn=eq", K2C & (hnC == 1)), ("KEY2_C&hn=neq", K2C & (hnC == 0)), ("KEY2_C&hn=miss", K2C & (hnC == -1)),
                  ("KEY2_B", K2B), ("KEY2_B&hnB=eq", K2B & (hnB == 1)), ("KEY2_C&KEY2_B", K2C & K2B), ("KEY2_C&~KEY2_B", K2C & ~K2B),
                  ("KEY2_C&rsB_match(>=0.8)", K2C & (rsB >= 0.8)), ("KEY2_C&rsB_undef", K2C & np.isnan(rsB)),
                  ("street_diff_C(any full)", DIFFC)]:
        R[nm] = blk(m)
    # genericity buckets
    G = {}
    for lo, hi in BUCK:
        b = (dc >= lo) & (dc <= hi)
        G[f"dupf_core[{lo},{hi if hi < 1e9 else 'inf'}]"] = dict(acc=int(b.sum()), kept=int((b & kept).sum()),
            KEY2_share_acc=round(float(K2C[b].mean()), 5) if b.any() else None, KEY2_share_kept=round(float(K2C[b & kept].mean()), 5) if (b & kept).any() else None,
            KEY2_hn_eq_share_kept=round(float((K2C & (hnC == 1))[b & kept].mean()), 5) if (b & kept).any() else None,
            KEY2_kept_n=int((K2C & b & kept).sum()))
    R["by_genericity"] = G
    # per-S1 exposure + ESTIMATED stakes (France and others)
    for flag_nm, flag in [("KEY2_C", K2C), ("KEY2_C&KEY2_B", K2C & K2B), ("street_diff_C", DIFFC)]:
        g = pd.DataFrame({"s1": sid[kept], "f": flag[kept]}).groupby("s1").agg(k=("f", "size"), fl=("f", "sum"))
        g = g[g.fl > 0]; k = g.k.values.astype(float); fl = g.fl.values.astype(float); t = k - fl
        gain = np.where(t > 0, 1 - 1.25 * t / (0.25 * t + k), 1.0)       # FP scenario: rest = GT exactly; t=0 -> S1 must be a singleton (F 0 -> 1)
        gain0 = np.where(t > 0, 1 - 1.25 * t / (0.25 * t + k), 0.0)      # FP scenario but t=0 S1 have unretrieved GT (F 0 -> 0)
        loss = 1 - np.where(t > 0, 1.25 * t / (0.25 * k + t), 0.0)        # TP scenario: flagged are GT
        Gs, G0, Ls = gain.sum(), gain0.sum(), loss.sum()
        R[f"stakes_{flag_nm}"] = dict(pairs=int(fl.sum()), s1=int(len(g)), s1_t0=int((t == 0).sum()), s1_share_country=round(len(g) / N_S1[c], 5),
            ESTIMATED_country_gain_if_all_FP=round(Gs / N_S1[c], 5), ESTIMATED_country_gain_if_all_FP_t0_not_singleton=round(G0 / N_S1[c], 5),
            ESTIMATED_country_loss_if_all_TP=round(Ls / N_S1[c], 5), ESTIMATED_LB_gain_if_all_FP=round(Gs / N_ALL, 6),
            ESTIMATED_LB_loss_if_all_TP=round(Ls / N_ALL, 6), ESTIMATED_breakeven_FP_fraction=round(Ls / (Gs + Ls), 3),
            ESTIMATED_breakeven_FP_fraction_t0_not_singleton=round(Ls / (G0 + Ls), 3))
    RES[c] = R
    log(c, json.dumps({k: v for k, v in R.items()}, default=str))
    if c == "France":
        np.savez_compressed(os.path.join(HERE, "WC_03_France_pairs.npz"), full=full, rsC=rsC, hnC=hnC, rsB=rsB, hnB=hnB, dupf_core=dc, dupf_full=df)
    else:
        np.savez_compressed(os.path.join(HERE, f"WC_03_{c}_pairs.npz"), full=full, rsC=rsC, hnC=hnC, rsB=rsB, hnB=hnB, dupf_core=dc, dupf_full=df)
json.dump(RES, open(os.path.join(HERE, "WC_03_rederive.json"), "w"), indent=1, default=str)
log("saved WC_03_rederive.json")
