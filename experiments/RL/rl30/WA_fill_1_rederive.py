"""RL-30 adversarial verifier WA (lens: statistical artifact / support) -- 'generator filler tokens (France)' claim, PART 1 (READ-ONLY).
(Files of this verifier carry the prefix WA_fill_ ; another verifier uses WA_1_*.)
Independent re-derivation of investigator A's numbers with my own code:
 - France pseudo-positive (PP) pool size: S005 France kept_final, p>=0.99, n_claims==1, same house number + street tokens.
 - per-token record-only add counts per 1k PP pairs, add_LR (= add/pair / S1 share), drop rates of legal forms.
 - the same French adders in US / India PP (all kept_final pairs, no sampling).
 - HI/LO band shares; how filler-bearing street pairs split between HI (>=0.99) and LO (0.78-0.99) = the selection effect.
 - IDF as the PIPELINE computes it (recon05 normalize: NFC, accents KEPT, pooled over all test S1) vs A's accent-folded IDF.
Tokenisation here = rl30_lib.toks (accent-folded [a-z0-9]+), typo-paired set differences.
Output: WA_fill_1_results.json
"""
import os, sys, json, time, re, math, unicodedata
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
CLAIM_ADD_PER_1K = dict(com=58.8, sas=10.0, sa=9.9, services=9.8, sarl=9.3, cie=8.6, sci=8.4, eurl=7.5, dba=6.6, sasu=5.1, formerly=3.6,
                        fils=3.1, et=3.1, associes=2.9, aka=1.6, compagnie=1.5, frs=1.1)
CLAIM_ADDLR = dict(services=2.07, cie=0.39, fils=0.16, associes=29.8)
CLAIM_IDF = dict(cie=5.72, fils=5.83, compagnie=6.20, developpement=7.66, et=9.82, associes=11.13, frs=12.01, dba=11.92, sarl=3.16, sas=3.50,
                 services=4.16)
FR_ONLY = ["cie", "fils", "associes", "compagnie", "developpement", "et", "frs"]
WATCH = sorted(set(CLAIM_ADD_PER_1K) | set(FR_ONLY) | {"service", "center", "partners", "freres"})
LEGAL_FR = ["sas", "sa", "sarl", "sci", "eurl", "sasu"]


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


def skey(addr):
    n, s, _ = street_parts(addr)
    return f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else None


def pipe_norm(text):   # src/recon05_baseline_scorer.normalize (copied; accents kept)
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE); t = re.sub(r"-+", " ", t)
    t = re.sub(r"\bnull\b", " ", t); return re.sub(r"\s+", " ", t).strip()


R = {}
D = load("test", verbose=False)
s1 = D["s1"]; REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
S1 = s1.set_index("id")
NC = s1.country.value_counts().to_dict()
dfC = {C: Counter(t for n in s1.name.values[s1.country.values == C] for t in set(toks(n))) for C in ("France", "US", "India")}
L("loaded")

# ---- IDF: A's folded pooled vs pipeline (accents kept)
Npool = len(s1)
dff = Counter(); dfp = Counter()
for n in s1.name.values:
    dff.update(set(toks(n))); dfp.update(set(pipe_norm(n).split()))
idf = lambda df, w: math.log((Npool - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5) + 1.0)
rec_fr = REC[REC.country == "France"]
recp = Counter()
for n in rec_fr.name.values:
    recp.update(set(pipe_norm(n).split()))
variants = {}
for w in ["associes", "associés", "developpement", "développement", "cie", "fils", "compagnie", "et", "frs", "dba", "sarl", "sas", "services",
          "freres", "frères"]:
    variants[w] = dict(idf_pipeline=round(idf(dfp, w), 3), df_S1_pipeline=dfp.get(w, 0), n_France_records_pipeline=recp.get(w, 0),
                       idf_folded=round(idf(dff, fold(w)), 3))
R["idf_pipeline_vs_folded"] = variants
R["idf_claim_vs_folded"] = {w: dict(claim=v, folded=round(idf(dff, w), 3)) for w, v in CLAIM_IDF.items()}
L("idf", json.dumps(variants, ensure_ascii=False)[:900])

# ---- per-country pass over accepted pairs
for C in ("France", "US", "India"):
    a = accepted(f"S005_{C}")
    a = a[a.kept_final].reset_index(drop=True)
    s1n = S1.name.reindex(a.s1.values).values; s1a = S1.addr.reindex(a.s1.values).values
    rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
    p = a.p.values; nc = a.n_claims.values
    band = np.where(p >= 0.99, "HI", "LO")
    pp_mask = (p >= 0.99) & (nc == 1)
    st = np.zeros(len(a), bool)
    for k in range(len(a)):
        if isinstance(ra[k], str):
            k1 = skey(s1a[k]); st[k] = k1 is not None and k1 == skey(ra[k])
    out = dict(n_kept_final=int(len(a)), n_HI=int((band == "HI").sum()), n_LO=int((band == "LO").sum()),
               LO_share=round(float((band == "LO").mean()), 4), n_PP=int((pp_mask & st).sum()),
               n_LO_street=int(((band == "LO") & st).sum()), LO_street_share_of_kept=round(float(((band == "LO") & st).mean()), 4))
    L(C, json.dumps(out))
    raw_add = Counter(); pur_add = Counter(); pur_drop = Counter(); s1has = Counter(); npp = 0
    band_rec_only = {b: Counter() for b in ("HI", "LO")}; band_n = Counter()
    band_pure_single_add = {b: Counter() for b in ("HI", "LO")}
    for k in np.where(st)[0]:
        if not isinstance(rn[k], str):
            continue
        A_, B_ = frozenset(toks(s1n[k])), frozenset(toks(rn[k]))
        pa, pb = pure(A_, B_)
        b = band[k]; band_n[b] += 1
        for t in pb:
            if t in WATCH: band_rec_only[b][t] += 1
        if not pa and len(pb) == 1:
            t = next(iter(pb))
            if t in WATCH: band_pure_single_add[b][t] += 1
        if pp_mask[k]:
            npp += 1
            for t in A_: s1has[t] += 1
            for t in B_ - A_: raw_add[t] += 1
            for t in pb: pur_add[t] += 1
            for t in pa: pur_drop[t] += 1
    tok = {}
    for t in WATCH:
        share = dfC[C].get(t, 0) / NC[C]
        tok[t] = dict(raw_add_per_1k=round(1000 * raw_add[t] / npp, 2), pure_add_per_1k=round(1000 * pur_add[t] / npp, 2), n_pure_add=pur_add[t],
                      add_LR=round((pur_add[t] / npp) / max(share, 1 / NC[C]), 3), s1_df=dfC[C].get(t, 0),
                      drop_rate=round(pur_drop[t] / s1has[t], 3) if s1has[t] else None,
                      HI_street_rec_only_per_1k=round(1000 * band_rec_only["HI"][t] / max(band_n["HI"], 1), 2),
                      LO_street_rec_only_per_1k=round(1000 * band_rec_only["LO"][t] / max(band_n["LO"], 1), 2),
                      n_HI_street_rec_only=band_rec_only["HI"][t], n_LO_street_rec_only=band_rec_only["LO"][t],
                      frac_of_street_rec_only_in_LO=round(band_rec_only["LO"][t] / max(band_rec_only["LO"][t] + band_rec_only["HI"][t], 1), 3),
                      pure_single_add_HI=band_pure_single_add["HI"][t], pure_single_add_LO=band_pure_single_add["LO"][t])
    out["n_PP_counted"] = npp; out["street_pairs_by_band"] = dict(band_n)
    out["LO_share_among_street_pairs"] = round(band_n["LO"] / max(band_n["LO"] + band_n["HI"], 1), 4)
    out["tokens"] = tok
    out["legal_drop_rates"] = {t: tok[t]["drop_rate"] for t in LEGAL_FR}
    if C == "France":
        out["claim_add_per_1k_vs_rederived_pure"] = {t: (v, tok[t]["pure_add_per_1k"]) for t, v in CLAIM_ADD_PER_1K.items()}
        out["claim_addLR_vs_rederived"] = {t: (v, tok[t]["add_LR"]) for t, v in CLAIM_ADDLR.items()}
    R[C] = out
    L(C, "done", json.dumps({t: tok[t] for t in FR_ONLY + ["services", "com"]})[:1500])
json.dump(R, open(os.path.join(OUT, "WA_fill_1_results.json"), "w"), indent=1, ensure_ascii=False, default=str)
L("written")
