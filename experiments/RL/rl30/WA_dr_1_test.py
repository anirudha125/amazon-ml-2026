"""RL-30 adversarial verifier WA (lens: statistical artifact / support), claim 'S1_side_token_drop_rate', PART 1 (READ-ONLY).
Independent re-derivation (own code, does not read any A_* file) of investigator A's TEST pseudo-positive drop rates:
 PP = accepted('S005_<C>') kept_final, p>=0.99, n_claims==1, record house number + street tokens == S1's (street_parts).
Two tokenisations: TA = A's (fold + dotted-acronym collapse, token SET); TB = rl30_lib.toks set (no acronym collapse).
Typo pairing (Levenshtein dist<=1 or norm-sim>=0.75, greedy) removes typo'd tokens from 'pure' drops, as A did.
Extra checks (France, TB): per-occurrence rows -> split-half reliability by S1 hash, position/name-length adjustment,
abbreviation-explained drops (e.g. freres->frs), token-in-S1-address flag, band comparison (HI1 / HI multi-claim / LO),
cluster (per-S1) effective sample size.
Outputs: WA_dr_1_France_occ.pkl (per-occurrence rows), WA_dr_1_tok_{C}.csv (token tables), WA_dr_1_summary.json
"""
import os, sys, json, time, re, zlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
L = lambda *a: print(*a, f"{time.time() - T0:.0f}s", flush=True)
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ta_tokens(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


def pure(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, a, b in sorted(((LV.normalized_similarity(a, b), a, b) for a in pa for b in pb if typo(a, b)), reverse=True):
            if a in pa and b in pb:
                pa.discard(a); pb.discard(b)
    return pa, pb


def subseq(r, t):
    it = iter(t)
    return all(ch in it for ch in r)


def abbrev_of(t, pb):
    """dropped S1 token t explained by a record-only abbreviation r (same first letter, shorter, subsequence)"""
    return any(len(r) >= 2 and len(r) < len(t) and r[0] == t[0] and subseq(r, t) and not r.isdigit() for r in pb)


_sp = {}


def sp(addr):
    v = _sp.get(addr)
    if v is None:
        n, s, _ = street_parts(addr); v = (n, s); _sp[addr] = v
    return v


def same_street(a1, a2):
    n1, s1 = sp(a1); n2, s2 = sp(a2)
    return n1 is not None and n1 == n2 and bool(s1) and s1 == s2


D = load("test", verbose=False)
S1 = D["s1"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
L("loaded")
SUM = {}
for C in ("France", "US", "India"):
    a = accepted(f"S005_{C}")
    SUM[f"{C}_accepted_rows"] = int(len(a)); SUM[f"{C}_accepted_kept_final"] = int(a.kept_final.sum())
    a = a[a.kept_final]
    if C != "France":
        a = a.sample(n=min(len(a), 1_200_000), random_state=1)      # independent sample (A used random_state=0)
    s1ids = a.s1.values
    s1n = S1.name.reindex(s1ids).values; s1a = S1.addr.reindex(s1ids).values
    rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
    pv = a.p.values; nc = a.n_claims.values
    band = np.where(pv >= 0.99, np.where(nc == 1, 0, 1), 2)             # 0 = HI1 (PP band), 1 = HI multi-claim, 2 = LO (0.78-0.99)
    # counters: key (band, street) -> Counter
    cnt = defaultdict(Counter)
    pp_s1 = Counter(); n_street = Counter(); n_band = Counter(band.tolist())
    rows = []   # France only: per S1-token occurrence (TB tokenisation)
    for k in range(len(a)):
        b = int(band[k])
        if not isinstance(rn[k], str) or not isinstance(s1n[k], str):
            continue
        st = same_street(s1a[k], ra[k]) if isinstance(ra[k], str) and isinstance(s1a[k], str) else False
        n_street[(b, st)] += 1
        # --- TA (reproduction)
        A, B = ta_tokens(s1n[k]), ta_tokens(rn[k])
        pa, pb = pure(A, B)
        c = cnt[("TA", b, st)]
        for t in A:
            c["has|" + t] += 1
        for t in pa:
            c["drop|" + t] += 1
        # --- TB (own tokenisation)
        tl = toks(s1n[k]); A2 = frozenset(tl); B2 = frozenset(toks(rn[k]))
        pa2, pb2 = pure(A2, B2)
        c2 = cnt[("TB", b, st)]
        for t in A2:
            c2["has|" + t] += 1
        for t in pa2:
            c2["drop|" + t] += 1
        if b == 0 and st:
            pp_s1[s1ids[k]] += 1
        if C == "France":
            order = list(dict.fromkeys(tl)); nl = len(order)
            aw = addr_words(s1a[k]) if isinstance(s1a[k], str) else frozenset()
            h = zlib.crc32(s1ids[k].encode()) & 1
            for i, t in enumerate(order):
                d = t in pa2
                rows.append((t, i, nl, b, st, d, d and abbrev_of(t, pb2), t in aw, h, s1ids[k], len(pb2)))
        if k % 200000 == 0:
            L(C, k, len(a))
    SUM[f"{C}_n_band"] = {str(k): int(v) for k, v in n_band.items()}
    SUM[f"{C}_n_band_street"] = {f"{k[0]}_{k[1]}": int(v) for k, v in n_street.items()}
    SUM[f"{C}_PP_pairs"] = int(n_street[(0, True)])
    SUM[f"{C}_PP_distinct_S1"] = int(len(pp_s1))
    SUM[f"{C}_PP_pairs_per_S1_quantiles"] = {str(q): float(v) for q, v in pd.Series(list(pp_s1.values())).quantile([.5, .9, .99]).items()}
    # token tables
    out = []
    for (tk, b, st), c in cnt.items():
        for key, v in c.items():
            kind, t = key.split("|", 1)
            out.append((tk, b, st, kind, t, v))
    T = pd.DataFrame(out, columns=["tok_scheme", "band", "street", "kind", "tok", "n"])
    T = T.pivot_table(index=["tok_scheme", "band", "street", "tok"], columns="kind", values="n", fill_value=0).reset_index()
    T.to_csv(os.path.join(OUT, f"WA_dr_1_tok_{C}.csv"), index=False)
    if C == "France":
        R = pd.DataFrame(rows, columns=["tok", "pos", "nlen", "band", "street", "drop", "abbrev", "in_addr", "half", "s1", "n_recpure"])
        R.to_pickle(os.path.join(OUT, "WA_dr_1_France_occ.pkl"))
        del rows, R
    L(C, "done", json.dumps({k: v for k, v in SUM.items() if k.startswith(C)}))
    json.dump(SUM, open(os.path.join(OUT, "WA_dr_1_summary.json"), "w"), indent=1)
L("all done")
