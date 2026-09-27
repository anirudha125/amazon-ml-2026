"""RL-30 Part 7 (investigator F): leakage / competition-rule validity measurements. READ-ONLY, label use only on TRAIN/V1.
Outputs: experiments/RL/rl30/F_results.json
M1 corpus symmetry: S1 co-location + near-duplicate co-located pairs per split/country (train vs test)
M2 ID-order leakage: |id gap| of co-located / near-dup S1 pairs vs random pairs; train S1-id vs linked-record-id rank correlation
M3 pseudo-label noise (family b): V1 precision by p band; test provable-conflict lower bound by p band; coverage
M4 street-type list (family d): hand list coverage vs data-derived (token after house number) list
M5 department<->region (family e): derivable from test co-occurrence? purity of the derived map
M6 decoy / one-token-substitution classes (family f/c): label-backed rates on V1; class mix of accepted pairs V1 vs France test
M7 LB overfitting (family g): per-S1 paired F0.5 differences between thresholds on V1 -> SE for public/private subset sizes
M8 vocabulary transfer (family a): France S1 name tokens covered by train S1 vocabulary
"""
import sys, os, re, json, time, random, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
R = {}
rng = np.random.default_rng(0)

TE = load("test", verbose=False); log("test loaded")
TR = load("train", verbose=False); log("train loaded")
idnum = lambda s: int(s[3:])


# ---------------- M1 + M2: co-location, near-dup pairs, ID gaps ----------------
def coloc(s1, split):
    out, gaps = {}, {}
    for c, g in s1.groupby("country"):
        ak = np.array([akey(a) for a in g.addr.values], dtype=object)
        names = g.name.values; ids = np.array([idnum(x) for x in g.id.values], dtype=np.int64)
        vc = collections.Counter(a for a in ak if a)
        grp = collections.defaultdict(list)
        for i, a in enumerate(ak):
            if a and vc[a] >= 2:
                grp[a].append(i)
        n_coloc = sum(len(v) for v in grp.values())
        pairs = no_share = nd = skipped = 0
        cg, ndg, nd_pairs = [], [], []
        for a, idx in grp.items():
            if len(idx) > 200:
                skipped += len(idx); continue
            ts = [set(toks(names[i])) for i in idx]
            for x, y in itertools.combinations(range(len(idx)), 2):
                pairs += 1
                A, B = ts[x], ts[y]
                gap = abs(int(ids[idx[x]]) - int(ids[idx[y]]))
                cg.append(gap)
                if not (A & B):
                    no_share += 1
                elif A != B and len(A - B) <= 1 and len(B - A) <= 1:
                    nd += 1; ndg.append(gap); nd_pairs.append((g.id.values[idx[x]], g.id.values[idx[y]]))
        rp = rng.integers(0, len(g), size=(200000, 2))
        rg = np.abs(ids[rp[:, 0]] - ids[rp[:, 1]])
        out[c] = dict(n_s1=len(g), coloc_s1=n_coloc, coloc_share=round(n_coloc / len(g), 4), coloc_pairs=pairs,
                      pairs_no_shared_token_share=round(no_share / max(pairs, 1), 4), neardup_pairs=nd,
                      neardup_per_100k_s1=round(nd / len(g) * 1e5, 2), s1_in_skipped_groups_gt200=skipped)
        gaps[c] = dict(coloc=np.array(cg), neardup=np.array(ndg), random=rg, nd_pairs=nd_pairs)
    return out, gaps


def gap_summary(gd):
    res = {}
    try:
        from scipy.stats import ks_2samp
    except Exception:
        ks_2samp = None
    for k in ("coloc", "neardup"):
        a = gd[k]
        if len(a) == 0:
            continue
        r = dict(n=int(len(a)), median_gap=float(np.median(a)), random_median_gap=float(np.median(gd["random"])),
                 share_gap_lt_1000=float(np.mean(a < 1000)), random_share_gap_lt_1000=float(np.mean(gd["random"] < 1000)))
        if ks_2samp is not None and len(a) >= 5:
            ks = ks_2samp(a, gd["random"]); r.update(ks_stat=round(float(ks.statistic), 4), ks_p=float(ks.pvalue))
        res[k] = r
    return res


R["M1_coloc"] = {}; R["M2_idgap"] = {}
for split, D in (("test", TE), ("train", TR)):
    o, gps = coloc(D["s1"], split)
    R["M1_coloc"][split] = o
    R["M2_idgap"][split] = {c: gap_summary(g) for c, g in gps.items()}
    if split == "test":
        fr_nd = gps.get("France", {}).get("nd_pairs", [])
    else:
        tr_nd = {c: g["nd_pairs"] for c, g in gps.items()}
    log("M1/M2", split, json.dumps(o))

# train near-dup pairs: label-backed (are both sides real, separately-linked businesses?) + V1 involvement
gt = TR["gt"]
ngt = gt.groupby("s1").size()
v1m = np.load(PATHS["v1_meta"], allow_pickle=True)
v1ids = set(v1m["s1_ids"].tolist())
R["M1_train_neardup_labels"] = {}
for c, prs in tr_nd.items():
    both = one = none = inv1 = 0
    for a, b in prs:
        na, nb = int(ngt.get(a, 0)), int(ngt.get(b, 0))
        both += (na > 0 and nb > 0); one += ((na > 0) != (nb > 0)); none += (na == 0 and nb == 0)
        inv1 += (a in v1ids) or (b in v1ids)
    R["M1_train_neardup_labels"][c] = dict(pairs=len(prs), both_have_links=both, one_has_links=one, neither=none, pairs_touching_V1=inv1)
log("train neardup labels", R["M1_train_neardup_labels"])

# train: S1 id vs linked record id rank correlation (ID-order leakage)
def rankcorr(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])
g2 = gt.sample(min(len(gt), 2_000_000), random_state=0)
a = np.array([idnum(x) for x in g2.s1.values]); b = np.array([idnum(x) for x in g2.rec.values])
is2 = np.array([x.startswith("S2") for x in g2.rec.values])
R["M2_train_s1_rec_id_spearman"] = dict(S2=round(rankcorr(a[is2], b[is2]), 5), S3=round(rankcorr(a[~is2], b[~is2]), 5), n=int(len(g2)))
log("M2 spearman", R["M2_train_s1_rec_id_spearman"])


# ---------------- M3: pseudo-label noise ----------------
p = np.load(PATHS["v1_p_new"]); y = v1m["y"].astype(bool); s1idx = v1m["s1idx"]
ctry_row = v1m["country"][s1idx]
bands = [0.78, 0.9, 0.95, 0.99, 0.999]
R["M3_V1_precision"] = {}
for c in ("US", "India", "all"):
    msk = np.ones_like(y) if c == "all" else (ctry_row == c)
    rr = {}
    for th in bands:
        acc = msk & (p >= th)
        rr[str(th)] = dict(accepted=int(acc.sum()), fp=int((acc & ~y).sum()), precision=round(float(y[acc].mean()), 5),
                           recall_of_pool_pos=round(float((acc & y).sum() / max((msk & y).sum(), 1)), 4))
    R["M3_V1_precision"][c] = rr
n_s1_test = TE["s1"].country.value_counts().to_dict()
R["M3_test_conflict_lower_bound"] = {}
for c in ("France", "US", "India"):
    A = accepted(f"S005_{c}")
    rr = {}
    for th in bands:
        a2 = A[A.p >= th]
        cl = a2.groupby("rec").size()
        extra = int((cl - 1).clip(lower=0).sum())
        rr[str(th)] = dict(accepted=int(len(a2)), multi_claimed_records=int((cl > 1).sum()),
                           provable_fp_lower_bound=extra, lb_fp_rate=round(extra / max(len(a2), 1), 5),
                           s1_coverage=round(a2.s1.nunique() / n_s1_test[c], 4))
    R["M3_test_conflict_lower_bound"][c] = rr
    del A
log("M3", json.dumps(R["M3_test_conflict_lower_bound"]["France"]))


# ---------------- M4: street-type words ----------------
def next_token_after_number(addr, last=False):
    for comp in fold(addr).split(","):
        m = re.search(r"\d+", comp)
        if m:
            rest = re.findall(r"[a-z]+", comp[m.end():])
            rest = [t for t in rest if len(t) > 1 and t not in ("bis", "ter")]
            return (rest[-1] if last else rest[0]) if rest else None
    return None

R["M4_street_type"] = dict(hand_list_size=len(STREET_TYPE))
for c in ("France", "US"):
    for src in ("s1", "s2", "s3"):
        d = TE[src]; d = d[d.country == c]
        nt = [next_token_after_number(x) for x in d.addr.values]
        nt = [t for t in nt if t]
        C = collections.Counter(nt); tot = len(nt)
        top = C.most_common(40)
        cov = lambda k: round(sum(n for _, n in C.most_common(k)) / tot, 4)
        R["M4_street_type"][f"{c}_{src}"] = dict(
            n_with_number=tot, hand_list_coverage=round(sum(n for t, n in C.items() if t in STREET_TYPE) / tot, 4),
            data_top10_coverage=cov(10), data_top20_coverage=cov(20), data_top40_coverage=cov(40),
            top20=top[:20], top20_not_in_hand_list=[t for t, _ in top[:20] if t not in STREET_TYPE],
            hand_words_seen=sum(1 for t in STREET_TYPE if C.get(t, 0) > 0))
        lt = [next_token_after_number(x, last=True) for x in d.addr.values]; lt = [t for t in lt if t]
        CL = collections.Counter(lt)
        R["M4_street_type"][f"{c}_{src}"].update(
            last_token_hand_list_coverage=round(sum(n for t, n in CL.items() if t in STREET_TYPE) / max(len(lt), 1), 4),
            last_token_data_top20_coverage=round(sum(n for _, n in CL.most_common(20)) / max(len(lt), 1), 4),
            last_token_top10=CL.most_common(10))
log("M4", {k: (v["hand_list_coverage"], v["data_top20_coverage"]) for k, v in R["M4_street_type"].items() if isinstance(v, dict)})


# ---------------- M5: department <-> region from test co-occurrence ----------------
def comps_nd(a):
    return [c.strip() for c in fold(a).split(",") if c.strip() and not re.search(r"\d", c)]

fr = {src: TE[src][TE[src].country == "France"].addr.values for src in ("s1", "s2", "s3")}
freq = {src: collections.Counter(c for a in fr[src] for c in comps_nd(a)) for src in fr}
tot = collections.Counter(); [tot.update(f) for f in freq.values()]
FREQ = {k for k, v in tot.items() if v >= 1000}                      # frequent components (label-free cut)
# S2/S3-only frequent components (S1 never uses them): candidates for "department"-style units / spelling variants
s23only = sorted([k for k in FREQ if freq["s1"].get(k, 0) < 0.001 * (freq["s2"].get(k, 0) + freq["s3"].get(k, 0)) + 1],
                 key=lambda k: -tot[k])
# admin units of S1 = frequent S1 components that co-occur with >=3 distinct frequent S1 components (regions)
co1 = collections.defaultdict(collections.Counter)
for a in fr["s1"]:
    cs = [c for c in comps_nd(a) if c in FREQ]
    for x in cs:
        for z in cs:
            if x != z:
                co1[x][z] += 1
s1_regions = sorted([k for k in co1 if sum(1 for z, n in co1[k].items() if n >= 100) >= 3], key=lambda k: -tot[k])
# city -> region in S1 (purity)
city_region = {}
for k, cnt in co1.items():
    if k in s1_regions:
        continue
    rc = {r: cnt.get(r, 0) for r in s1_regions}
    s = sum(rc.values())
    if s >= 100:
        r = max(rc, key=rc.get); city_region[k] = (r, round(rc[r] / s, 4), s)
# derived dept -> region: via cities co-occurring with dept in S2/S3
co23 = collections.defaultdict(collections.Counter)
for src in ("s2", "s3"):
    for a in fr[src]:
        cs = [c for c in comps_nd(a) if c in FREQ]
        for x in cs:
            for z in cs:
                if x != z:
                    co23[x][z] += 1
derived = {}
for d in s23only:
    mass = collections.Counter()
    for city, n in co23[d].items():
        if city in city_region:
            mass[city_region[city][0]] += n
    direct = {r: co23[d].get(r, 0) for r in s1_regions}          # dept & region in the same address (should be ~0)
    if sum(mass.values()) > 0:
        r, m = mass.most_common(1)[0]
        derived[d] = dict(region=r, purity=round(m / sum(mass.values()), 4), support=int(sum(mass.values())),
                          cities=co23[d].most_common(4), direct_cooccurrence_with_regions=direct)
dept_share = {}
for src in ("s2", "s3"):
    n = len(fr[src]); hit = 0
    depts = set(derived)
    for a in fr[src]:
        if any(c in depts for c in comps_nd(a)):
            hit += 1
    dept_share[src] = round(hit / n, 4)
R["M5_dept_region"] = dict(n_frequent_components=len(FREQ), s1_regions=s1_regions,
                          s1_region_counts={r: freq["s1"][r] for r in s1_regions},
                          s1_share_with_region=round(sum(1 for a in fr["s1"] if any(c in s1_regions for c in comps_nd(a))) / len(fr["s1"]), 4),
                          s1_dept_counts={d: freq["s1"].get(d, 0) for d in s23only},
                          city_region=city_region, s23_only_frequent=s23only, derived_map=derived,
                          share_s23_records_with_dept_style_unit=dept_share)
log("M5", json.dumps({k: (v["region"], v["purity"]) for k, v in derived.items()}))


# ---------------- M6: one-token-substitution / add-drop classes ----------------
def rel(A, B):
    if A == B:
        return "same"
    if not (A & B):
        return "disjoint"
    da, db = len(A - B), len(B - A)
    if da + db == 1:
        return "add_drop_1"
    if da == 1 and db == 1:
        return "sub_1for1"
    return "other_overlap"

s1n = pd.Series(TR["s1"].name.values, index=TR["s1"].id.values)
s1a = pd.Series(TR["s1"].addr.values, index=TR["s1"].id.values)
recn = pd.concat([pd.Series(TR[s].name.values, index=TR[s].id.values) for s in ("s2", "s3")])
reca = pd.concat([pd.Series(TR[s].addr.values, index=TR[s].id.values) for s in ("s2", "s3")])
v1_s1 = v1m["s1_ids"][s1idx]
N1 = s1n.reindex(v1m["s1_ids"]).values; A1 = s1a.reindex(v1m["s1_ids"]).values
CN = recn.reindex(v1m["cand"]).values; CA = reca.reindex(v1m["cand"]).values
log("M6 names mapped")
core1 = [core_tokens(x) for x in N1]; ak1 = [akey(x) for x in A1]
cls = np.empty(len(y), dtype=object); same_addr = np.zeros(len(y), bool)
cache_c = {}
for i in range(len(y)):
    j = s1idx[i]
    B = core_tokens(CN[i]) if isinstance(CN[i], str) else frozenset()
    cls[i] = rel(core1[j], B)
    same_addr[i] = bool(ak1[j]) and isinstance(CA[i], str) and akey(CA[i]) == ak1[j]
log("M6 classes built")
rl27 = np.load(PATHS["v1_rl27"]); coloc_col = rl27[:, 4]
acc = p >= NEW_TH
R["M6_V1_classes"] = {}
for key, msk in (("all", np.ones_like(y)), ("same_addr", same_addr), ("s1_colocated", coloc_col > 0)):
    rr = {}
    for k in ("same", "add_drop_1", "sub_1for1", "other_overlap", "disjoint"):
        m = msk & (cls == k)
        if m.sum() == 0:
            continue
        rr[k] = dict(pairs=int(m.sum()), pos_rate=round(float(y[m].mean()), 4), accepted=int((m & acc).sum()),
                     acc_fp=int((m & acc & ~y).sum()), fn=int((m & ~acc & y).sum()))
    R["M6_V1_classes"][key] = rr
log("M6 V1", json.dumps(R["M6_V1_classes"]["all"]))
# accepted-pair class mix: V1 (US/India) vs France test (label-free diagnostic, NOT a tuning target)
v1acc = collections.Counter(cls[acc]); tv = sum(v1acc.values())
A = accepted("S005_France")
fs1 = TE["s1"][TE["s1"].country == "France"]
tn1 = pd.Series(fs1.name.values, index=fs1.id.values)
trn = pd.concat([pd.Series(TE[s].name.values, index=TE[s].id.values) for s in ("s2", "s3")])
samp = A.sample(300000, random_state=0)
n1 = tn1.reindex(samp.s1.values).values; nr = trn.reindex(samp.rec.values).values
fc = collections.Counter(rel(core_tokens(a), core_tokens(b) if isinstance(b, str) else frozenset()) for a, b in zip(n1, nr))
tf = sum(fc.values())
R["M6_accepted_class_mix"] = dict(V1_US_India={k: round(v / tv, 4) for k, v in v1acc.items()},
                                  France_test_S005_sample300k={k: round(v / tf, 4) for k, v in fc.items()})
log("M6 mix", R["M6_accepted_class_mix"])


# ---------------- M7: LB threshold overfitting ----------------
n_gt = v1m["n_gt"]
def per_s1_f(th):
    pred = p >= th
    tp = np.bincount(s1idx, weights=(pred & y).astype(np.float64), minlength=len(n_gt))
    npred = np.bincount(s1idx, weights=pred.astype(np.float64), minlength=len(n_gt))
    f = np.where((n_gt == 0), (npred == 0).astype(float), 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9))
    return f
ths = [0.6, 0.7, 0.72, 0.78, 0.85, 0.9, 0.95]
F = {th: per_s1_f(th) for th in ths}
base = F[0.78]
n_test = len(TE["s1"]); n_fr = n_s1_test["France"]
R["M7_threshold_lb"] = dict(V1_macro={str(t): round(float(F[t].mean() * 100), 3) for t in ths},
                            per_s1_sd_at_078=round(float(base.std()), 4), paired={})
for t in ths:
    if t == 0.78:
        continue
    d = F[t] - base
    sd = float(d.std())
    e = dict(V1_delta_pp=round(float(d.mean() * 100), 3), per_s1_sd_diff=round(sd, 5), share_s1_changed=round(float((d != 0).mean()), 4))
    for f in (0.1, 0.3, 0.5):                         # public fraction UNKNOWN -> scenarios
        for nm, n in (("all", n_test), ("France", n_fr)):
            npub = n * f; npriv = n * (1 - f)
            e[f"{nm}_pub{int(f*100)}_se_pub_pp"] = round(sd / np.sqrt(npub) * 100, 4)
            e[f"{nm}_pub{int(f*100)}_sd_pub_minus_priv_pp"] = round(sd * np.sqrt(1 / npub + 1 / npriv) * 100, 4)
    R["M7_threshold_lb"]["paired"][str(t)] = e
R["M7_threshold_lb"]["n_test_s1"] = int(n_test); R["M7_threshold_lb"]["n_france_s1"] = int(n_fr)
log("M7", R["M7_threshold_lb"]["V1_macro"])


# ---------------- M8: France vocabulary covered by train vocabulary ----------------
trdf = collections.Counter()
for nm in TR["s1"].name.values:
    trdf.update(set(toks(nm)))
frocc = collections.Counter()
for nm in fs1.name.values:
    frocc.update(toks(nm))
totocc = sum(frocc.values())
cov = {k: round(sum(n for t, n in frocc.items() if trdf.get(t, 0) >= k) / totocc, 4) for k in (1, 10, 100, 1000)}
words = "ecole amicale club sportive centre sas sarl comite union amis association lycee college loisirs".split()
R["M8_vocab"] = dict(france_token_occurrences=totocc, france_distinct=len(frocc),
                     occ_share_with_train_df_ge=cov,
                     top30_france_tokens=[(t, n, trdf.get(t, 0)) for t, n in frocc.most_common(30)],
                     listed_words_train_df={w: trdf.get(w, 0) for w in words},
                     listed_words_france_occ={w: frocc.get(w, 0) for w in words},
                     n_train_s1=len(TR["s1"]))
log("M8", cov)

json.dump(R, open(os.path.join(OUT, "F_results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("done ->", os.path.join(OUT, "F_results.json"))
