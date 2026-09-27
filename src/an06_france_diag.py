"""
AN06 -- label-free test-prediction diagnostics per country: S002 / S003 (S002 + max-claimer) / D2b pre-max-claimer / S004 (D2b + max-claimer).
Every set is rebuilt from stored per-pair probabilities and checked against the frozen submission files (per-S1 set equality) first.
Metrics: non-empty rate, preds/S1, records claimed by >1 S1 (pre max-claimer), max-claimer removals, high-confidence conflict rate,
probability distribution, per-S1 change classes S004 vs S002 and S004 vs S003. Read-only on all inputs. Output: experiments/P3/an06_diag.json
"""
import os, sys, json, pickle, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "experiments", "P3"); SP = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "slim_preds")
FILES = {"S002": os.path.join(ROOT, "experiments", "SUBMISSIONS", "S002_E018C_s42", "matching_results.tsv"),
         "S003": os.path.join(ROOT, "experiments", "TEST_PIPELINE", "submission_S003_E018C_s42_maxclaim", "matching_results.tsv"),
         "S004": os.path.join(P3, "submission_D2b_union_rrUb_big_mc", "matching_results.tsv")}
TAB, NL = chr(9), chr(10)


def read_sub(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, m = line.rstrip(NL).split(TAB); out[s] = set(m.split(",")) if m else set()
    return out


def maxclaim(P):
    cl = collections.defaultdict(list)
    for s, v in P.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in P.items()}


def main():
    ctry = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    subs = {k: read_sub(v) for k, v in FILES.items()}
    res = {}
    for c in ["US", "India", "France"]:
        S1 = [s for s, v in ctry.items() if v == c]; n = len(S1)
        old = pickle.load(open(os.path.join(SP, f"E018C_s42_{c}.pkl"), "rb"))["preds"]
        new = pickle.load(open(os.path.join(P3, f"preds_D2b_union_rrUb_big_{c}.pkl"), "rb"))["preds"]
        M = {"S002": (old, .72), "S003": (maxclaim(old), .72), "D2b_pre": (new, .72), "S004": (maxclaim(new), .72)}
        for k in ["S002", "S003", "S004"]:        # reconstruction check against the frozen files
            bad = sum(1 for s in S1 if {x for x, _ in M[k][0].get(s, [])} != subs[k][s])
            assert bad == 0, (c, k, bad)
        r = {}
        for k, (P, th) in M.items():
            cl = collections.defaultdict(list)
            for s, v in P.items():
                for x, p in v:
                    cl[x].append(p)
            npred = sum(len(v) for v in P.values()); p = np.array([x for v in P.values() for _, x in v])
            cont = {x: L for x, L in cl.items() if len(L) > 1}
            top = np.array([max(x for _, x in v) for v in P.values() if v])
            r[k] = dict(nonempty_pct=round(100 * sum(1 for s in S1 if P.get(s)) / n, 2), preds_per_s1=round(npred / n, 3),
                        records_multi_claimed=len(cont), claims_on_multi_claimed=sum(len(L) for L in cont.values()),
                        maxclaim_removals=sum(len(L) - 1 for L in cont.values()),
                        maxclaim_removals_per_1k=round(1000 * sum(len(L) - 1 for L in cont.values()) / max(npred, 1), 2),
                        s1_touching_conflict_pct=round(100 * sum(1 for v in P.values() if any(x in cont for x, _ in v)) / n, 2),
                        hc_conflict_records=sum(1 for L in cont.values() if sum(q >= 0.95 for q in L) >= 2),
                        hc_conflict_share_of_conflicts=round(sum(1 for L in cont.values() if sum(q >= 0.95 for q in L) >= 2) / max(len(cont), 1), 4),
                        p_quantiles_1_5_25_50=[round(float(np.percentile(p, q)), 4) for q in (1, 5, 25, 50)] if len(p) else None,
                        frac_p_ge_95=round(float((p >= .95).mean()), 4) if len(p) else None,
                        frac_p_within_05_of_th=round(float((p < th + .05).mean()), 4) if len(p) else None,
                        top1_q5_q25=[round(float(np.percentile(top, q)), 4) for q in (5, 25)] if len(top) else None)
        def change(a, b):   # per-S1 classes, b relative to a
            C = collections.Counter()
            for s in S1:
                A, B = a[s], b[s]
                if A == B: C["identical"] += 1
                elif not A: C["newly_filled"] += 1
                elif not B: C["emptied"] += 1
                elif B > A: C["superset"] += 1
                elif B < A: C["subset"] += 1
                elif A & B: C["overlap_changed"] += 1
                else: C["disjoint"] += 1
            return {k: dict(n=v, pct=round(100 * v / n, 2)) for k, v in sorted(C.items(), key=lambda x: -x[1])}
        r["changes_S004_vs_S002"] = change(subs["S002"], subs["S004"]); r["changes_S004_vs_S003"] = change(subs["S003"], subs["S004"])
        r["n_s1"] = n
        res[c] = r
        print(c, json.dumps({k: r[k] for k in ["S002", "S003", "D2b_pre", "S004"]}, indent=None)[:4000], flush=True)
        print(c, "changes S004 vs S002", json.dumps(r["changes_S004_vs_S002"]), flush=True)
        print(c, "changes S004 vs S003", json.dumps(r["changes_S004_vs_S003"]), flush=True)
    json.dump(res, open(os.path.join(P3, "an06_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
