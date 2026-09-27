"""RL-15 -- independent audit of S003 (= S002 + max-claimer) and per-country change breakdown. Read-only, CPU only.

1. Re-derive max-claimer from the FULL E018C prediction pickles (S003 used the slim ones):
   per country, a record predicted for >1 S1 is kept only for the highest-probability S1 (ties -> smallest S1 id).
2. Check: full preds == S002 per-S1 sets; my result == S003 per-S1 sets; S003 subset of S002; every removal is a contested loser.
3. Breakdown by country: contested records, claims removed, S1 affected/emptied, claimant counts, loser confidence, margins.
Output: experiments/RL/rl15_maxclaim_audit.json
"""
import os, json, pickle, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(ROOT, "experiments", "TEST_PIPELINE")
S002 = os.path.join(ROOT, "experiments", "SUBMISSIONS", "S002_E018C_s42", "matching_results.tsv")
S003 = os.path.join(TP, "submission_S003_E018C_s42_maxclaim", "matching_results.tsv")
TAB, NL = chr(9), chr(10)


def read_sub(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        assert f.readline().rstrip(NL) == "source1_entity_id" + TAB + "matched_entity_ids"
        for line in f:
            s, m = line.rstrip(NL).split(TAB)
            d[s] = set(m.split(",")) if m else set()
    return d


def main():
    s002, s003 = read_sub(S002), read_sub(S003)
    assert list(s002) == list(s003), "row order differs"
    res = {"rows": len(s002)}
    preds, ctry_of = {}, {}
    for c in ["US", "India", "France"]:
        d = pickle.load(open(os.path.join(TP, f"pred_E018C_s42_{c}_full.pkl"), "rb"))["preds"]
        for s, v in d.items():
            preds[s] = v; ctry_of[s] = c
    # 1) full preds reproduce S002 exactly
    mism = sum(1 for s, L in s002.items() if L != {x for x, _ in preds.get(s, [])})
    res["full_preds_vs_S002_mismatched_rows"] = mism
    # cross-country records
    rec_ctry = collections.defaultdict(set)
    for s, v in preds.items():
        for x, _ in v:
            rec_ctry[x].add(ctry_of[s])
    res["records_predicted_in_2_countries"] = sum(1 for v in rec_ctry.values() if len(v) > 1)
    # 2) my max-claimer
    claims = collections.defaultdict(list)
    for s, v in preds.items():
        for x, p in v:
            claims[(ctry_of[s], x)].append((-p, s))
    keep = set(); ties = 0
    stats = {c: collections.Counter() for c in ["US", "India", "France"]}
    ncl = {c: collections.Counter() for c in stats}
    loser_p = {c: [] for c in stats}; margin = {c: [] for c in stats}; win_p = {c: [] for c in stats}
    src_cont = {c: collections.Counter() for c in stats}
    for (c, x), L in claims.items():
        L.sort()
        keep.add((L[0][1], x))
        if len(L) > 1:
            stats[c]["contested_records"] += 1; stats[c]["claims_removed"] += len(L) - 1
            ncl[c][min(len(L), 5)] += 1; src_cont[c][x[:2]] += 1
            ties += L[0][0] == L[1][0]
            win_p[c].append(-L[0][0])
            for q, _ in L[1:]:
                loser_p[c].append(-q); margin[c].append(-L[0][0] + q)
    mine = {s: {x for x, _ in v if (s, x) in keep} for s, v in preds.items()}
    res["my_maxclaim_vs_S003_mismatched_rows"] = sum(1 for s, L in s003.items() if L != mine.get(s, set()))
    res["S003_not_subset_of_S002_rows"] = sum(1 for s in s002 if not s003[s] <= s002[s])
    res["exact_probability_ties_in_contests"] = ties
    n_s1 = collections.Counter(ctry_of[s] for s in s002 if s in ctry_of)
    for s in s002:
        if s not in ctry_of:
            pass
    # S1 counts per country from the test file order is not needed: use S1 with rows; empty-prediction S1 have no preds entry
    import sys
    sys.path.insert(0, OUT)
    from rl_data import load
    T = load("test", verbose=False)
    tot = T["s1"].country.value_counts().to_dict()
    for c in stats:
        ids = [s for s in preds if ctry_of[s] == c]
        before = sum(len(preds[s]) for s in ids); after = sum(len(mine[s]) for s in ids)
        affected = sum(1 for s in ids if len(mine[s]) < len(preds[s]))
        emptied = sum(1 for s in ids if preds[s] and not mine[s])
        lp = np.array(loser_p[c]); mg = np.array(margin[c]); wp = np.array(win_p[c])
        stats[c].update(dict(test_s1=int(tot[c]), pred_pairs_before=before, pred_pairs_after=after,
                             s1_affected=affected, s1_affected_pct=round(100 * affected / tot[c], 3),
                             s1_emptied=emptied, s1_emptied_pct=round(100 * emptied / tot[c], 3),
                             removed_per_1k_preds=round(1000 * stats[c]["claims_removed"] / before, 2)))
        stats[c]["claimants_per_contested_record"] = {str(k) + ("+" if k == 5 else ""): v for k, v in sorted(ncl[c].items())}
        stats[c]["contested_by_source"] = dict(src_cont[c])
        stats[c]["removed_claim_p_quantiles"] = {q: round(float(np.quantile(lp, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        stats[c]["removed_claims_p_ge_0.99_pct"] = round(100 * float(np.mean(lp >= 0.99)), 2)
        stats[c]["winner_minus_loser_margin_quantiles"] = {q: round(float(np.quantile(mg, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        stats[c]["margin_lt_0.01_pct"] = round(100 * float(np.mean(mg < 0.01)), 2)
        stats[c]["winner_p_ge_0.99_pct"] = round(100 * float(np.mean(wp >= 0.99)), 2)
    res["by_country"] = {c: dict(v) for c, v in stats.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rl15_maxclaim_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
