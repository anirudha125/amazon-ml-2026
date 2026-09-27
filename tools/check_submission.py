"""Streaming integrity check for a submission dir (bounded memory; order-agnostic between files is NOT assumed:
both files must list S1 rows in the same order, which predict_test.write/write2 guarantee).
Checks: headers, every test S1 exactly once, predictions subset of candidate row, no duplicate ids, all predicted ids exist
in test S2/S3, no S1-prefixed ids; prints per-country stats. Usage: python tools/check_submission.py <submission_dir>"""
import sys, os, collections
D = sys.argv[1]
TEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "student_resource", "dataset", "test")
TAB, NL = chr(9), chr(10)
ctry = {}
with open(os.path.join(TEST, "test_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for l in f:
        p = l.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
seen = set(); pred_ids = set(); bad = collections.Counter(); st = collections.defaultdict(lambda: [0, 0, 0, 0])
with open(os.path.join(D, "matching_results.tsv"), encoding="utf-8") as fm, open(os.path.join(D, "candidate_pairs.tsv"), encoding="utf-8") as fc:
    assert fm.readline().rstrip(NL) == "source1_entity_id" + TAB + "matched_entity_ids"
    assert fc.readline().rstrip(NL) == "source1_entity_id" + TAB + "candidate_entity_ids"
    for lm, lc in zip(fm, fc):
        s, m = lm.rstrip(NL).split(TAB); s2, c = lc.rstrip(NL).split(TAB)
        if s != s2: bad["row_order_mismatch"] += 1; continue
        if s in seen: bad["duplicate_s1_row"] += 1
        if s not in ctry: bad["unknown_s1"] += 1; continue
        seen.add(s)
        L = m.split(",") if m else []; CL = c.split(",") if c else []
        if len(L) != len(set(L)): bad["dup_in_match_list"] += 1
        if len(CL) != len(set(CL)): bad["dup_in_cand_list"] += 1
        if not set(L) <= set(CL): bad["pred_not_in_candidates"] += 1
        pred_ids.update(L)
        z = st[ctry[s]]; z[0] += 1; z[1] += bool(L); z[2] += len(L); z[3] += len(CL)
valid = set()
for src in "23":
    with open(os.path.join(TEST, f"test_source{src}.tsv"), encoding="utf-8") as f:
        f.readline()
        for l in f:
            valid.add(l[:l.find(TAB)])
bad["pred_id_not_in_test_S2_S3"] = len(pred_ids - valid); bad["pred_id_S1_prefixed"] = sum(1 for x in pred_ids if x.startswith("S1-"))
bad["test_S1_missing"] = len(set(ctry) - seen)
print("rows", len(seen), "| issues:", dict(bad), "| unique predicted ids", len(pred_ids))
for k, (n, ne, npred, nc) in sorted(st.items()):
    print(f"{k:7s} S1 {n:,}  nonempty {ne/n*100:.1f}%  preds/S1 {npred/n:.2f}  cands/S1 {nc/n:.1f}")
print("CHECK PASS" if sum(bad.values()) == 0 else "CHECK FAIL")
