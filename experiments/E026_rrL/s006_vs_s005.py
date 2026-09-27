"""Row-level comparison of the S006 submission against S005 (both read-only) -> experiments/P3_rrL/s006_vs_s005.json.
Per country: S1 rows whose matched list differs, predicted pairs added / removed, non-empty rows, candidate rows identical."""
import os, json, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S5 = os.path.join(ROOT, "experiments", "RL", "submissions", "S005_RL27NEW_s42_mc")
S6 = os.path.join(ROOT, "experiments", "P3_rrL", "submission_S006_rrL_mc")
TAB, NL = chr(9), chr(10)
ctry = {}
with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for l in f:
        p = l.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]


def read(d, name):
    out = {}
    with open(os.path.join(d, name), encoding="utf-8") as f:
        f.readline()
        for l in f:
            s, m = l.rstrip(NL).split(TAB); out[s] = m
    return out


m5, m6 = read(S5, "matching_results.tsv"), read(S6, "matching_results.tsv")
assert set(m5) == set(m6) == set(ctry)
c5, c6 = read(S5, "candidate_pairs.tsv"), read(S6, "candidate_pairs.tsv")
st = collections.defaultdict(collections.Counter)
for s, c in ctry.items():
    a, b = set(filter(None, m5[s].split(","))), set(filter(None, m6[s].split(",")))
    t = st[c]; t["s1"] += 1; t["rows_differ"] += a != b; t["pairs_S005"] += len(a); t["pairs_S006"] += len(b)
    t["added"] += len(b - a); t["removed"] += len(a - b); t["nonempty_S005"] += bool(a); t["nonempty_S006"] += bool(b)
    t["cand_rows_identical"] += c5[s] == c6[s]
res = {c: dict(v, rows_differ_pct=round(100 * v["rows_differ"] / v["s1"], 3)) for c, v in st.items()}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "experiments", "P3_rrL", "s006_vs_s005.json"), "w"), indent=1)
