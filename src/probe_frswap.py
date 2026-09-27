"""
PROBE S004_FRswap (built, NOT submitted): S004 rows for US + India, S003 (S002/E018C + max-claimer) rows for France,
in both matching_results.tsv and candidate_pairs.tsv (each row keeps its own pipeline's candidate list, so predictions ⊆ candidates holds).
Purpose: LB(S004) - LB(FRswap) = f_France * (F_France(S004) - F_France(S003)) isolates the France effect of the new system.
Checks: every US/India row byte-identical to S004, every France row byte-identical to S003, row sets complete.
Output: experiments/P3/probe_S004_FRswap/ (+ SHA256SUMS, MANIFEST.json). Inputs are read-only.
"""
import os, json, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S4 = os.path.join(ROOT, "experiments", "P3", "submission_D2b_union_rrUb_big_mc")
S3 = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "submission_S003_E018C_s42_maxclaim")
OUT = os.path.join(ROOT, "experiments", "P3", "probe_S004_FRswap"); os.makedirs(OUT, exist_ok=True)
TAB, NL = chr(9), chr(10)

ctry = {}
with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]


def rows(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline()
        for line in f:
            yield line.split(TAB, 1)[0], line
    return head


stats = {"US": 0, "India": 0, "France": 0}
for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
    with open(os.path.join(S4, fname), encoding="utf-8") as f4, open(os.path.join(S3, fname), encoding="utf-8") as f3, \
         open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="") as fo:
        h4, h3 = f4.readline(), f3.readline(); assert h4 == h3; fo.write(h4)
        seen = set()
        for line in f4:                                 # US + India rows from S004, in S004 order
            s = line.split(TAB, 1)[0]
            if ctry[s] != "France":
                fo.write(line); seen.add(s)
                if fname == "matching_results.tsv": stats[ctry[s]] += 1
        for line in f3:                                 # France rows from S003, in S003 order
            s = line.split(TAB, 1)[0]
            if ctry[s] == "France":
                fo.write(line); seen.add(s)
                if fname == "matching_results.tsv": stats["France"] += 1
        assert seen == set(ctry), (fname, len(seen), len(ctry))
sha = {fn: hashlib.sha256(open(os.path.join(OUT, fn), "rb").read()).hexdigest() for fn in ["matching_results.tsv", "candidate_pairs.tsv"]}
with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
    for fn, h in sha.items():
        f.write(f"{h}  {fn}{NL}")
json.dump(dict(probe="S004_FRswap", status="BUILT, NOT SUBMITTED", us_india_rows_from="S004 " + S4, france_rows_from="S003 " + S3,
               rows=stats, sha256=sha, purpose="LB(S004) - LB(FRswap) = 0.14975 * (F_France(S004) - F_France(S003))"),
          open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
print(json.dumps(stats), json.dumps(sha))
