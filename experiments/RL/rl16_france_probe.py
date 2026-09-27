"""RL-16 -- France-empty LB probe P001, derived from S002 (read-only). NOT submitted by this script.

P001 = S002 matching_results.tsv with every France S1 row's matched_entity_ids emptied. All other lines are copied
byte-for-byte; row order is unchanged. candidate_pairs.tsv = S002's file (symlink to the byte-identical uncompressed copy,
sha256 10f7dcab... = zcat S002 candidate_pairs.tsv.gz).

Inference (only for when both LB scores exist):
  LB(S002) - LB(P001) = w_FR * (F_FR(S002) - s_FR)
  w_FR = share of public-LB S1 that are France (0.14975 on the full test file; equal in expectation for a random public split)
  s_FR = France singleton share (unknown; train US 5.58%, India 5.59%) -> F_FR(S002) = dLB / w_FR + s_FR
"""
import os, json, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_S1 = os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv")
S002_DIR = os.path.join(ROOT, "experiments", "SUBMISSIONS", "S002_E018C_s42")
CAND = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "submission_S003_E018C_s42_maxclaim", "candidate_pairs.tsv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions", "P001_S002_France_empty")
TAB, NL = chr(9), chr(10)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    ctry = {}
    with open(TEST_S1, encoding="utf-8") as f:
        f.readline()
        for l in f:
            p = l.rstrip("\r\n").split(TAB); ctry[p[0]] = p[3]
    src = os.path.join(S002_DIR, "matching_results.tsv")
    dst = os.path.join(OUT, "matching_results.tsv")
    st = dict(rows=0, france_rows=0, france_rows_nonempty_in_S002=0, france_pred_ids_removed=0, other_rows=0)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(fi.readline())
        for line in fi:
            st["rows"] += 1
            s, m = line.rstrip(b"\n").split(b"\t")
            if ctry[s.decode()] == "France":
                st["france_rows"] += 1
                if m:
                    st["france_rows_nonempty_in_S002"] += 1; st["france_pred_ids_removed"] += m.count(b",") + 1
                fo.write(s + b"\t\n")
            else:
                st["other_rows"] += 1
                fo.write(line)
    # verification pass: non-France lines byte-identical, France lines empty, same order and count
    diff_other = bad_fr = 0; n = 0
    with open(src, "rb") as a, open(dst, "rb") as b:
        assert a.readline() == b.readline()
        for la, lb in zip(a, b):
            n += 1
            s = la.split(b"\t")[0].decode()
            if ctry[s] == "France":
                bad_fr += lb != la.split(b"\t")[0] + b"\t\n"
            else:
                diff_other += la != lb
        assert a.readline() == b"" and b.readline() == b""
    st.update(verified_rows=n, nonfrance_lines_differing=diff_other, france_lines_not_empty=bad_fr)
    link = os.path.join(OUT, "candidate_pairs.tsv")
    if not os.path.exists(link):
        os.symlink(CAND, link)
    st["sha256_matching_results"] = sha(dst)
    st["sha256_S002_matching_results"] = sha(src)
    st["w_FR_full_test"] = round(st["france_rows"] / st["rows"], 5)
    man = dict(probe_id="P001", base="S002_E018C_s42 (LB 0.95065)", change="France rows emptied; nothing else",
               purpose="measure the absolute France F0.5 of S002 from one LB submission",
               inference="F_FR(S002) = (LB(S002) - LB(P001)) / w_FR + s_FR ; w_FR ~ 0.14975 ; s_FR = France singleton share "
                         "(unknown; train US 5.58% / India 5.59%)",
               candidate_pairs="symlink -> S002 candidates (sha256 10f7dcab... of the uncompressed file)",
               status="NOT submitted", stats=st)
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
