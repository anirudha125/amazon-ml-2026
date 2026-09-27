# Validation record (verbatim tool output captured in the A100 session, 2026-09-26; transcribed at shutdown audit 22:3x UTC)

These validators were run once on each file. Their output was printed to the session log, not written to disk, so it is copied here.
Nothing was re-run for this record. The files themselves are unchanged. Verify integrity with the sha256 lines below
(S004: `sha256sum -c experiments/P3/submission_D2b_union_rrUb_big_mc/SHA256SUMS` from the repo root).

## S004 = experiments/P3/submission_D2b_union_rrUb_big_mc/  (LB 0.979772; matching sha256 b9982fd7c91b7b6c4c21f03586a940d57046047dc90d8a753ba3d74689937650)
tools/check_submission.py:
    rows 1732544 | issues: {'pred_id_not_in_test_S2_S3': 0, 'pred_id_S1_prefixed': 0, 'test_S1_missing': 0} | unique predicted ids 5809618
    France  S1 259,452  nonempty 94.6%  preds/S1 3.31  cands/S1 127.9
    India   S1 809,986  nonempty 94.2%  preds/S1 3.36  cands/S1 129.1
    US      S1 663,106  nonempty 94.2%  preds/S1 3.36  cands/S1 123.8
    CHECK PASS
student_resource/utils/validate_submission.py (default mode, no --check-ids):
    matching_results.tsv: 1732544 rows (99504 empty, 1633040 non-empty).
    candidate_pairs.tsv: 1732544 rows (0 empty, 1732544 non-empty).
    PASS — no blocking issues found. Safe to submit.

## S004-nomc = experiments/P3/submission_D2b_union_rrUb_big/  (not submitted; matching sha256 56c359f4d19ff60b20373f27cc3e5f5b188d4b704a04c10c8e42f727568478ea)
check_submission: rows 1732544, issues all 0, unique predicted ids 5809618; France nonempty 95.2% preds/S1 3.48; India 94.2% / 3.37; US 94.2% / 3.36; CHECK PASS
official: matching 1732544 rows (97519 empty, 1635025 non-empty); candidate 1732544 rows (0 empty); PASS

## S004_FRswap = experiments/P3/probe_S004_FRswap/  (built, NOT submitted; matching sha256 0664654b7dc50f543f1c7b274437d0c2cc8e450c1e1a20055c3618865b5a807c)
check_submission: rows 1732544, issues all 0, unique predicted ids 5792132; France nonempty 94.4% preds/S1 3.24 cands/S1 115.0;
  India 94.2% / 3.36 / 129.1; US 94.2% / 3.36 / 123.8; CHECK PASS
official: matching 1732544 rows (100035 empty, 1632509 non-empty); candidate 1732544 rows (0 empty); PASS
row identity: US 663,106 / India 809,986 rows identical to S004; France 259,452 rows identical to S003.

## S003 = experiments/TEST_PIPELINE/submission_S003_E018C_s42_maxclaim/  (not submitted; matching sha256 in sha256.txt)
check_submission: rows 1732544, issues all 0, unique predicted ids 5543947; France 94.4% / 3.24; India 93.0% / 3.11; US 94.1% / 3.29; CHECK PASS
official: matching 1732544 rows (110645 empty, 1621899 non-empty); candidate 1732544 rows (0 empty); PASS

## S002 = experiments/SUBMISSIONS/S002_E018C_s42/  (LB 0.950650) -- validated in the previous session (see PROGRESS.md S002 entry); sha256 re-verified at shutdown: 52bef507... OK
