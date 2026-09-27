# E027_FR: France-only post-process patch on S006 (mirror census)

**Verdict: no bucket clears the bar. Do not submit.** The census gives clear results, and the 5x bars pass for buckets A, B and D. But
none of the three reaches the one-slot rule of expected LB >= +0.0005, even with a false share far above what the census supports. A
validated hedge with bucket D only (legal-form flip together with a +k house-number shift) is at `submission_S007FR_patch/`. It is **not recommended**, because its
expected gain is +0.00012 LB.

Run: 2026-09-27 14:12-14:27 UTC. CPU only (12 procs, nice 10), no GPU. S006 SHA256SUMS was checked before and after: OK (`s006_sha_before.log`, `s006_sha_after.log`).

## Method
- Parsing (`e027_pairs.py`): lowercase and NFKD accent strip, then drop '.', so that s.a.s becomes sas. Tokens are [a-z0-9]+. Name stopwords (de/du/des/la/le/les/l/d/et/...) are dropped
  and the name is compared as a token multiset. The house number is the first `\d{1,5}[a-z]?` token found when scanning comma-components left to right.
  The street is that component with the number removed, abbreviations expanded (r->rue, av->avenue, bd->boulevard, ...) and stopwords, no, bis and ter removed.
  Street-equal means exact equality of the two street strings. delta = cand_number - s1_number.
- Pool: the **full** France union pool `experiments/P3/France/chunk_*.npz` has 33,180,559 pairs. 16,006,222 of them are street-equal, and the census uses the
  pairs in the delta window [-25, 25].
- Pre-registration: `preregistration.json` (sha256 be0ce1f2...0412) was written at 14:18 UTC. It came after the token-frequency scan
  (`vocab.log`, which pools all deltas and uses no S006 match information) and before any census by offset or count of S006 rows that would move.

## Classes (pre-registered)
- **A generic-role sub**: a single substitution where both tokens are in ROLE = {club, clubs, comite, centre, amicale, amis, groupement, groupe, societe,
  association, cercle, union, foyer, federation, collectif, section, maison, institut, ligue, entente, syndicat, cooperative, conseil}.
- **B filler/legal attach**: one token added or dropped from FILLER = {cie, compagnie, co, fils, freres, frs, associes, associe, developpement, dev, groupe,
  group, france} or LEGAL = {sarl, sas, sa, eurl, sci, sasu, snc, ei, eirl, sarlu, selarl, scp, scm, scop, gie, sca, 5arl, 5as}.
- **C rare sub**: min document frequency < 200. Left alone. **D legal flip**: both tokens are in LEGAL. **X content sub**: other substitutions, census only.
  Mixed role/filler swaps such as club/fils or club/france fall into X under this definition. That is why A is much smaller than the "generic 64%" slice in the brief.

## Census (full France pool, street-equal; bg = mean n(k) over k in {-2,-1,+6,+8,+10})
| bucket | n(0) | bg/offset | n(0)/bg | shift sum | mirror sum | shift/mirror |
|---|---|---|---|---|---|---|
| A generic-role sub | 7,651 | 143.6 | 53.3 | 13,132 | 1,076 | **12.2** |
| B filler/legal attach | 118,081 | 89.0 | **1,327** | 83,812 | 796 | 105.3 |
| C rare sub | 34,099 | 845.8 | 40.3 | 10,473 | 7,122 | 1.47 |
| D legal flip | 840 | 73.0 | 11.5 | 48,803 | 648 | **75.3** (per-k 46-126x) |
| X content sub | 64,849 | 990.8 | 65.5 | 77,284 | 8,214 | 9.4 |
| Z exact name | 375,117 | 107.2 | 3,499 | 738 | 1,188 | 0.62 |

Per-offset counts for A, from 0 through neighbours, shifts and mirrors: n(0)=7651. The neighbours are -1:139, -2:146, +6:171, +8:127, +10:135. The shifts are +1:1356, +2:1403, +3:1347,
+5:1251, +7:1241, +21:1272. The mirrors are -3:129, -5:103. The top A pairs, as n(0) with shift/mirror in brackets: club/groupe 533 (8.9), amicale/club 405 (18.2), club/comite 370 (13.5),
amicale/groupe 266 (8.7), centre/club 260 (10.1), comite/groupe 234 (7.5), amis/club 233 (12.3). The full table is in `census.md` and `census.json`.

**Reading.** The generator makes decoys by shifting the house number (+k) and changing one name token at the same time: a legal flip, a role swap or a filler attach.
Exact names show no shift pile-up. At delta 0, every class, including A, spikes about 50x above a flat background. By the census logic this marks a true variant, not a decoy.

## Bars (applied mechanically)
| bucket | 5x bar | links moved | S1 changed | S1 skipped (would empty) | census false share | dLB at f=31/50/70% | dLB at census estimate | one-slot (>= +0.0005) |
|---|---|---|---|---|---|---|---|---|
| A drop (role sub, delta 0, in S006) | clears (shift/mirror 12.2) | 3,356 of 3,584 | 3,307 | 225 | 1.9% (bg/n0); 15.3% (shift excess/n0) | +0.00004 / +0.00014 / +0.00025 | **-0.00012 / -0.00005** | FAIL |
| D drop (legal flip at shift k, in S006) | clears (all k, 46-126x) | 877 of 923 | 873 | 46 | 98.5% (1 - bg/shift) | +0.00001 / +0.00004 / +0.00007 | **+0.00012** | FAIL |
| B add (filler/legal, delta 0, unique, unclaimed, in candidate_pairs) | clears (n0/bg 1,327) | 421 (94 onto empty S1) | 421 | - | 0.08% for the class (not credible for the model-rejected residue, p6 median 0.27) | +0.00002 / -0.00002 / -0.00006 | +0.00008 | FAIL |

- dF and dLB use the exact per-S1 F0.5 (`e027_gain.py`). The assumptions: the untouched S006 links are true, the GT equals the true links in the list, and moved links are false independently with
  probability f. For B, f is the false share of the added links. dLB = 0.14975 x dF_France. As a check, A at 70% false gives dF_France = +0.170 pt. The linear rule in the brief,
  0.7 x 3356/15000, gives +0.157 pt, so the two agree.
- A in pt, France dF at 31/50/70%: +0.026 / +0.096 / +0.170. D: +0.007 / +0.027 / +0.048, and +0.078 at 98.5%. B: +0.011 / -0.013 / -0.039.
- **Best case, all three buckets at their most favourable f: about +0.00045 LB, still below +0.0005.** At the false share the census supports, bucket A
  loses LB (-0.00012 to -0.00005). The 5x rule for A fires only because role swaps pile up at shifted numbers. That pile-up happens away from delta 0, where the drop would be applied.
- Not pre-registered, reported only: of the 3,584 accepted A delta-0 pairs, 2,932 belong to an S1 that also has an exact-name or B-class match at the same address.
  This hints that some of them are extra decoys. Even at 70% false, though, A gives only +0.00025.
- Also not pre-registered: across all classes, S006 accepts about 3,300 pairs at shift offsets against about 1,370 at mirror offsets. The largest excesses are D 923 vs 5,
  X 656 vs 23, B 708 vs 231, A 295 vs 2 and Y 129 vs 5. The total excess is about 2k links, and at most about +0.0003 LB even if all of them are false. This is not enough for a slot either.

## Hedge built: `submission_S007FR_patch/` (bucket D only, NOT recommended)
- It is S006 with 877 France links removed across 873 S1. No links were added, and no S1 becomes empty (46 S1 were skipped by the empty guard). Row order is preserved.
- Checked row by row: all 1,473,092 US and India rows are byte-identical to S006 in both files. candidate_pairs.tsv is byte-identical to S006, France rows included (byte copy).
  In matching_results.tsv, 873 France rows differ.
- tools/check_submission.py: CHECK PASS. It saw 1,732,544 rows, all issue counters 0, and 5,827,039 unique predicted ids, which is exactly S006's 5,827,916 minus 877.
  The official `student_resource/utils/validate_submission.py` in default mode: PASS, with 99,476 empty and 1,633,068 non-empty rows.
- sha256: matching_results.tsv `738d5e17c9e35e213860d7ac8fcf0b29a173737dd369d5e5577a166a7ca29b16`. candidate_pairs.tsv
  `1e95e08725eeb3d9425345758a01cca70573a582d4e96ed1b388033c99d0f0f3`, the same as S006. `sha256sum -c SHA256SUMS` passes.

## Files
`preregistration.json`, `census.json` / `census.md`, `moves.json` (the per-bucket pair lists), `gain.json`, `*.log`, and scripts `e027_{pairs,vocab,census,moves,gain,build}.py`.
The intermediate files pairs_streq.npz (3.7 GB), cls_win.npy and idx_win.npy were deleted to save disk. `e027_pairs.py` followed by `e027_census.py` regenerates them in about 3 minutes.

## Follow-up (14:30-14:36 UTC): shift-set census of S006-ACCEPTED links, and the S007FR_shift hedge. **POST-HOC**
This is **POST-HOC**. It is not in `preregistration.json`, and it extends the pre-registered "legal-flip riding a +k shift" rule to the other one-token classes.
The intermediates were rebuilt with `e027_pairs.py` and `e027_census.py`, and the census.json that came out was byte-identical to the original. Script: `e027_shift.py`; output: `shift.json`.
Street-equal France pairs only. The shift set is {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21} and the mirror is {-k}. Implied false share = 1 - accepted_mirror/accepted_shift.

| class | accepted_shift | accepted_mirror | pool_shift | pool_mirror | pool ratio | accepted ratio | implied false share | p6 q10/25/50/75/90 of accepted shift | selected |
|---|---|---|---|---|---|---|---|---|---|
| A role sub | 295 | 2 | 13,132 | 1,076 | 12.2 | 147.5 | 99.3% | .752/.814/.886/.940/.969 | yes |
| B filler/legal attach | 708 | 231 | 83,812 | 796 | 105.3 | **3.06** | 67.4% | .770/.849/.924/.966/.985 | no (accepted ratio < 5) |
| C rare sub | 43 | 69 | 10,473 | 7,122 | 1.47 | 0.62 | <0 | .789/.872/.936/.967/.989 | no |
| D legal flip | 923 | 5 | 48,803 | 648 | 75.3 | 184.6 | 99.5% | .748/.789/.853/.921/.958 | yes |
| X content sub | 656 | 23 | 77,284 | 8,214 | 9.4 | 28.5 | 96.5% | .749/.804/.877/.936/.969 | yes |
| Y other attach | 129 | 5 | 82,501 | 273 | 302.2 | 25.8 | 96.1% | .785/.857/.920/.961/.977 | yes |
| Z exact name | 463 | 934 | 738 | 1,188 | 0.62 | 0.50 | <0 | .873/.934/.971/.988/.995 | no |
| W multi-token | 103 | 104 | 2,118,306 | 1,830,490 | 1.16 | 0.99 | ~0 | .755/.816/.908/.983/.994 | no |

`shift.json` also has the counts for each offset.

**Hedge `submission_S007FR_shift/`** (`e027_build_shift.py`) drops every accepted shift-set link of classes A, D, X and Y. The empty-list guard is on:
124 S1 were skipped because their lists would have emptied (A 19, D 47, X 45, Y 14 links kept).

| class | links removed | S1 touched | f (implied) | dF France (pt) | dLB |
|---|---|---|---|---|---|
| A | 276 | 276 | 99.3% | +0.023 | +0.000035 |
| D | 876 | 872 | 99.5% | +0.079 | +0.000118 |
| X | 611 | 609 | 96.5% | +0.049 | +0.000073 |
| Y | 115 | 115 | 96.1% | +0.010 | +0.000015 |
| **total (joint, exact per-S1)** | **1,878** | **1,864** | per class | **+0.161** | **+0.000241** |
| total at uniform f = 31/50/70% | | | | +0.015 / +0.056 / +0.100 | +0.000023 / +0.000084 / +0.000149 |

- Validation: 1,864 France rows differ, and all 1,473,092 US and India rows are byte-identical in both files. candidate_pairs.tsv is byte-identical to S006, and no S1 becomes empty.
  check_submission: CHECK PASS (0 issues, 5,826,038 unique predicted ids, which is 5,827,916 - 1,878). Official validator (default mode): PASS, with 99,476 empty and 1,633,068 non-empty rows.
  sha256: matching_results.tsv `5b3fec8093d45c6e4e7bfbbe2e3929bebeb9b58ffacb237db72ae8245b6ee4ed`, candidate_pairs.tsv
  `1e95e08725eeb3d9425345758a01cca70573a582d4e96ed1b388033c99d0f0f3`. S006 re-verified OK (`s006_sha_after2.log`), and submission_S007FR_patch was left untouched (sha OK).
- **Verdict unchanged: do not use the single slot for it.** The expected gain is +0.00024 LB, which is below +0.0005. The false shares are strongly supported, since mirror
  counts are nearly zero, but there are too few links to matter. If a spare slot exists, S007FR_shift is the better of the two hedges (+0.00024 versus +0.00012).
  The intermediates were deleted again after the build.
