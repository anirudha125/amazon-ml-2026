"""WD: compile the adversarial-verification verdict on investigator D's 'France net record deficit' claim (NEW file only)."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
s1 = json.load(open(os.path.join(OUT, "WD_step1_raw.json"))); s2 = json.load(open(os.path.join(OUT, "WD_step2_v1calib.json")))
s3 = json.load(open(os.path.join(OUT, "WD_step3_unclaimed.json"))); s4 = json.load(open(os.path.join(OUT, "WD_step4_thinning.json")))
R = dict(
    claim="D: France accepted records/S1 3.293 vs US 3.372 vs GT 3.459 => France residual is FN-heavy (~20k missed records)",
    rederived_counts={cc: dict(mean=s1["sub_hist"][cc]["mean"], p0=s1["sub_hist"][cc]["p0"], p1=s1["sub_hist"][cc]["p1"]) for cc in ("France", "US", "India")},
    se_mean={cc: s2[f"se_mean_{cc}"] for cc in ("France", "US", "India")},
    pre_maxclaimer={cc: s4[f"pre_mc_{cc}"] for cc in ("France", "US", "India")},
    raw_recs_per_s1_test={cc: s1["test_raw_ratio"][cc]["rec_per_s1"] for cc in ("France", "US", "India")},
    raw_recs_per_s1_train={cc: s1["train_raw_ratio"][cc]["rec_per_s1"] for cc in ("US", "India")},
    claimed_frac_of_raw={cc: s1["sub_hist"][cc]["claimed_frac_of_raw"] for cc in ("France", "US", "India")},
    france_expected_at_US_claim_rate=s1["test_raw_ratio"]["France"]["rec_per_s1"] * s1["sub_hist"]["US"]["claimed_frac_of_raw"],
    france_gt_if_proportional_to_raw=3.459057 * s1["test_raw_ratio"]["France"]["rec_per_s1"] / s1["test_raw_ratio"]["US"]["rec_per_s1"],
    V1_net_deficit={cc: s2[f"V1_{cc}"]["net_deficit"] for cc in ("US", "India")},
    test_minus_trainGT={cc: s2[f"test_{cc}_minus_trainGT"] for cc in ("US", "India")},
    V1_loss_split={cc: {k: s2[f"V1_{cc}"][k] for k in ("macroF05", "FN_per_s1", "FP_per_s1", "loss_from_FN_only_S1", "loss_from_FP_only_S1")} for cc in ("US", "India")},
    sim_cost_pts=s2["sim"],
    thinning_test=dict(chi2_10bins=s4["thin_US_to_France"]["chi2_10bins"], France_minus_thinnedUS_pct=s4["thin_US_to_France"]["table_pct"]["France_minus_thinned"]),
    unclaimed_nearmatch={cc: {k: s3[f"test_{cc}"][k] for k in ("unclaimed_fire_near_per_s1", "claimed_fire_own_near_frac")} for cc in ("France", "US", "India")},
    V1_nearmatch={cc: {k: s3[f"V1_{cc}"][k] for k in ("r_FN_near", "r_TP_near", "FN_fire_near_per_s1", "orphan_fire_near_per_s1")} for cc in ("US", "India")},
    verdict="REFUTED (as evidence for a France FN excess): premise of identical GT generator is contradicted by raw supply and by the count-histogram shape; "
            "the deficit is produced by max-claimer FP removal; the magnitude even if all FN is ~0.7 France pts (~22% of gap), not 'a large part'.",
)
json.dump(R, open(os.path.join(OUT, "WD_deficit_verdict.json"), "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ("france_expected_at_US_claim_rate", "france_gt_if_proportional_to_raw", "thinning_test")}, indent=1, default=str))
