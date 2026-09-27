"""RL-30 investigator A: compile headline numbers from A_p*_summary.json into A_results.json (READ-ONLY on everything else)."""
import os, json
OUT = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(OUT, f))) if os.path.exists(os.path.join(OUT, f)) else None
p1, p2, p2b, p2c, p2d, p2e, p2f, p2h, p2i, p2j = (L(f) for f in ["A_p1_summary.json", "A_p2_summary.json", "A_p2b_summary.json", "A_p2c_summary.json",
                                                            "A_p2d_summary.json", "A_p2e_summary.json", "A_p2f_summary.json", "A_p2h_summary.json", "A_p2i_summary.json", "A_p2j_summary.json"])
R = dict(
    part1_colocation=p1,
    part2_noise_aggregates={k: v for k, v in p2.items()},
    part2_validation_pp_vs_gt=p2c["validation_pp_vs_gt"],
    part2_france_roles=dict(counts=p2c["France_role_counts"], occ_share=p2c["France_role_occ_share"],
                            spearman_addLR_vs_pooled_idf=p2c["France_spearman_addLR_vs_idf"],
                            share_lift_quantiles=p2c.get("France_share_lift_quantiles_colocsupport>=100"),
                            top_A=p2c["top_A"], top_B=p2c["top_B"], top_M=p2c["top_M"]),
    part2_france_accepted_role_volume=p2c["France_accepted_role_volume"],
    part2_substitution_kinds={k: v for k, v in p2d.items() if "examples" not in k},
    part2_substitution_pair_lift=p2f,
    part2_extra_record_count_test=p2e,
    part2_replication_test=p2i,
    part2_V1_same_street_one_word_subs_by_kind={C: p2j[C]["counts"] for C in ("US", "India")},
    part2_decision_value_remove_content_subs=p2h,
    files=sorted(f for f in os.listdir(OUT) if f.startswith("A_")),
)
json.dump(R, open(os.path.join(OUT, "A_results.json"), "w"), indent=1, default=str, ensure_ascii=False)
print("wrote A_results.json", len(json.dumps(R, default=str)))
