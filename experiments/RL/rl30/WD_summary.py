"""WD: compile verifier summary from WD_1..WD_6 result files -> rl30/WD_summary.json"""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
r1 = json.load(open(os.path.join(OUT, "WD_1_results.json"))); r3 = json.load(open(os.path.join(OUT, "WD_3_shapecalib_results.json")))
r5 = json.load(open(os.path.join(OUT, "WD_5_results.json"))); r6 = json.load(open(os.path.join(OUT, "WD_6_results.json")))
S = dict(
  reproduced_exact_from_D_pickles=dict(word_swaps=r1["a_D_repro"]["n_word_swaps"], distinct=r1["a_D_repro"]["n_distinct"], share=r1["a_D_repro"]["share"],
                                       reverse_mass_measured=r1["a_D_repro"]["reverse_mass"], reverse_mass_claimed=0.76, top=r1["a_D_repro"]["top"][:4]),
  independent_recount_France=dict(word_swaps=r1["b_independent"]["France"]["n_word_swaps"], distinct=r1["b_independent"]["France"]["n_distinct"],
                                  reverse_mass=r1["b_independent"]["France"]["reverse_mass"], overlap=r1["b_independent"]["France"]["overlap"]),
  shape_test_V1_calibration=dict(corr=r3["V1_corr_f_FP"], principle=r3["V1_principle"],
                                 true_classes_high_f={k: r3["V1_class_f_vs_FP"][k] for k in ("N_ADD|exact", "N_DISJOINT|exact", "N_ALIAS|any", "N_ACRONYM|any", "N_DROP|exact")}),
  appended_tokens_train_India=r5["train"]["India"]["appended_tokens"], appended_tokens_train_US=r5["train"]["US"]["appended_tokens"],
  classes_train=dict(US=r5["train"]["US"]["by_class"], India=r5["train"]["India"]["by_class"]),
  V1_model_in_class=dict(US=r5["train"]["US"]["V1_model_in_class"], India=r5["train"]["India"]["V1_model_in_class"], US_total=r5["train"]["US"]["V1_total"], India_total=r5["train"]["India"]["V1_total"]),
  classes_test={c: r5[f"test_{c}"]["by_class"] for c in ("France", "US", "India")},
  rival=r6,
  stakes_estimated=dict(n_s1=13902, share_France_S1=round(13902 / 259452, 4), gain_if_decoy_per_S1=0.2659, loss_if_true_per_S1=0.1228,
                        breakeven_f=round(0.1228 / (0.2659 + 0.1228), 3), France_pts_f1=round(100 * 0.2659 * 13902 / 259452, 3), France_pts_f0=round(-100 * 0.1228 * 13902 / 259452, 3)),
)
json.dump(S, open(os.path.join(OUT, "WD_summary.json"), "w"), indent=1, default=str)
print(json.dumps(S["stakes_estimated"]))
