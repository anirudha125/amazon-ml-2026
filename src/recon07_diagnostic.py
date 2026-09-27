"""
recon07_diagnostic.py
RECON-07: Frozen-Feature LightGBM Capacity Probe + Set-Shape Diagnostic

Strict Protocol:
1. Exact same 22 features from RECON-05 / RECON-06.
2. Exact same 1,994 Train S1 / 2,001 Val S1 split.
3. LightGBM fitted ONLY on Train S1.
4. Threshold tuned ONLY on Train S1 using S1 macro F0.5.
5. Evaluated ONCE on untouched 2,001 Val S1.
6. Set-Shape Diagnostic on RECON-05 LR validation scores (Group A vs Group B).
"""

import os, sys, time, math, collections, pickle
import numpy as np
import scipy.stats as stats
import lightgbm as lgb
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')

SCRATCH_DIR_05 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\b15afa6e-4b1b-492d-ba1c-a5c13926ce28\scratch"
SCRATCH_DIR_06 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\a42fa9b5-3788-49d2-bd9b-801e45dfd4c2\scratch"
SCRATCH_DIR_07 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\33c623d5-9976-4fc0-9556-7a5c10d224ce\scratch"

os.makedirs(SCRATCH_DIR_07, exist_ok=True)

CAND_CACHE_FILE = os.path.join(SCRATCH_DIR_05, "recon05_candidates.pkl")
EXACT_SPLITS_FILE = os.path.join(SCRATCH_DIR_06, "exact_splits.pkl")
OUTPUT_SCORED_FILE_05 = os.path.join(SCRATCH_DIR_06, "recon05_scored_pairs.pkl")
CACHED_FEATS_07 = os.path.join(SCRATCH_DIR_07, "recon07_features.pkl")

# Exact 22 feature extraction from RECON-05
def extract_features_for_pair(s1_name, s1_addr, cand_name, cand_addr, is_s2, is_india, rank_addr, rank_name, s1_freq):
    n_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name)
    n_jw  = distance.JaroWinkler.similarity(s1_name, cand_name)
    n_ts  = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    n_tset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0

    tok1 = set(s1_name.split())
    tok2 = set(cand_name.split())
    u = len(tok1 | tok2)
    n_jacc = (len(tok1 & tok2) / u) if u > 0 else 0.0

    log_freq = math.log1p(s1_freq)
    l1, l2 = len(s1_name), len(cand_name)
    l_diff = abs(l1 - l2)
    l_ratio = min(l1, l2) / max(l1, l2, 1)

    s1_has = 1.0 if s1_addr and s1_addr != "null" else 0.0
    c_has  = 1.0 if cand_addr and cand_addr != "null" else 0.0
    both   = 1.0 if (s1_has and c_has) else 0.0

    if both:
        a_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr)
        a_jw  = distance.JaroWinkler.similarity(s1_addr, cand_addr)
        a_ts  = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        a_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        atok1 = set(s1_addr.split())
        atok2 = set(cand_addr.split())
        au = len(atok1 | atok2)
        a_jacc = (len(atok1 & atok2) / au) if au > 0 else 0.0
    else:
        a_lev = a_jw = a_ts = a_tset = a_jacc = 0.0

    ret_addr = 1.0 if rank_addr <= 50 else 0.0
    ret_name = 1.0 if rank_name <= 10 else 0.0
    inv_r_addr = (1.0 / rank_addr) if rank_addr <= 50 else 0.0
    inv_r_name = (1.0 / rank_name) if rank_name <= 10 else 0.0

    return [
        n_lev, n_jw, n_ts, n_tset, n_jacc, log_freq, l_diff, l_ratio,
        s1_has, c_has, both, a_lev, a_jw, a_ts, a_tset, a_jacc,
        float(is_s2), float(is_india), ret_addr, ret_name, inv_r_addr, inv_r_name
    ]

def compute_entity_f05(true_matches: set, pred_matches: set) -> float:
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0
    if len(pred_matches) == 0:
        return 0.0
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    p = tp / len(pred_matches)
    r = tp / len(true_matches)
    return (1.25 * p * r) / (0.25 * p + r)

def evaluate_s1_macro_f05(s1_list, s1_dict, pred_dict):
    scores = [compute_entity_f05(s1_dict[eid]["gt"], pred_dict.get(eid, set())) for eid in s1_list]
    return float(np.mean(scores))

def process_s1_batch(batch_s1_ids, s1_dict, candidates_by_s1, name_freq):
    pairs_meta = []
    X_rows = []
    y_rows = []
    for s1_id in batch_s1_ids:
        s1_info = s1_dict[s1_id]
        s1_nm = s1_info["name"]
        s1_ad = s1_info["addr"]
        ctr = s1_info["country"]
        gt_set = s1_info["gt"]
        is_india = 1 if ctr == "India" else 0
        freq = name_freq.get(s1_nm, 1)

        cands = candidates_by_s1.get(s1_id, {})
        for cand_id, c_info in cands.items():
            c_nm = c_info["name"]
            c_ad = c_info["addr"]
            is_s2 = c_info["is_s2"]
            rank_addr = c_info["rank_addr"]
            rank_name = c_info["rank_name"]

            feats = extract_features_for_pair(
                s1_nm, s1_ad, c_nm, c_ad, is_s2, is_india, rank_addr, rank_name, freq
            )
            is_match = 1 if cand_id in gt_set else 0

            pairs_meta.append((s1_id, cand_id, is_match))
            X_rows.append(feats)
            y_rows.append(is_match)
    return pairs_meta, np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)

def main():
    print("=" * 80)
    print("RECON-07: FROZEN-FEATURE LIGHTGBM CAPACITY PROBE + SET-SHAPE DIAGNOSTIC")
    print("=" * 80)

    # 1. Load data
    if os.path.exists(CACHED_FEATS_07):
        print(f"Loading cached features from {CACHED_FEATS_07}...")
        t0 = time.time()
        with open(CACHED_FEATS_07, "rb") as f:
            feat_data = pickle.load(f)
        train_s1_ids = feat_data["train_s1_ids"]
        val_s1_ids = feat_data["val_s1_ids"]
        s1_dict = feat_data["s1_dict"]
        name_freq = feat_data["name_freq"]
        train_meta = feat_data["train_meta"]
        X_train = feat_data["X_train"]
        y_train = feat_data["y_train"]
        val_meta = feat_data["val_meta"]
        X_val = feat_data["X_val"]
        y_val = feat_data["y_val"]
        print(f"Loaded cached features in {time.time()-t0:.1f}s.")
    else:
        print("Loading candidate cache and splits...")
        with open(CAND_CACHE_FILE, "rb") as f:
            cand_data = pickle.load(f)
        s1_dict = cand_data["s1_dict"]
        name_freq = cand_data["name_freq"]
        candidates_by_s1 = cand_data["candidates_by_s1"]

        with open(EXACT_SPLITS_FILE, "rb") as f:
            train_s1_ids, val_s1_ids = pickle.load(f)

        print(f"Train S1: {len(train_s1_ids):,} | Val S1: {len(val_s1_ids):,}")
        print("Extracting features for Train S1...")
        t_feat0 = time.time()
        train_meta, X_train, y_train = process_s1_batch(train_s1_ids, s1_dict, candidates_by_s1, name_freq)
        print(f"Extracting features for Val S1...")
        val_meta, X_val, y_val = process_s1_batch(val_s1_ids, s1_dict, candidates_by_s1, name_freq)
        print(f"Extracted all features in {time.time()-t_feat0:.1f}s.")

        print(f"Caching features to {CACHED_FEATS_07}...")
        with open(CACHED_FEATS_07, "wb") as f:
            pickle.dump({
                "train_s1_ids": train_s1_ids,
                "val_s1_ids": val_s1_ids,
                "s1_dict": s1_dict,
                "name_freq": name_freq,
                "train_meta": train_meta,
                "X_train": X_train,
                "y_train": y_train,
                "val_meta": val_meta,
                "X_val": X_val,
                "y_val": y_val,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"X_train shape: {X_train.shape} | Matches: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"X_val shape:   {X_val.shape} | Matches: {y_val.sum():,} ({y_val.mean()*100:.3f}%)")

    # =========================================================================
    # EXPERIMENT A: FROZEN-FEATURE LIGHTGBM
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT A: FROZEN-FEATURE LIGHTGBM TRAINING")
    print("=" * 80)

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    print("Configuration:")
    for k, v in lgb_params.items():
        print(f"  {k}: {v}")

    t_train_start = time.time()
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train, y_train)
    training_time = time.time() - t_train_start
    print(f"LightGBM trained in {training_time:.2f}s on CPU.")

    # Threshold selection on Train S1 ONLY
    print("\nTuning threshold on Train S1 ONLY...")
    train_probs = clf.predict_proba(X_train)[:, 1]

    train_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(train_meta, train_probs):
        train_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    thresholds = np.arange(0.10, 0.96, 0.02)
    best_th = 0.50
    best_train_f05 = -1.0
    th_results = []
    for th in thresholds:
        pred_dict = {s1: {cid for cid, p, _ in cands if p >= th} for s1, cands in train_preds_by_s1.items()}
        f05 = evaluate_s1_macro_f05(train_s1_ids, s1_dict, pred_dict)
        th_results.append((th, f05))
        if f05 > best_train_f05:
            best_train_f05 = f05
            best_th = th

    print(f"Optimal Train Threshold: {best_th:.2f} (Train S1 Macro F0.5: {best_train_f05*100:.2f}%)")

    # Evaluate ONCE on untouched Val S1
    print("\nEvaluating ONCE on untouched 2,001 Validation S1...")
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_preds_by_s1 = collections.defaultdict(list)
    for (s1_id, cand_id, is_m), prob in zip(val_meta, val_probs):
        val_preds_by_s1[s1_id].append((cand_id, prob, is_m))

    # Binary evaluation at best_th
    val_pred_dict = {s1: {cid for cid, p, _ in cands if p >= best_th} for s1, cands in val_preds_by_s1.items()}
    val_macro_f05 = evaluate_s1_macro_f05(val_s1_ids, s1_dict, val_pred_dict)

    # Breakdown by country
    val_us_s1 = [s for s in val_s1_ids if s1_dict[s]["country"] != "India"]
    val_in_s1 = [s for s in val_s1_ids if s1_dict[s]["country"] == "India"]
    us_macro_f05 = evaluate_s1_macro_f05(val_us_s1, s1_dict, val_pred_dict)
    in_macro_f05 = evaluate_s1_macro_f05(val_in_s1, s1_dict, val_pred_dict)

    # Pair-level metrics
    val_preds_binary = (val_probs >= best_th).astype(int)
    val_tp = int(np.sum((val_preds_binary == 1) & (y_val == 1)))
    val_fp = int(np.sum((val_preds_binary == 1) & (y_val == 0)))
    val_fn = int(np.sum((val_preds_binary == 0) & (y_val == 1)))
    val_total_pred = val_tp + val_fp
    pair_precision = (val_tp / val_total_pred) if val_total_pred > 0 else 0.0

    retrieved_true = int(y_val.sum()) # 6,513
    cand_pair_recall = val_tp / retrieved_true
    total_gt = sum(len(s1_dict[eid]["gt"]) for eid in val_s1_ids) # 6,920
    end_to_end_recall = val_tp / total_gt

    # Singleton analysis
    singleton_s1_ids = [s for s in val_s1_ids if len(s1_dict[s]["gt"]) == 0] # 112
    singleton_fp_entities = 0
    singleton_fps_total = 0
    for s in singleton_s1_ids:
        preds = val_pred_dict.get(s, set())
        if len(preds) > 0:
            singleton_fp_entities += 1
            singleton_fps_total += len(preds)

    # High-confidence false positives
    fps_ge_80 = int(np.sum((val_probs >= 0.80) & (y_val == 0)))
    fps_ge_90 = int(np.sum((val_probs >= 0.90) & (y_val == 0)))

    print("\n--- EXPERIMENT A: LIGHTGBM VALIDATION METRICS ---")
    print(f"Chosen Threshold:              {best_th:.2f}")
    print(f"S1 Macro F0.5:                 {val_macro_f05*100:.2f}% (Baseline LR: 77.59%, Delta: {(val_macro_f05*100 - 77.59):+.2f}%)")
    print(f"US S1 Macro F0.5:              {us_macro_f05*100:.2f}% (Baseline LR: 80.08%, Delta: {(us_macro_f05*100 - 80.08):+.2f}%)")
    print(f"India S1 Macro F0.5:           {in_macro_f05*100:.2f}% (Baseline LR: 73.87%, Delta: {(in_macro_f05*100 - 73.87):+.2f}%)")
    print(f"Pair Precision:                {pair_precision*100:.2f}% (Baseline LR: 86.56%)")
    print(f"Candidate-Pair Recall:         {cand_pair_recall*100:.2f}% (Baseline LR: 75.83%)")
    print(f"End-to-End Recall:             {end_to_end_recall*100:.2f}% (Baseline LR: 71.37%)")
    print(f"TP:                            {val_tp:,} (Baseline LR: 4,939)")
    print(f"FP:                            {val_fp:,} (Baseline LR: 767)")
    print(f"Total Predicted Pairs:         {val_total_pred:,} (Baseline LR: 5,706)")
    print(f"Retrieved True Matches:        {retrieved_true:,}")
    print(f"Total Ground Truth Matches:    {total_gt:,}")
    print(f"True Singleton/No-Match S1:    {len(singleton_s1_ids):,}")
    print(f"Singleton S1 receiving >=1 FP: {singleton_fp_entities} (Baseline LR: 76)")
    print(f"Total Singleton/No-Match FPs:  {singleton_fps_total} (Baseline LR: 111)")
    print(f"FP count with score >= 0.80:   {fps_ge_80} (Baseline LR: 383)")
    print(f"FP count with score >= 0.90:   {fps_ge_90} (Baseline LR: 194)")

    # Feature importances
    feature_names = [
        "n_lev", "n_jw", "n_ts", "n_tset", "n_jacc", "log_freq", "l_diff", "l_ratio",
        "s1_has", "c_has", "both", "a_lev", "a_jw", "a_ts", "a_tset", "a_jacc",
        "is_s2", "is_india", "ret_addr", "ret_name", "inv_r_addr", "inv_r_name"
    ]
    imp_gain = clf.booster_.feature_importance(importance_type="gain")
    imp_split = clf.booster_.feature_importance(importance_type="split")
    print("\n--- LIGHTGBM TOP FEATURES BY GAIN ---")
    feat_ranks = sorted(zip(feature_names, imp_gain, imp_split), key=lambda x: x[1], reverse=True)
    for rank, (fname, gain, split) in enumerate(feat_ranks[:12], 1):
        print(f"  {rank:2d}. {fname:<18}: gain={gain:10.1f} | splits={split:4d}")

    # =========================================================================
    # EXPERIMENT B: SET-SHAPE DIAGNOSTIC
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT B: SET-SHAPE DIAGNOSTIC (RECON-05 LR SCORES)")
    print("=" * 80)

    # Load RECON-05 LR scored pairs
    with open(OUTPUT_SCORED_FILE_05, "rb") as f:
        data_05 = pickle.load(f)
    lr_val_preds_by_s1 = data_05["val_preds_by_s1"]
    lr_th = 0.58

    # Define groups
    # Group A: True singleton/no-match S1s that received >= 1 false positive at baseline threshold 0.58
    # Group B: S1s with genuine matches (len(gt) > 0)
    group_a_s1 = []
    group_b_s1 = []
    group_c_s1 = [] # Singletons with 0 FPs (clean singletons)

    for s in val_s1_ids:
        gt_len = len(s1_dict[s]["gt"])
        cands = lr_val_preds_by_s1[s]
        preds_at_th = [cid for cid, p, _ in cands if p >= lr_th]
        if gt_len == 0:
            if len(preds_at_th) > 0:
                group_a_s1.append(s)
            else:
                group_c_s1.append(s)
        else:
            group_b_s1.append(s)

    print(f"Group A (Singleton S1 with >=1 FP): {len(group_a_s1):,} S1 entities")
    print(f"Group B (Genuine-Match S1):         {len(group_b_s1):,} S1 entities")
    print(f"Group C (Clean Singleton S1, 0 FP): {len(group_c_s1):,} S1 entities")

    def compute_s1_set_stats(s1_ids, preds_dict):
        stats_dict = {
            "top_score": [],
            "second_score": [],
            "margin": [],
            "n_gt_80": [],
            "score_std": [],
        }
        for s in s1_ids:
            cands = preds_dict[s]
            if len(cands) == 0:
                top = 0.0
                second = 0.0
                margin = 0.0
                n80 = 0
                sstd = 0.0
            else:
                scores = [float(p) for _, p, _ in cands]
                sorted_s = sorted(scores, reverse=True)
                top = sorted_s[0]
                second = sorted_s[1] if len(sorted_s) > 1 else 0.0
                margin = top - second
                n80 = sum(1 for p in scores if p > 0.80)
                sstd = float(np.std(scores))
            stats_dict["top_score"].append(top)
            stats_dict["second_score"].append(second)
            stats_dict["margin"].append(margin)
            stats_dict["n_gt_80"].append(n80)
            stats_dict["score_std"].append(sstd)
        return stats_dict

    stats_A = compute_s1_set_stats(group_a_s1, lr_val_preds_by_s1)
    stats_B = compute_s1_set_stats(group_b_s1, lr_val_preds_by_s1)
    stats_C = compute_s1_set_stats(group_c_s1, lr_val_preds_by_s1)

    print("\n--- LR SET-SHAPE COMPARISON: GROUP A vs GROUP B ---")
    print(f"{'Statistic':<16} | {'Group A (Singleton FP)':<28} | {'Group B (Genuine Match)':<28} | {'Mann-Whitney U':<14} | {'AUC':<6} | {'Cohen d':<7}")
    print("-" * 110)

    stat_names = ["top_score", "second_score", "margin", "n_gt_80", "score_std"]
    display_names = {
        "top_score": "Top Score",
        "second_score": "Second Score",
        "margin": "Margin (Top-2nd)",
        "n_gt_80": "# Cands > 0.80",
        "score_std": "Score Std Dev",
    }

    diag_results = {}
    for st in stat_names:
        arrA = np.array(stats_A[st])
        arrB = np.array(stats_B[st])

        meanA, medA = np.mean(arrA), np.median(arrA)
        p25A, p75A = np.percentile(arrA, 25), np.percentile(arrA, 75)

        meanB, medB = np.mean(arrB), np.median(arrB)
        p25B, p75B = np.percentile(arrB, 25), np.percentile(arrB, 75)

        # Mann-Whitney U: testing if Group B has higher values than Group A (or vice versa)
        # Note: mannwhitneyu returns U statistic for arrB vs arrA
        u_res = stats.mannwhitneyu(arrB, arrA, alternative="two-sided")
        # AUC: probability that a randomly chosen genuine match S1 has higher value than a singleton FP S1
        auc = u_res.statistic / (len(arrA) * len(arrB))

        # Pooled std and Cohen's d
        s_pool = math.sqrt(((len(arrA)-1)*np.var(arrA, ddof=1) + (len(arrB)-1)*np.var(arrB, ddof=1)) / (len(arrA)+len(arrB)-2))
        cohen_d = (meanB - meanA) / s_pool if s_pool > 0 else 0.0

        diag_results[st] = {
            "meanA": meanA, "medA": medA, "p25A": p25A, "p75A": p75A,
            "meanB": meanB, "medB": medB, "p25B": p25B, "p75B": p75B,
            "U": u_res.statistic, "p_val": u_res.pvalue, "auc": auc, "cohen_d": cohen_d
        }

        strA = f"med={medA:.4f} [p25={p25A:.3f}, p75={p75A:.3f}] (avg={meanA:.4f})"
        strB = f"med={medB:.4f} [p25={p25B:.3f}, p75={p75B:.3f}] (avg={meanB:.4f})"
        print(f"{display_names[st]:<16} | {strA:<28} | {strB:<28} | U={u_res.statistic:9.0f} p={u_res.pvalue:.1e} | {auc:.4f} | {cohen_d:+.3f}")

    # Also compute set-shape statistics under LightGBM validation scores for direct comparison
    print("\n--- COMPLEMENTARY: LIGHTGBM SET-SHAPE COMPARISON (GROUP A_lgb vs GROUP B) ---")
    lgb_group_a_s1 = [s for s in singleton_s1_ids if len(val_pred_dict.get(s, set())) > 0]
    print(f"LightGBM Group A (Singleton S1 with >=1 FP at th={best_th:.2f}): {len(lgb_group_a_s1)} entities")
    stats_A_lgb = compute_s1_set_stats(lgb_group_a_s1, val_preds_by_s1)
    stats_B_lgb = compute_s1_set_stats(group_b_s1, val_preds_by_s1)

    for st in stat_names:
        arrA = np.array(stats_A_lgb[st])
        arrB = np.array(stats_B_lgb[st])
        if len(arrA) > 0:
            meanA, medA = np.mean(arrA), np.median(arrA)
            meanB, medB = np.mean(arrB), np.median(arrB)
            u_res = stats.mannwhitneyu(arrB, arrA, alternative="two-sided")
            auc = u_res.statistic / (len(arrA) * len(arrB))
            print(f"LGBM {display_names[st]:<16} | A: med={medA:.4f}, mean={meanA:.4f} | B: med={medB:.4f}, mean={meanB:.4f} | AUC={auc:.4f}")

    # Save summary dictionary
    summary_out = {
        "lr_baseline": {
            "macro_f05": 77.59,
            "us_macro_f05": 80.08,
            "in_macro_f05": 73.87,
            "precision": 86.56,
            "cand_recall": 75.83,
            "e2e_recall": 71.37,
            "tp": 4939,
            "fp": 767,
            "total_pred": 5706,
            "th": 0.58,
            "singleton_fp_entities": 76,
            "singleton_fps_total": 111,
            "fps_ge_80": 383,
            "fps_ge_90": 194,
        },
        "lightgbm": {
            "macro_f05": val_macro_f05 * 100,
            "us_macro_f05": us_macro_f05 * 100,
            "in_macro_f05": in_macro_f05 * 100,
            "precision": pair_precision * 100,
            "cand_recall": cand_pair_recall * 100,
            "e2e_recall": end_to_end_recall * 100,
            "tp": val_tp,
            "fp": val_fp,
            "total_pred": val_total_pred,
            "th": best_th,
            "singleton_fp_entities": singleton_fp_entities,
            "singleton_fps_total": singleton_fps_total,
            "fps_ge_80": fps_ge_80,
            "fps_ge_90": fps_ge_90,
            "training_time": training_time,
        },
        "set_shape_lr": diag_results,
    }
    with open(os.path.join(SCRATCH_DIR_07, "recon07_summary.pkl"), "wb") as f:
        pickle.dump(summary_out, f)
    print("\nRECON-07 Execution Complete!")

if __name__ == "__main__":
    main()
