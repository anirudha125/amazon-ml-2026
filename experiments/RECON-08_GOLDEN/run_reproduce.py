"""
RECON-08_GOLDEN compatibility shim (recreated 2026-09-27 with explicit user approval; the original directory was deleted
by an unrecoverable cleanup -- no copy existed in git, trash or elsewhere on disk).

This is NOT the original RECON-08 pipeline and does not reconstruct it. It restores only what current code needs at import:
  LGB_PARAMS          recovered EXACTLY from saved artifacts:
                        experiments/SUBMISSIONS/S002_E018C_s42/model_E018C_s42.pkl['lgb_params'] (= LGB_PARAMS + reg_lambda 1.0)
                        experiments/E024/model_D2b_union_rrUb_big.pkl base/stage2 LightGBM get_params() (consistent)
  compute_entity_f05  copied verbatim from the original run_reproduce.py (read in the RL session on 2026-09-26)
Everything else below is an explicit stub. The legacy feature builders (blocks B/C/D/E, OOF base scores) and the golden
cache files CANNOT be recovered; any code path that calls them fails loudly with RuntimeError / FileNotFoundError.
Current production paths (RL-27 seeds, E023/E024 stage-2 refits on cached features, P3 scoring on cached test chunks)
do not call them. Rebuilding E009-D features for NEW pairs (src/pair_features.py) is no longer possible.
"""
import os

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))

LGB_PARAMS = {
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

# Original golden cache files were deleted; these names make any attempt to open them fail with an explicit path.
CACHED_FEATS_07 = os.path.join(GOLDEN_DIR, "cache", "DELETED__cached_feats_07__unrecoverable.pkl")
CAND_CACHE_FILE = os.path.join(GOLDEN_DIR, "cache", "DELETED__candidates__unrecoverable.pkl")
ADDR_CACHE_FILE = os.path.join(GOLDEN_DIR, "cache", "DELETED__address_counts__unrecoverable.pkl")
TOKEN_IDF_FILE = os.path.join(GOLDEN_DIR, "cache", "DELETED__token_idf__unrecoverable.pkl")


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


def _unrecoverable(name):
    def _stub(*args, **kwargs):
        raise RuntimeError(f"golden.{name} is unavailable: the original RECON-08_GOLDEN code was deleted and is not "
                           f"reconstructed (compatibility shim). Current production paths must not call it; use cached features "
                           f"or the vectorised equivalents in src/e023_stage2.py.")
    _stub.__name__ = name
    return _stub


generate_oof_and_val_base_scores = _unrecoverable("generate_oof_and_val_base_scores")
extract_block_a_competition = _unrecoverable("extract_block_a_competition")
extract_block_b_retrieval = _unrecoverable("extract_block_b_retrieval")
extract_block_c_idf = _unrecoverable("extract_block_c_idf")
extract_block_d_address = _unrecoverable("extract_block_d_address")
extract_block_e_cross_source = _unrecoverable("extract_block_e_cross_source")
evaluate_s1_macro_f05 = _unrecoverable("evaluate_s1_macro_f05")
