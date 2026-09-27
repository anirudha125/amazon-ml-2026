"""Stand-in for experiments/RECON-08_GOLDEN/run_reproduce.py, which is absent on this machine (restart of 2026-09-27).
src/harness.py only needs LGB_PARAMS at import time. Values = src/recon08_relational_features.LGB_PARAMS; boot.py asserts they
equal the params stored in experiments/RL/rl27_model_NEW_s42.pkl (n_jobs is always overridden by the callers)."""
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
