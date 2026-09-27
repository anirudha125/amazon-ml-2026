"""E026 bootstrap: import src/harness.py (read-only) with its golden module redirected to golden_shim.py.
The redirect is unconditional, so E026 runs do not depend on whether anyone restores experiments/RECON-08_GOLDEN.
Usage (first import in every E026 script):  from boot import H, S2, ROOT, HERE
"""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
_GOLD = os.path.join(ROOT, "experiments", "RECON-08_GOLDEN", "run_reproduce.py")
_SHIM = os.path.join(HERE, "golden_shim.py")
_orig = importlib.util.spec_from_file_location


def _redirect(name, location=None, *a, **k):
    if name == "golden" and location is not None and os.path.abspath(str(location)) == _GOLD:
        location = _SHIM
    return _orig(name, location, *a, **k)


importlib.util.spec_from_file_location = _redirect
try:
    import harness as H
finally:
    importlib.util.spec_from_file_location = _orig
assert os.path.abspath(H.golden.__file__) == _SHIM, H.golden.__file__
import e023_stage2 as S2

_REF = {"boosting_type": "gbdt", "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": -1, "n_estimators": 300,
        "num_leaves": 31, "random_state": 42, "subsample": 0.8, "subsample_freq": 1, "verbose": -1}
assert all(H.LGB_PARAMS.get(k, "gbdt" if k == "boosting_type" else None) == v for k, v in _REF.items()), H.LGB_PARAMS
