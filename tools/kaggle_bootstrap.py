"""
kaggle_bootstrap.py -- one-line setup for the Kaggle GPU notebook (lives at /kaggle/working/amlc/kaggle_bootstrap.py).

In a notebook cell:
    exec(open("/kaggle/working/amlc/kaggle_bootstrap.py").read())

Sets cwd/sys.path to the mirrored project, points the HF cache outside /kaggle/working (so model weights are not
saved as notebook output), prints the GPU state, and defines load_pairs() for pair-level diagnostics.
Nothing here trains, downloads a model, or touches the test split.
"""
import os, sys, json, pickle
ROOT = "/kaggle/working/amlc"
os.chdir(ROOT)
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch
print(f"[amlc] root={ROOT} torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"gpus={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]} HF_HOME={os.environ['HF_HOME']}")


def load_pairs(which="val"):
    """Pair table with RAW text. which: 'val' (2,001 val S1, frozen pool), 'train_orig' (1,994 train S1, frozen pool),
    'train_new' (E014 +10,000 S1, engine pools; streams the train TSVs, ~1-2 min).
    Columns: s1_id, cand_id, label, country, is_s2, s1_name, s1_addr, cand_name, cand_addr."""
    import harness as H
    D = H.load_e008(verbose=False)
    if which in ("val", "train_orig"):
        meta = D["val_meta"] if which == "val" else D["train_meta"]
        raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
        is_s2 = lambda s, c: D["cands"][s][c]["is_s2"]
    elif which == "train_new":
        import build_pools as BP
        C = pickle.load(open(os.path.join(ROOT, "experiments", "E014", "e014_feats_10000_translit.pkl"), "rb"))
        meta = C["meta_new"]
        raw = BP.fetch_raw("train", {s for s, _, _ in meta} | {c for _, c, _ in meta})
        is_s2 = lambda s, c: None   # engine pools not expanded here; derive from pools/*.pkl if needed
    else:
        raise ValueError(which)
    rows = [(s, c, int(y), raw[s][2], is_s2(s, c), raw[s][0], raw[s][1], raw[c][0], raw[c][1]) for s, c, y in meta]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "label", "country", "is_s2", "s1_name", "s1_addr", "cand_name", "cand_addr"])
