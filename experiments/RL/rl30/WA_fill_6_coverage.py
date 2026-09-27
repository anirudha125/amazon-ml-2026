"""RL-30 verifier WA (filler claim), PART 6 (READ-ONLY): are all S005 France accepted pairs inside the P3 France pool (so that
'not accepted' in WA_fill_2/4 means p<0.78 on a scored pair)? Output: printed only."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
acc = accepted("S005_France")
key = pd.Series(1, index=pd.MultiIndex.from_arrays([acc.s1.values, acc.rec.values]))
found = 0; n = 0
for f in sorted(glob.glob(PATHS["test_chunks"].format(country="France"))):
    z = np.load(f, allow_pickle=True); n += len(z["s1"])
    found += int(key.reindex(pd.MultiIndex.from_arrays([z["s1"], z["cand"]])).notna().sum())
print(dict(pool_rows=n, accepted=len(acc), accepted_found_in_pool=found, frac=round(found / len(acc), 5)))
