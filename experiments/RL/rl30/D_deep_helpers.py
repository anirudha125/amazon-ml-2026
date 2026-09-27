# RL-30 Part 5 (investigator D): shared post-typing helpers, exec'd by D_counts.py / D_final.py after loading
# fr, us, ind, v (enriched tables). Defines address relation 'arel' and swap class 'swapcls'.
import numpy as np, pandas as pd

NOISE = {"France": ["fils", "services", "associes", "developpement", "france", "compagnie", "fka", "labs", "one"],
         "US": ["incorporated", "center", "services", "www", "com", "service", "fka", "partners"],
         "India": ["center", "services", "service", "www", "com", "partners", "fka"]}   # top N_ADD tokens on accepted test pairs (D_deep.log)


def _is_sub(b, a):
    it = iter(a)
    return all(ch in it for ch in b)


def is_abbrev(a, b):
    """b is an abbreviation of a (or vice versa): short, same first letter, subsequence"""
    if not a or not b:
        return False
    s, l = (b, a) if len(b) <= len(a) else (a, b)
    return len(s) <= 3 and s[0] == l[0] and _is_sub(s, l)


def addr_rel(d):
    same_st = d.st.isin(["S_SAME", "S_TYPO"])
    return np.select([d.same_akey, (d.ht == "H_SAME") & same_st, d.hnum_shift & same_st, (d.ht == "H_SAME") & (d.st == "S_DIFF"),
                      d.at == "A_RECEMPTY", d.ht == "H_RECMISS"],
                     ["exact_addr", "same_num_street", "num_shift_same_street", "street_sub_same_num", "rec_addr_empty", "rec_num_missing"],
                     "other")


def swap_class(d, cc):
    ab = np.array([is_abbrev(a, b) for a, b in zip(d.sw_a, d.sw_b)])
    return np.where(d.nt != "N_SWAP1", "", np.where(d.sw_b.isin(NOISE[cc]), "to_noise_suffix",
                    np.where(ab, "abbrev", np.where(d.swap_kind == "WORD", "content_word", "garble_or_rare"))))


for _d, _cc in ((fr, "France"), (us, "US"), (ind, "India")):
    _d["arel"] = addr_rel(_d); _d["swapcls"] = swap_class(_d, _cc)
v["arel"] = addr_rel(v); v["swapcls"] = ""
for _cc in ("US", "India"):
    _k = (v.country == _cc).values
    v.loc[_k, "swapcls"] = swap_class(v[_k], _cc)
