"""RL-30 shared definitions for the France same-address forensics (READ-ONLY investigation, 2026-09-27).
Import:  sys.path.insert(0, '<repo>/experiments/RL/rl30'); from rl30_lib import *
Every function is label-free and uses only record/S1 text of the split being analysed.
"""
import os, re, sys, unicodedata
import numpy as np, pandas as pd

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # experiments/RL
ROOT = os.path.dirname(os.path.dirname(RL))                                 # repo root
OUT = os.path.dirname(os.path.abspath(__file__))                            # experiments/RL/rl30  (write NEW files only)
sys.path.insert(0, RL); sys.path.insert(0, os.path.join(ROOT, "src"))
from rl_data import load                                                    # load("train") / load("test"): s1, s2, s3 (+gt for train)

LEGAL = set("""sarl sas sa sasu eurl sci snc scop scs sca gie selarl earl ei eirl
llc inc co corp corporation company ltd limited private pvt pc plc lp llp pllc""".split())
HONOR = set("sri shri smt mr mrs ms dr m s".split())
STREET_TYPE = set("""rue r avenue ave av bd boulevard blvd allee all chemin ch che place pl impasse imp quai route rte cours crs
square sq residence res lotissement lot hameau ham voie passage pass sentier cite parvis esplanade promenade prom rond point
street st road rd drive dr lane ln court ct circle cir way parkway pkwy trail highway hwy terrace ter plaza loop""".split())
UNIT = set("appt appartement apt app unit suite ste fl floor etage bat batiment escalier esc porte bis ter bureau box po pmb no n num".split())
STOP = set("de du des la le les l d the of and et a au aux en".split())


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("’", "'")


def toks(s):
    return re.findall(r"[a-z0-9]+", fold(s))


def name_tokens(s):
    """all name tokens (legal forms and honorifics kept)"""
    return tuple(toks(s))


def core_tokens(s):
    """name tokens minus legal forms and honorifics"""
    return frozenset(t for t in toks(s) if t not in LEGAL and t not in HONOR)


def akey(addr):
    """canonical address = sorted token multiset (same definition as RL-09/RL-27 coloc)"""
    return " ".join(sorted(toks(addr)))


def street_parts(addr):
    """(house_number, street_name_tokens, other_tokens) from the first comma component containing a digit.
    street_name_tokens exclude the number, street-type words, unit words and stopwords."""
    comps = [c.strip() for c in fold(addr).split(",") if c.strip()]
    num, street, rest = None, frozenset(), set()
    for c in comps:
        m = re.search(r"\d+", c)
        if m and num is None:
            num = m.group(0).lstrip("0") or "0"
            street = frozenset(t for t in re.findall(r"[a-z]+", c) if t not in STREET_TYPE and t not in UNIT and t not in STOP and len(t) > 1)
        else:
            rest |= {t for t in re.findall(r"[a-z]{3,}", c) if t not in STOP}
    return num, street, frozenset(rest)


def addr_words(addr):
    """full-address word set (the RL-27 / RL-04 style overlap basis: >=3 letters, stopwords removed)"""
    return frozenset(t for t in re.findall(r"[a-z]{3,}", fold(addr)) if t not in STOP)


def accepted(tag):
    """compact accepted-pair tables built from the saved prediction pickles (before max-claimer):
    tag in {'S005_France','S005_US','S005_India','S004_France'} -> DataFrame[s1, rec, p, n_claims(, kept_final)]"""
    return pd.read_pickle(os.path.join(OUT, f"accepted_{tag}.pkl"))


PATHS = dict(
    v1_meta=os.path.join(ROOT, "experiments/E024/V1_a50n10d10a/meta.npz"),          # s1_ids, s1idx, cand, y, n_gt, country, ranks
    v1_p_new=os.path.join(RL, "cache/rl27_p_NEW_V1_s42.npy"),                        # RL-27 NEW seed 42 probabilities (row-aligned)
    v1_p_base=os.path.join(RL, "cache/rl27_p_BASE_V1_s42.npy"),
    v1_rl27=os.path.join(RL, "cache/rl27_V1.npy"),                                    # the 10 RL-27 columns
    v1_LF=os.path.join(ROOT, "experiments/E024/V1_a50n10d10a/LF.npy"),               # 87 E009-D label-free columns
    t_sets=[os.path.join(ROOT, f"experiments/E024/{s}_a50n10d10a/meta.npz") for s in ("T0", "E014", "T2X")],
    test_chunks=os.path.join(ROOT, "experiments/P3/{country}/chunk_*.npz"),          # s1, cand, LF(87), rank_dense, dcos
    test_rl27=os.path.join(RL, "test_feats/{country}/rl27_chunk_*.npy"),
    s005_final=os.path.join(RL, "submissions/S005_RL27NEW_s42_mc/matching_results.tsv"),
)
NEW_TH = 0.78   # RL-27 NEW s42 OOF threshold (S005)
