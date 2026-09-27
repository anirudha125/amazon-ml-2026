"""RL-30 investigator A, PART 2h (READ-ONLY, no submission). Decision value of removing France content-for-content substitution
pairs from S005 France (kept_final), under the two extreme hypotheses (all FP vs all TP), assuming every other accepted pair of the
S1 is a TP and nothing is missing (ESTIMATE; per-S1 F0.5, macro over all 1,732,544 test S1).
Output: A_p2h_summary.json"""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import importlib.util
spec = importlib.util.spec_from_file_location("e", os.path.join(OUT, "A_part2e_count_test.py"))
# reuse helpers without running the module body: copy minimal pieces
import re, math
from collections import Counter
from rapidfuzz.distance import Levenshtein as LV
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")
def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s); return frozenset(re.findall(r"[a-z0-9]+", s))
def core(A): return frozenset(t for t in A if t not in LEGAL and t not in HONOR)
def typo(a, b): return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)
roles = pd.read_csv(os.path.join(OUT, "A_roles_France.csv"), keep_default_na=False, na_values=[""]).set_index("tok")
content = set(roles.index[(roles.add_LR < 0.05) & (roles.occ >= 300) & (~roles.legal.astype(bool))])
D = load("test", verbose=False)
NTOT = len(D["s1"])
fr = D["s1"][D["s1"].country == "France"].set_index("id")
REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a = accepted("S005_France"); a = a[a.kept_final]
s1n = fr.name.reindex(a.s1.values).values; rn = REC.name.reindex(a.rec.values).values
flag = defaultdict(lambda: defaultdict(int)); tot = a.groupby("s1").size()
for k in range(len(a)):
    if not isinstance(rn[k], str): continue
    A_, B_ = ntoks(s1n[k]), ntoks(rn[k])
    pa, pb = set(core(A_ - B_)), set(core(B_ - A_))
    if pa and pb:
        for _, x, y_ in sorted(((LV.normalized_similarity(x, y_), x, y_) for x in pa for y_ in pb if typo(x, y_)), reverse=True):
            if x in pa and y_ in pb: pa.discard(x); pb.discard(y_)
    if len(pa) == 1 and len(pb) == 1 and next(iter(pa)) in content and next(iter(pb)) in content:
        band = "HI" if a.p.values[k] >= 0.99 else "LO"
        flag[band][a.s1.values[k]] += 1
def f05(P, R): return 0.0 if P + R == 0 else 1.25 * P * R / (0.25 * P + R)
S = {}
for name, sets in (("LO", ["LO"]), ("HI", ["HI"]), ("HI+LO", ["HI", "LO"])):
    fl = defaultdict(int)
    for b in sets:
        for s, c in flag[b].items(): fl[s] += c
    gain_fp = loss_tp = 0.0; n_pairs = sum(fl.values())
    for s, kf in fl.items():
        m = int(tot[s]) - kf
        # H_FP: flagged are FPs, m TPs, no misses. now: P=m/(m+kf), R=1 (m>0) ; after removal: F=1 (m>0) ; m==0: gold empty -> now F=0, after F=1
        now = f05(m / (m + kf), 1.0) if m > 0 else 0.0
        gain_fp += 1.0 - now
        # H_TP: flagged are TPs. now F=1 ; after removal: P=1, R=m/(m+kf) (m==0 -> F=0)
        loss_tp += 1.0 - (f05(1.0, m / (m + kf)) if m > 0 else 0.0)
    be = loss_tp / (gain_fp + loss_tp)
    S[name] = dict(n_pairs=n_pairs, n_s1=len(fl), dLB_if_all_FP_pp=round(100 * gain_fp / NTOT, 4), dLB_if_all_TP_pp=round(-100 * loss_tp / NTOT, 4),
                   dFrance_if_all_FP_pp=round(100 * gain_fp / len(fr), 3), dFrance_if_all_TP_pp=round(-100 * loss_tp / len(fr), 3),
                   breakeven_FP_fraction=round(be, 3))
    print(name, S[name], flush=True)
json.dump(S, open(os.path.join(OUT, "A_p2h_summary.json"), "w"), indent=1)
