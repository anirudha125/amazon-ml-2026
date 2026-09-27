"""RL-30 Part 5 (investigator D): S1 -> record transformation typology (label-free, per pair).
READ-ONLY investigation. Import: from D_transform import classify_pairs
Every function uses only the text of the S1 and the record.
"""
import re, sys, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import fold, toks, akey, LEGAL, HONOR, STOP, STREET_TYPE, UNIT
from rapidfuzz.distance import Levenshtein, OSA
from rapidfuzz import fuzz

FR_CITIES = ["bordeaux", "nantes", "lille", "tourcoing", "dunkerque", "roubaix", "calais", "saint nazaire", "pessac",
             "la teste de buch", "merignac", "lege cap ferret", "pornic", "la baule escoublac", "saint herblain", "lomme",
             "hellemmes", "le clion", "sainte marie"]
REGION = set("""hauts france nouvelle aquitaine pays loire gironde nord atlantique pas
alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas
kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada hampshire jersey
mexico york carolina dakota ohio oklahoma oregon pennsylvania rhode island south north tennessee texas utah vermont virginia
washington west wisconsin wyoming null township county city
maharashtra delhi uttar pradesh karnataka tamil nadu gujarat bengal telangana haryana kerala rajasthan bihar madhya andhra orissa
odisha punjab jharkhand assam goa uttarakhand chhattisgarh himachal jammu kashmir chandigarh puducherry east""".split())
UNIT_RE = re.compile(r"\b(appt|appartement|apt|app|apartment|unit|suite|ste|flat|flmt|fl|floor|etage|bat|batiment|bureau|porte|office|shop|room|block|wing|escalier|esc|bldg|building)\b\.?\s*(?:no\.?|n|#|num)?\s*([a-z]?-?\d+[a-z]?|[a-z])\b")
LEET = {("0", "o"), ("1", "l"), ("1", "i"), ("5", "s"), ("3", "e"), ("4", "a"), ("8", "b"), ("7", "t")}


def _norm_city(a):
    a = re.sub(r"\bst\.?\s*-?\s*", "saint ", fold(a))
    return " " + re.sub(r"[^a-z0-9]+", " ", a) + " "


_LEGAL3 = [x for x in LEGAL if len(x) >= 3]
_legal_like_cache = {}
ALIAS_RE = re.compile(r"\b(dba|d b a|nee|doing business as|aka|formerly|t/a|trading as)\b")
HANDLE_RE = re.compile(r"(\.(com|fr|in|net|org|co)\b|^[@#])")


def legal_like(t):
    if t in LEGAL:
        return True
    if t not in _legal_like_cache:
        _legal_like_cache[t] = len(t) >= 3 and any(len(x) == len(t) and OSA.distance(x, t) == 1 for x in _LEGAL3)
    return _legal_like_cache[t]


def street_parts2(addr):
    """like rl30_lib.street_parts but prefers the comma component that has a digit AND a street-type word,
    and skips unit-led components (appt/apartment/unit/residence...)."""
    comps = [c.strip() for c in fold(addr).split(",") if c.strip()]
    dig = [c for c in comps if re.search(r"\d", c)]
    pick = None
    for c in dig:
        w = re.findall(r"[a-z]+", c)
        if any(x in STREET_TYPE for x in w) and not (w and w[0] in UNIT | {"apartment", "flat", "residence", "floor"}):
            pick = c; break
    if pick is None and dig:
        pick = dig[0]
    num, street = None, frozenset()
    if pick is not None:
        m = re.search(r"\d+", pick)
        num = m.group(0).lstrip("0") or "0"
        street = frozenset(t for t in re.findall(r"[a-z]+", pick) if t not in STREET_TYPE and t not in UNIT and t not in STOP and len(t) > 1)
    rest = set()
    for c in comps:
        if c is not pick:
            rest |= {t for t in re.findall(r"[a-z]{3,}", c) if t not in STOP}
    return num, street, frozenset(rest)


def parse_name(name):
    f = fold(name)
    f = re.sub(r"\b([a-z])\.(?=[a-z]\b)", r"\1", f)          # s.c.i. -> sci.
    f = re.sub(r"\b([a-z])\.(?=[a-z]\b)", r"\1", f)
    t = re.findall(r"[a-z0-9]+", f)
    core = [x for x in t if not legal_like(x) and x not in HONOR and x not in STOP]
    legal = frozenset(x for x in t if legal_like(x))
    return dict(t=tuple(t), core=frozenset(core), core_seq=tuple(core), legal=legal, raw=bool(name.strip()),
                alias=bool(ALIAS_RE.search(f)), handle=bool(HANDLE_RE.search(f.strip())), nospace=re.sub(r"[^a-z0-9]", "", f) if " " not in f.strip() else "")


def parse_addr(addr, country):
    f = fold(addr)
    if not f.strip() or f.strip() in ("<null>", "null"):
        return None
    num, street, rest = street_parts2(addr)
    if country == "France":
        na = _norm_city(addr)
        city = frozenset(c for c in FR_CITIES if f" {c} " in na)
    else:
        city = frozenset(x for x in rest if x not in REGION)
    units = frozenset(m.group(2).lstrip("-").lstrip("0") or "0" for m in UNIT_RE.finditer(f))
    bis = bool(re.search(r"\b\d+\s*(bis|ter|b)\b", f))
    return dict(num=num, street=street, city=city, units=units, akey=akey(addr), bis=bis)


def char_edit(a, b):
    """classify a typo-level token pair a->b"""
    if OSA.distance(a, b) == 1 and Levenshtein.distance(a, b) == 2:
        return "transpose"
    ops = Levenshtein.editops(a, b)
    if len(ops) == 1:
        op = ops[0]
        if op.tag == "replace":
            ca, cb = a[op.src_pos], b[op.dest_pos]
            if (ca, cb) in LEET or (cb, ca) in LEET:
                return "leet"
            return "sub1"
        if op.tag == "insert":
            ch = b[op.dest_pos]
            nb = (b[op.dest_pos - 1] if op.dest_pos > 0 else "") + (b[op.dest_pos + 1] if op.dest_pos + 1 < len(b) else "")
            return "double_ins" if ch in nb else "ins1"
        ch = a[op.src_pos]
        nb = (a[op.src_pos - 1] if op.src_pos > 0 else "") + (a[op.src_pos + 1] if op.src_pos + 1 < len(a) else "")
        return "double_del" if ch in nb else "del1"
    return "multi"


def match_tokens(A, B):
    """greedy typo/stem matching between S1-only tokens A and record-only tokens B.
    returns (list of (a,b,kind)), residual A, residual B"""
    A, B = list(A), list(B)
    cands = []
    for a in A:
        for b in B:
            if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
                cands.append((1.0, a, b, "stem"))
                continue
            s = Levenshtein.normalized_similarity(a, b)
            d = OSA.distance(a, b)
            if (len(a) >= 4 and len(b) >= 4 and (s >= 0.6 or d <= 2)) or (d <= 1 and min(len(a), len(b)) >= 2):
                cands.append((s, a, b, "typo"))
            elif min(len(a), len(b)) >= 4 and Levenshtein.normalized_similarity("".join(sorted(a)), "".join(sorted(b))) >= 0.75:
                cands.append((s * 0.5, a, b, "scramble"))
    cands.sort(key=lambda x: -x[0])
    used_a, used_b, out = set(), set(), []
    for s, a, b, k in cands:
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b); out.append((a, b, k))
    return out, [a for a in A if a not in used_a], [b for b in B if b not in used_b]


def name_type(ps, pr, s1_raw_name, rec_raw_name):
    if not pr["raw"]:
        return dict(nt="N_EMPTY")
    if not pr["t"]:
        return dict(nt="N_NONLATIN")
    cs, cr = ps["core"], pr["core"]
    A, B, C = cs - cr, cr - cs, cs & cr
    pairs, Ar, Br = match_tokens(sorted(A), sorted(B))
    ntypo = sum(k in ("typo", "scramble") for _, _, k in pairs); nstem = sum(k == "stem" for _, _, k in pairs)
    nscr = sum(k == "scramble" for _, _, k in pairs)
    legal_diff = ps["legal"] != pr["legal"]
    d = dict(ntypo=ntypo, nstem=nstem, nscr=nscr, legal_diff=legal_diff, pairs=pairs, nA=len(Ar), nB=len(Br), nC=len(C))
    js = "".join(ps["t"])
    if (pr["handle"] and len(pr["t"]) <= 3) or (len(pr["nospace"]) >= 8 and len(ps["t"]) >= 2 and fuzz.partial_ratio(pr["nospace"], js) >= 80):
        d["nt"] = "N_HANDLE"
        return d
    if pr["alias"] and not ps["alias"]:
        d["nt"] = "N_ALIAS"
        return d
    if not C and not pairs:
        ini = "".join(x[0] for x in ps["core_seq"]) if ps["core_seq"] else ""
        ini_all = "".join(x[0] for x in ps["t"])
        joined = "".join(ps["core_seq"])
        rt = pr["t"]
        rj = "".join(pr["core_seq"])
        if len(rt) >= 1 and (rj in (ini, ini_all) or "".join(rt) in (ini, ini_all)) and len(rj) >= 2:
            d["nt"] = "N_ACRONYM"
        elif len(joined) >= 6 and (joined in "".join(rt) or ("".join(rt) in joined and len("".join(rt)) >= 6)):
            d["nt"] = "N_JOIN"
        elif not cr:
            d["nt"] = "N_LEGAL_ONLY_REC"
        else:
            d["nt"] = "N_DISJOINT"
        return d
    if not Ar and not Br:
        if ntypo == 0 and nstem == 0:
            d["nt"] = "N_SAME"
        elif nstem and not ntypo:
            d["nt"] = "N_STEM"
        else:
            d["nt"] = "N_TYPO"
    elif len(Ar) == 1 and len(Br) == 1:
        d["nt"] = "N_SWAP1"; d["swap"] = (Ar[0], Br[0])
    elif Ar and not Br:
        d["nt"] = "N_DROP"; d["drop"] = tuple(Ar)
    elif Br and not Ar:
        d["nt"] = "N_ADD"; d["add"] = tuple(Br)
    else:
        d["nt"] = "N_MULTI"
    return d


def num_type(a, b):
    if a is None and b is None:
        return "H_NONE"
    if a is None:
        return "H_S1MISS"
    if b is None:
        return "H_RECMISS"
    if a == b:
        return "H_SAME"
    ia, ib = int(a[:9]), int(b[:9])
    dd = ib - ia
    if abs(dd) == 1:
        return "H_D1"
    if abs(dd) == 2:
        return "H_D2"
    if len(a) >= 2 and sorted(a) == sorted(b):
        return "H_PERM"
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return "H_DIGSUB"
    if abs(len(a) - len(b)) == 1 and Levenshtein.distance(a, b) == 1:
        return "H_DIGINDEL"
    if abs(dd) <= 10:
        return "H_D3_10_EVEN" if dd % 2 == 0 else "H_D3_10_ODD"
    return "H_GT10_EVEN" if dd % 2 == 0 else "H_GT10_ODD"


def street_type(a, b):
    if not a or not b:
        return "S_NA"
    if a == b:
        return "S_SAME"
    ja, jb = " ".join(sorted(a)), " ".join(sorted(b))
    r = fuzz.ratio(ja, jb)
    if r >= 80:
        return "S_TYPO"
    if a & b or fuzz.token_set_ratio(ja, jb) >= 80:
        return "S_PARTIAL"
    return "S_DIFF"


def addr_type(pa, pb):
    if pb is None:
        return dict(at="A_RECEMPTY", ht="H_NA", st="S_NA", ct="C_NA", ut="U_NA", same_akey=False)
    if pa is None:
        return dict(at="A_S1EMPTY", ht="H_NA", st="S_NA", ct="C_NA", ut="U_NA", same_akey=False)
    ht = num_type(pa["num"], pb["num"])
    st = street_type(pa["street"], pb["street"])
    if not pa["city"] or not pb["city"]:
        ct = "C_NA"
    else:
        ct = "C_SAME" if pa["city"] & pb["city"] else "C_DIFF"
    ua, ub = pa["units"], pb["units"]
    if not ua and not ub:
        ut = "U_NONE"
    elif ua and not ub:
        ut = "U_RECDROP"
    elif ub and not ua:
        ut = "U_RECADD"
    else:
        ut = "U_SAME" if ua & ub else "U_CHANGE"
    return dict(at="A_OK", ht=ht, st=st, ct=ct, ut=ut, same_akey=pa["akey"] == pb["akey"])


def classify_pairs(s1_ids, rec_ids, s1df, recdf, country_of_s1):
    """s1df/recdf indexed by id with name, addr, country. Returns DataFrame of types (row-aligned)."""
    pn_s, pa_s, pn_r, pa_r = {}, {}, {}, {}
    rows = []
    for s, r in zip(s1_ids, rec_ids):
        if s not in pn_s:
            nm, ad, c = s1df.at[s, "name"], s1df.at[s, "addr"], s1df.at[s, "country"]
            pn_s[s] = (parse_name(nm), nm); pa_s[s] = parse_addr(ad, c)
        if r not in pn_r:
            nm, ad = recdf.at[r, "name"], recdf.at[r, "addr"]
            c = s1df.at[s, "country"]
            pn_r[r] = (parse_name(nm), nm); pa_r[r] = parse_addr(ad, c)
        (ps, sn), (pr, rn) = pn_s[s], pn_r[r]
        nd = name_type(ps, pr, sn, rn)
        ad = addr_type(pa_s[s], pa_r[r])
        ce = [char_edit(a, b) if k == "typo" else "scramble" for a, b, k in nd.get("pairs", []) if k in ("typo", "scramble")]
        rows.append((nd["nt"], nd.get("ntypo", 0), nd.get("nstem", 0), nd.get("legal_diff", False),
                     "|".join(nd["swap"]) if "swap" in nd else "", " ".join(nd.get("drop", ())), " ".join(nd.get("add", ())),
                     ",".join(ce), ad["at"], ad["ht"], ad["st"], ad["ct"], ad["ut"], ad["same_akey"]))
    return pd.DataFrame(rows, columns=["nt", "ntypo", "nstem", "legal_diff", "swap", "drop", "add", "cedit",
                                       "at", "ht", "st", "ct", "ut", "same_akey"])


def composite(df):
    """composite flags on a typed DataFrame"""
    same_street = df.st.isin(["S_SAME", "S_TYPO"])
    out = pd.DataFrame(index=df.index)
    out["swap_same_addr"] = (df.nt == "N_SWAP1") & (df.ht == "H_SAME") & same_street
    out["swap_any"] = df.nt == "N_SWAP1"
    out["street_sub_same_num_city"] = (df.ht == "H_SAME") & (df.st == "S_DIFF") & (df.ct == "C_SAME")
    out["num_shift_same_street"] = df.ht.str.match(r"H_(D1|D2|D3|GT10|PERM|DIGSUB|DIGINDEL)") & same_street
    out["unit_change"] = df.ut == "U_CHANGE"
    out["exact_addr"] = df.same_akey
    return out
