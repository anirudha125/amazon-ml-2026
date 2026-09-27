"""
translit.py -- dependency-free Indic -> Latin transliteration + corrected normalization.

Rule-based, derived only from Unicode character names (no data, no labels):
  "<SCRIPT> LETTER KA" -> "ka" (consonant with inherent a), "VOWEL SIGN I" replaces the inherent a,
  "SIGN VIRAMA" kills it, word-final inherent a is dropped (schwa deletion), then a light phonetic
  simplification (long vowels collapsed, retroflex/aspirate variants folded).
Also provides normalize_fixed(): the E008 normalize() without destroying combining marks, plus
accent folding (for France) as a separate option.
"""
import re, unicodedata

INDIC = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM")
_INDIC_RE = re.compile(r"[ऀ-෿]")

_VOWELS = {"A": "a", "AA": "aa", "I": "i", "II": "ii", "U": "u", "UU": "uu", "VOCALIC R": "ri",
           "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li", "E": "e", "EE": "e", "AI": "ai",
           "O": "o", "OO": "o", "AU": "au", "SHORT E": "e", "SHORT O": "o", "CANDRA E": "e",
           "CANDRA O": "o", "OE": "o", "OOE": "o", "AW": "au", "UE": "u", "UUE": "u"}
_CONS_FIX = {"NNA": "na", "NNNA": "na", "TTA": "ta", "TTHA": "tha", "DDA": "da", "DDDHA": "dha",
             "DDHA": "dha", "LLA": "la", "LLLA": "la", "RRA": "ra", "SSA": "sha", "SHA": "sha",
             "NGA": "na", "NYA": "nya", "JNYA": "gya", "VA": "va", "YYA": "ya", "FA": "pha", "ZA": "ja",
             "QA": "ka", "KHHA": "kha", "GHHA": "gha", "DDDA": "da", "RHA": "dha", "NA": "na"}

_cache = {}

def _classify(ch):
    r = _cache.get(ch)
    if r is not None:
        return r
    name = unicodedata.name(ch, "")
    r = ("other", ch)
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0] in INDIC:
        rest = parts[1]
        if rest.startswith("LETTER "):
            key = rest[7:]
            if key in _VOWELS:
                r = ("vowel", _VOWELS[key])
            else:
                r = ("cons", _CONS_FIX.get(key, key.lower()))
        elif rest.startswith("VOWEL SIGN "):
            r = ("sign", _VOWELS.get(rest[11:], rest[11:].lower()))
        elif rest in ("SIGN VIRAMA", "SIGN PULLI") or rest.endswith("VIRAMA"):
            r = ("virama", "")
        elif rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN TIPPI"):
            r = ("nasal", "n")
        elif rest == "SIGN VISARGA":
            r = ("lit", "h")
        elif rest.startswith("DIGIT "):
            r = ("lit", str(unicodedata.digit(ch)))
        elif rest in ("SIGN NUKTA", "SIGN ADDAK", "AU LENGTH MARK", "SIGN AVAGRAHA") or "LENGTH MARK" in rest:
            r = ("skip", "")
        else:
            r = ("skip", "")
    _cache[ch] = r
    return r

def _simplify(w):
    w = w.replace("aa", "a").replace("ii", "i").replace("uu", "u").replace("ee", "e").replace("oo", "o")
    w = re.sub(r"([kgcjtdpb])h", r"\1", w)   # fold aspirates
    w = w.replace("sh", "s").replace("v", "w").replace("w", "v")
    return re.sub(r"(.)\1+", r"\1", w)

def transliterate(text, simplify=True):
    """Transliterate Indic-script characters to Latin; other characters pass through."""
    if not text or not _INDIC_RE.search(text):
        return text
    out = []
    pending_a = False   # inherent vowel of last consonant
    for ch in text:
        kind, val = _classify(ch)
        if kind == "cons":
            if pending_a:
                out.append("a")
            out.append(val[:-1] if val.endswith("a") else val)
            pending_a = val.endswith("a")
        elif kind == "sign":
            out.append(val); pending_a = False
        elif kind == "virama":
            pending_a = False
        elif kind == "skip":
            continue
        else:
            if pending_a:
                # word-final schwa deletion: drop inherent a before space / end
                if not (kind == "other" and not ch.isalnum()):
                    out.append("a")
                pending_a = False
            if kind == "nasal":
                out.append("n")
            else:
                out.append(val)
    # trailing inherent a is dropped
    s = "".join(out)
    if simplify:
        s = " ".join(_simplify(w) if w.isascii() else w for w in s.split(" "))
    return s

def normalize_fixed(text, fold_accents=False, translit=False):
    """E008 normalize() semantics but keeps combining marks (fixes Indic vowel-sign damage)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).lower()
    if translit:
        t = transliterate(t)
    if fold_accents:
        t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c) or _INDIC_RE.match(c))
        t = unicodedata.normalize("NFC", t)
    t = "".join(c if (c.isalnum() or c.isspace() or c in "-_" or unicodedata.category(c).startswith("M")) else " " for c in t)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\bnull\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def has_indic(text):
    return bool(text and _INDIC_RE.search(text))

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in ["राम मार्केटिंग प्राइवेट लिमिटेड", "आदित्य प्रॉपर्टीज एलएलपी", "लक्ष्मी एग्रो प्राइवेट लिमिटेड",
              "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ",
              "ग्लोबल इन्वेस्टमेंट प्रा. लि.", "মধ্য প্রদেশ", "ಹರಿ ಫೌಂಡೇಶನ್ ಲಿಮಿಟೆಡ್", "హైటెక్ ఇన్వెస్ట్మెంట్ లిమిటెడ్",
              "Société Générale SARL"]:
        print(s, "->", normalize_fixed(s, translit=True), "| fold:", normalize_fixed(s, fold_accents=True))
