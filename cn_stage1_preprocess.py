# CN PS3 - Stage 1: labeling sample, address parsing (P1-P4), integrity signals.

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as _rf

    def sim(a, b):
        return _rf.ratio(a, b)
    FUZZ_BACKEND = "rapidfuzz"
except Exception:  
    from difflib import SequenceMatcher

    def sim(a, b):
        return 100.0 * SequenceMatcher(None, a, b).ratio()
    FUZZ_BACKEND = "difflib (slower; pip install rapidfuzz)"

DATA_DIR = "."
OUT_DIR = "ps3_out"
SEED = 42
N_LABEL = 100
N_LABELERS = 3
LABELS_FILLED = "label_sample_filled.csv"
EXTRA_DICT = "dict_extra.json"
EVAL_HALF = "dev"

FUZZY_ACCEPT = 88.0          
FUZZY_MARGIN = 5.0           
DBSCAN_EPS_M = 75
DBSCAN_MIN_SAMPLES = 10
HUB_MIN_VISITS = 50
HUB_MIN_SPREAD_M = 1500.0    
PHOTO_ADDR_THRESHOLD = 3     
RISK_W = {"hub": 0.40, "photo": 0.30, "tiny": 0.15, "start": 0.15}

def hdr(title):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)

def find_file(name):
    root = Path(DATA_DIR)
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root.resolve()}")
    return hits[0]

def validate_all_datasets_present():
    expected = [
        "accounts.csv", "addresses.csv", "agents.csv", "baseline_geocodes.csv",
        "field_visits.csv", "landmarks_poi.csv", "lenders.csv", "localities.csv",
        "splits.csv", "surveyed_addresses.csv", "towns.csv", "visit_gps_points.csv"
    ]
    for name in expected:
        find_file(name)

def load(name, **kw):
    p = find_file(name)
    return pd.read_csv(p, **kw)

def pick_col(df, candidates, what, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find a {what} column. Columns present: {list(df.columns)}")
    return None

def detect_text_col(df):
    for c in ["address_text", "raw_address", "address_raw", "address", "full_address", "address_line", "text"]:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object and not c.endswith("_id")]
    best = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    print(f"[warn] guessed the address text column: '{best}' (columns: {list(df.columns)})")
    return best

def dist(x1, y1, x2, y2):
    return np.hypot(np.asarray(x1, float) - np.asarray(x2, float), np.asarray(y1, float) - np.asarray(y2, float))

def err_stats(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p90=np.nan, max=np.nan)
    return dict(n=len(a), mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90), max=a.max())

def PH(phrase):
    return r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"

ABBR = {
    "nr": "near", "opp": "opposite", "oppo": "opposite", "oppt": "opposite",
    "bhd": "behind", "beh": "behind",
    "govt": "government", "gov": "government", "gvt": "government", "sch": "school",
    "ngr": "nagar", "nagr": "nagar", "ngar": "nagar", "nager": "nagar",
    "colny": "colony", "coloney": "colony", "colni": "colony", "col": "colony",
    "lyt": "layout", "layot": "layout", "lout": "layout",
    "ext": "extension", "extn": "extension",
    "rd": "road", "hno": "house no", "dno": "door no",
    "mohla": "mohalla", "muhalla": "mohalla",
}

VOCAB = ["cross", "main", "road", "street", "gali", "ward", "block", "near", "opposite", "behind", "beside",
         "temple", "school", "masjid", "mosque", "church", "colony", "nagar", "layout", "extension",
         "dairy", "store", "shop", "office", "stop", "tank", "hall", "park", "pump", "bunk", "petrol",
         "medical", "ration", "community", "government", "village", "house"]
VOCAB_SET = set(VOCAB)

GENERIC = {"nagar", "colony", "layout", "extension", "puri", "mohalla", "vihar", "enclave", "park", "road",
           "street", "town", "city", "garden", "gardens", "block", "sector", "stage", "phase", "village",
           "camp", "basti", "east", "west", "north", "south"}

REL_PRE_TOKENS = {"near", "opposite", "behind", "beside", "adjacent"}
TYPE_HEAD_TOKENS = {"temple", "mandir", "gudi", "school", "masjid", "mosque", "church", "park", "tank",
                    "dairy", "shop", "store", "hall", "office", "stop", "stand", "pump", "bunk", "garden"}

REL_WORDS = ["near", "opposite", "behind", "beside", "next to", "close to", "adjacent to", "opp to",
             "hattira", "hatthira", "hathira", "hattirada", "ke paas", "paas", "samne", "ke samne",
             "ke peeche", "peeche", "ke pichhe", "pichhe", "pakka", "pakkada", "pakkadalli", "edurige",
             "eduru", "edurugade", "mundhe", "hinde", "ಹತ್ತಿರ", "ಎದುರು", "ಪಕ್ಕ", "ಹಿಂದೆ"]
REL_RE = re.compile(r"(?<![a-z0-9])(?:%s)(?![a-z0-9])" % "|".join(
    re.escape(w) for w in sorted(REL_WORDS, key=len, reverse=True)))

LM_TERMS = {
    "ration_shop": [r"ration (?:shop|store|depot|dukan|dukaan|angadi)", r"(?:rashan|rasan) (?:dukan|dukaan|shop|store)",
                    r"fair price (?:shop|store)", r"padithara angadi", "ಪಡಿತರ"],
    "bus_stop": [r"bus (?:stop|stand|stn|station|nildana|nilduna|nildhana|shelter)", r"busstop", "ಬಸ್ ನಿಲ್ದಾಣ"],
    "govt_school": [r"(?:government|sarkari|sarakari) (?:high |primary |higher primary |urdu )?(?:school|shaale|shala|shale|pathshala|paathshala|vidyalaya)",
                    "ಸರ್ಕಾರಿ ಶಾಲೆ"],
    "masjid": [r"masjid", r"mosque", r"masidi", r"masjeed", r"masjith", "ಮಸೀದಿ"],
    "hanuman_temple": [r"(?:hanuman|haanuman|hanumaan|hanumantha|anjaneya|maruti)(?: swamy| swami| ji ka| ji| ka)? (?:temple|mandir|gudi|devasthana)",
                       "ಹನುಮಾನ್ ದೇವಸ್ಥಾನ", "ಆಂಜನೇಯ ದೇವಸ್ಥಾನ", "ಆಂಜನೇಯ ಗುಡಿ"],
    "ganesha_temple": [r"(?:ganesh|ganesha|ganapati|ganapathi|vinayaka|vinayak|pillayar)(?: ji ka| ji| ka)? (?:temple|mandir|gudi|devasthana)",
                       "ಗಣೇಶ ದೇವಸ್ಥಾನ", "ಗಣೇಶ ಗುಡಿ"],
    "medical_store": [r"medical (?:store|stores|shop|hall)", r"medicals", r"pharmacy", r"chemist",
                      r"(?:dawai|dawa|davai) (?:ki |ka )?(?:dukan|dukaan|shop)", r"medicine (?:shop|store)",
                      r"(?:aushadhi|oushadhi|aushadh) angadi", "ಮೆಡಿಕಲ್"],
    "water_tank": [r"water (?:tank|tower)", r"overhead tank", r"oht", r"(?:paani|pani) (?:ki )?tanki",
                   r"neeru tank", r"neerina tank", "ನೀರಿನ ಟ್ಯಾಂಕ್"],
    "milk_dairy": [r"milk (?:dairy|dair|booth|parlour|parlor|centre|center)", r"(?:dudh|doodh|halina|haalina) (?:dairy|dair)",
                   r"dairy", r"nandini (?:milk )?(?:parlour|parlor|booth)", "ಹಾಲಿನ ಡೈರಿ", "ಡೈರಿ"],
    "community_hall": [r"community (?:hall|centre|center|bhavan|bhavana)", r"samudaya bhavana?", r"samaj bhavan",
                       r"comm hall", "ಸಮುದಾಯ ಭವನ"],
    "park": [r"park", r"udyanavana", r"udyana", r"garden", r"bagicha", r"bageecha", "ಉದ್ಯಾನ", "ಪಾರ್ಕ್"],
    "church": [r"church", r"chapel", r"girja ?ghar", "ಚರ್ಚ್"],
    "post_office": [r"post office", r"dak ?ghar", r"anche kacheri", r"ancheri", "ಅಂಚೆ ಕಚೇರಿ"],
    "petrol_bunk": [r"petrol (?:bunk|pump|bank|punk)", r"fuel station", r"gas station", "ಪೆಟ್ರೋಲ್ ಬಂಕ್"],
}
LM_RE = {}

R = {}  
PROTECT = set()  

ORD_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

@lru_cache(maxsize=None)
def repair_token(t):
    if (not t.isalpha()) or (not t.isascii()) or len(t) < 5 or t in PROTECT or t in VOCAB_SET:
        return t
    best, bs = None, 0.0
    for w in VOCAB:
        if w[0] != t[0] or abs(len(w) - len(t)) > 2:
            continue
        sc = sim(t, w)
        if sc > bs:
            best, bs = w, sc
    thr = 86.0 if len(t) == 5 else 80.0
    return best if (best is not None and bs >= thr) else t

def clean_text(raw):
    s = unicodedata.normalize("NFKC", str(raw)).lower()
    s = re.sub(r"#\s*(\d+[a-z]?)", r" house no \1 ", s)
    s = re.sub(r"\bh\s*\.?\s*no\b\.?", " house no ", s)
    s = re.sub(r"\b(\d+)\s*(?:st|nd|rd|th)\b", r"\1th", s)
    for w, n in ORD_WORDS.items():
        s = re.sub(rf"\b{w}\b(?=\s+(?:cross|main|road))", f"{n}th", s)
    s = re.sub(r"[.,;:()\[\]{}|\"'`~!?*+=<>_\\]", " ", s)
    s = re.sub(r"(?<![0-9])[-/]|[-/](?![0-9])", " ", s)
    s = re.sub(r"\b(\d+)th\s*(?:x|cr|crs|crss)\b", r"\1th cross", s)
    s = re.sub(r"\b(\d+)th\s*(?:mn|mnn|maiin|mainn)\b", r"\1th main", s)
    s = re.sub(r"\b(?:cr|crs)\b(?=\s+\d)", "cross", s)
    s = re.sub(r"\s+", " ", s).strip()
    out = []
    for t in s.split(" "):
        t = ABBR.get(t, t)
        for u in t.split(" "):
            out.append(repair_token(u))
    return " ".join(out)

def script_fracs(raw):
    n = max(len(str(raw)), 1)
    return len(KANNADA_RE.findall(str(raw))) / n, len(DEVANAGARI_RE.findall(str(raw))) / n


def load_extra():
    p = Path(OUT_DIR) / EXTRA_DICT
    if p.exists():
        with open(p, encoding="utf-8") as f:
            extra = json.load(f)
        print(f"[info] loaded extra dictionary {p}")
        return extra
    return {}

def apply_extra_pre(extra):
    for k, v in extra.get("abbrev", {}).items():
        ABBR[k.lower()] = v.lower()
    for t, terms in extra.get("landmark_terms", {}).items():
        LM_TERMS.setdefault(t, [])
        for term in terms:
            LM_TERMS[t].append(re.escape(" ".join(term.lower().split())))

def compile_landmark_regexes():
    LM_RE.clear()
    for t, terms in LM_TERMS.items():
        LM_RE[t] = re.compile(r"(?<![a-z0-9])(?:%s)(?![a-z0-9])" % "|".join(terms))

def raw_tokens(s):
    return set(re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", str(s)).lower()))

def init_nlp(extra):
    apply_extra_pre(extra)
    compile_landmark_regexes()

    towns = load("towns.csv")
    locs = load("localities.csv")
    poi = load("landmarks_poi.csv")

    PROTECT.clear()
    for s in list(towns["town_name"]) + list(locs["locality_name"]):
        PROTECT.update(raw_tokens(s))
    repair_token.cache_clear()

    locs = locs.copy()
    locs["pincode"] = locs["pincode"].astype(str)
    locs["name_clean"] = locs["locality_name"].map(clean_text)
    locs["toks"] = locs["name_clean"].str.split()
    locs["dist_phrase"] = locs["toks"].map(
        lambda ts: " ".join(t for t in ts if t not in GENERIC) or None)
    locs.loc[locs["dist_phrase"] == locs["name_clean"], "dist_phrase"] = None
    locs.loc[locs["dist_phrase"].fillna("").str.len() < 4, "dist_phrase"] = None

    towns = towns.copy()
    towns["name_clean"] = towns["town_name"].map(clean_text)

    pin2town = {}
    for r in locs.itertuples():
        pin2town.setdefault(str(r.pincode), set()).add(r.town_id)

    poi = poi.copy()
    loc_by_town = {t: g for t, g in locs.groupby("town_id")}
    pl = []
    for r in poi.itertuples():
        g = loc_by_town.get(r.town_id)
        if g is None or g.empty:
            pl.append((None, np.nan))
            continue
        d = dist(g["centroid_x"].values, g["centroid_y"].values, r.x, r.y)
        k = int(np.argmin(d))
        pl.append((g["locality_id"].iloc[k], float(d[k])))
    poi["locality_id"] = [a for a, _ in pl]
    poi["poi_to_centroid_m"] = [b for _, b in pl]

    poi_sorted = poi.dropna(subset=["locality_id"]).sort_values("poi_to_centroid_m")
    dup = poi_sorted.duplicated(["locality_id", "landmark_type"]).sum()
    poi_sorted = poi_sorted.drop_duplicates(["locality_id", "landmark_type"])
    poi_lookup = {(r.locality_id, r.landmark_type): (r.poi_id, r.x, r.y) for r in poi_sorted.itertuples()}

    types_in_poi = set(poi["landmark_type"].unique())
    types_in_dict = set(LM_TERMS)
    if types_in_poi != types_in_dict:
        print("[warn] landmark_type names differ between landmarks_poi.csv and the dictionary")

    alias = {}
    for lid, phrases in extra.get("locality_aliases", {}).items():
        for ph in phrases:
            alias[clean_text(ph)] = lid

    R.update(dict(towns=towns, locs=locs, poi=poi, pin2town=pin2town, poi_lookup=poi_lookup,
                  alias=alias, poi_dup=int(dup)))
    R["town_names"] = {r.town_id: r.name_clean for r in towns.itertuples()}
    R["loc_centroid"] = {r.locality_id: (r.centroid_x, r.centroid_y, r.town_id) for r in locs.itertuples()}
    R["known_tokens"] = (set(PROTECT) | VOCAB_SET | set(ABBR) | set(ABBR.values()) | GENERIC
                         | set(" ".join(REL_WORDS).split()) | TYPE_HEAD_TOKENS)


def resolve_town(raw, text, town_in=None):
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", str(raw))
    pin = m.group(1) if m else None
    if town_in is not None and pd.notna(town_in):
        return str(town_in), "column", pin
    if pin and len(R["pin2town"].get(pin, ())) == 1:
        return next(iter(R["pin2town"][pin])), "pincode", pin
    hits = {tid for tid, nm in R["town_names"].items() if re.search(PH(nm), text)}
    if len(hits) == 1:
        return next(iter(hits)), "town_name", pin
    return None, "unresolved", pin

def ctx_blocked(text, s, e):
    before = text[:s].split()[-1:]
    after = text[e:].split()[:1]
    if before and before[0] in REL_PRE_TOKENS:
        return True
    if after and after[0] in TYPE_HEAD_TOKENS:
        return True
    return False

def resolve_locality(text, town_id):
    L = R["locs"]
    cands = L if town_id is None else L[L["town_id"] == town_id]
    if cands.empty:
        cands = L
    cand_ids = set(cands["locality_id"])

    for ph, lid in R["alias"].items():
        if lid in cand_ids:
            m = re.search(PH(ph), text)
            if m:
                return lid, "alias_dict", 0.95, m.span()

    hits = []
    for r in cands.itertuples():
        m = re.search(PH(r.name_clean), text)
        if m:
            hits.append((r.locality_id, m.start(), m.end()))
    if hits:
        hits = [h for h in hits if not any(o is not h and o[1] <= h[1] and h[2] <= o[2]
                                           and (o[2] - o[1]) > (h[2] - h[1]) for o in hits)]
        ids = {h[0] for h in hits}
        if len(ids) == 1:
            h = hits[0]
            return h[0], "exact", 1.0, (h[1], h[2])
        return None, "ambiguous_exact", 0.0, None

    dp_count = cands["dist_phrase"].value_counts()
    hits = []
    for r in cands.itertuples():
        dp = r.dist_phrase
        if dp is None or (isinstance(dp, float) and np.isnan(dp)) or dp_count.get(dp, 0) != 1:
            continue
        m = re.search(PH(dp), text)
        if m and not ctx_blocked(text, m.start(), m.end()):
            hits.append((r.locality_id, m.start(), m.end()))
    if len({h[0] for h in hits}) == 1:
        h = hits[0]
        return h[0], "distinctive", 0.8, (h[1], h[2])

    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    words = [w for _, _, w in spans]
    best = {}
    for r in cands.itertuples():
        n = len(r.toks)
        for ln in {max(1, n - 1), n, n + 1}:
            if ln > 4:
                continue
            for i in range(len(words) - ln + 1):
                win = words[i:i + ln]
                if any(w.isdigit() for w in win) or all(w in GENERIC for w in win):
                    continue
                wt = " ".join(win)
                if len(wt) < 5 or wt[0] != r.name_clean[0]:
                    continue
                s_, e_ = spans[i][0], spans[i + ln - 1][1]
                if ctx_blocked(text, s_, e_):
                    continue
                sc = sim(wt, r.name_clean)
                if sc > best.get(r.locality_id, (0.0,))[0]:
                    best[r.locality_id] = (sc, s_, e_)
    if best:
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        top_id, (top_sc, s_, e_) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        if top_sc >= FUZZY_ACCEPT:
            if top_sc - second >= FUZZY_MARGIN:
                return top_id, "fuzzy", round(top_sc / 100.0, 3), (s_, e_)
            return None, "fuzzy_ambiguous", 0.0, None
    return None, "none", 0.0, None

def detect_landmarks(text, mask):
    hits = []
    for t, rx in LM_RE.items():
        for m in rx.finditer(text):
            s, e = m.span()
            if mask is not None and not (e <= mask[0] or s >= mask[1]):
                continue  
            before = text[max(0, s - 24):s]
            after = text[e:e + 16]
            rel = bool(REL_RE.search(before) or REL_RE.search(after))
            hits.append((s, e, t, rel))
    hits.sort()
    return hits

RE_CROSS = [re.compile(r"\b(\d{1,3})th cross\b"), re.compile(r"\bcross (?:no )?(\d{1,3})\b")]
RE_MAIN = [re.compile(r"\b(\d{1,3})th main\b"), re.compile(r"\bmain (?:no )?(\d{1,3})\b")]
RE_GALI = [re.compile(r"\bgall?i (?:no |number )?(\d{1,3})\b")]
RE_ROAD = [re.compile(r"\b(\d{1,3})th road\b"), re.compile(r"\broad (?:no )?(\d{1,3})\b")]
RE_WARD = [re.compile(r"\bward (?:no )?(\d{1,3})\b")]
RE_BLOCK = [re.compile(r"\b([a-z]) block\b"), re.compile(r"\bblock (?:no )?([a-z]|\d{1,2})(?= |$)")]
RE_HOUSE = [re.compile(r"\bhouse (?:no )?([a-z]?\d+[a-z]?(?:[/-]\d+)?)(?= |$)"),
            re.compile(r"\b(?:door|plot|flat) (?:no )?([a-z]?\d+[a-z]?)(?= |$)")]
RE_LEAD_NUM = re.compile(r"^(\d{1,4}[a-z]?(?:[/-]\d+)?)(?= )")
NOT_HOUSE_NEXT = {"cross", "main", "gali", "road", "ward", "block", "floor", "th"}
RE_VILLAGE = re.compile(r"\b(?:village|gram|taluk|tq|district|dist)\b")

def first_match(patterns, text):
    for rx in patterns:
        m = rx.search(text)
        if m:
            return m.group(1)
    return None

def parse_house(text):
    v = first_match(RE_HOUSE, text)
    if v:
        return v
    m = RE_LEAD_NUM.match(text)
    if m:
        nxt = text[m.end():].split()[:1]
        if not (nxt and nxt[0] in NOT_HOUSE_NEXT):
            return m.group(1)
    return None

def parse_addresses(addr, text_col, type_col, town_col):
    rows = []
    types = addr[type_col].values if type_col else [None] * len(addr)
    towns_in = addr[town_col].values if town_col else [None] * len(addr)
    for aid, raw, atype, tin in zip(addr["address_id"].values, addr[text_col].values, types, towns_in):
        text = clean_text(raw)
        kn, dv = script_fracs(raw)
        town_id, town_method, pin = resolve_town(raw, text, tin)
        lid, lmethod, lconf, lspan = resolve_locality(text, town_id)
        if town_id is None and lid is not None:
            town_id, town_method = R["loc_centroid"][lid][2], "via_locality"

        hits = detect_landmarks(text, lspan)
        types_all, types_hc = [], []
        for _, _, t, rel in hits:
            if t not in types_all:
                types_all.append(t)
            if rel and t not in types_hc:
                types_hc.append(t)
        primary = types_hc[0] if types_hc else (types_all[0] if types_all else None)
        lm_conf = "high" if types_hc else ("low" if types_all else None)
        rel_no_type = bool(REL_RE.search(text)) and not types_all
        rel_ctx = None
        if rel_no_type:
            m = REL_RE.search(text)
            rel_ctx = text[max(0, m.start() - 25): m.end() + 25]

        poi = R["poi_lookup"].get((lid, primary)) if (lid is not None and primary) else None

        rec = dict(
            address_id=aid, address_type=atype, text_raw=raw, text_clean=text,
            script_kn_frac=round(kn, 3), script_dev_frac=round(dv, 3), pincode=pin,
            town_id=town_id, town_method=town_method,
            locality_id=lid, locality_method=lmethod, locality_conf=lconf,
            house_no=parse_house(text), cross_no=first_match(RE_CROSS, text), main_no=first_match(RE_MAIN, text),
            gali_no=first_match(RE_GALI, text), road_no=first_match(RE_ROAD, text),
            ward_no=first_match(RE_WARD, text), block=first_match(RE_BLOCK, text),
            is_village_style=bool(RE_VILLAGE.search(text)),
            lm_types_all="|".join(types_all) or None, lm_types_hc="|".join(types_hc) or None,
            lm_primary=primary, lm_conf=lm_conf, lm_ambiguous=len(types_hc) > 1,
            rel_no_type=rel_no_type, rel_context=rel_ctx,
            poi_id=poi[0] if poi else None, poi_x=poi[1] if poi else np.nan, poi_y=poi[2] if poi else np.nan,
        )
        rec["n_fields_found"] = sum(rec[k] is not None and not (isinstance(rec[k], float) and np.isnan(rec[k]))
                                    for k in ["locality_id", "house_no", "cross_no", "main_no", "gali_no",
                                              "road_no", "ward_no", "block", "lm_primary"])
        rows.append(rec)
    return pd.DataFrame(rows)

def print_parse_summary(P):
    hdr("P1-P4 PARSER COVERAGE (all addresses)")
    print(f"fuzzy backend: {FUZZ_BACKEND}   |   rows: {len(P)}   |   POI (locality,type) duplicates: {R['poi_dup']}")
    print("\nTown resolution method:")
    print(P["town_method"].value_counts(dropna=False).to_string())
    print("\nLocality resolution method:")
    print(P["locality_method"].value_counts(dropna=False).to_string())
    if P["address_type"].notna().any():
        print("\nLocality resolved share by address_type:")
        t = P.groupby("address_type").agg(n=("address_id", "size"),
                                          resolved=("locality_id", lambda s: s.notna().mean()))
        print(t.round(3).to_string())
    print("\nLocality resolved share by town:")
    t = P.groupby(P["town_id"].fillna("unknown")).agg(n=("address_id", "size"),
                                                      resolved=("locality_id", lambda s: s.notna().mean()))
    print(t.round(3).to_string())
    print("\nStreet-field coverage (share of addresses with the field), by town:")
    f = P.assign(**{c: P[c].notna() for c in ["house_no", "cross_no", "main_no", "gali_no", "road_no", "ward_no", "block"]})
    print(f.groupby(P["town_id"].fillna("unknown"))[["house_no", "cross_no", "main_no", "gali_no", "road_no",
                                                     "ward_no", "block"]].mean().round(3).to_string())
    print("\nLandmark detection:")
    print(f"  any landmark type found      : {P['lm_types_all'].notna().sum()}")
    print(f"  high confidence (relation word nearby): {P['lm_types_hc'].notna().sum()}")
    print(f"  low confidence (type only)   : {(P['lm_conf'] == 'low').sum()}")
    print(f"  ambiguous (>1 high-conf type): {int(P['lm_ambiguous'].sum())}")
    print(f"  relation word but NO type    : {int(P['rel_no_type'].sum())}   (candidates for the LLM dictionary pass)")
    print(f"  resolved to a POI            : {P['poi_id'].notna().sum()}")
    print("\nLandmark type counts (all / high-confidence):")
    ca, ch = Counter(), Counter()
    for v in P["lm_types_all"].dropna():
        ca.update(v.split("|"))
    for v in P["lm_types_hc"].dropna():
        ch.update(v.split("|"))
    print(pd.DataFrame({"all": pd.Series(ca), "high_conf": pd.Series(ch)}).fillna(0).astype(int)
          .sort_values("all", ascending=False).to_string())


def trail_features(gps):
    g = gps.sort_values(["visit_id", "seq"]).copy()
    g["step"] = np.hypot(g.groupby("visit_id")["x"].diff(), g.groupby("visit_id")["y"].diff())
    a = g.groupby("visit_id").agg(n_points=("seq", "size"), trail_len=("step", "sum"),
                                  x0=("x", "first"), y0=("y", "first"), x1=("x", "last"), y1=("y", "last"),
                                  t0=("point_ts", "first"), t1=("point_ts", "last")).reset_index()
    a["net_disp"] = np.hypot(a["x1"] - a["x0"], a["y1"] - a["y0"])
    a["path_eff"] = np.where(a["trail_len"] > 0, a["net_disp"] / a["trail_len"].replace(0, np.nan), np.nan)
    if pd.api.types.is_numeric_dtype(a["t0"]):
        a["trail_dur_s"] = a["t1"] - a["t0"]
    else:
        a["trail_dur_s"] = (pd.to_datetime(a["t1"], errors="coerce") - pd.to_datetime(a["t0"], errors="coerce")).dt.total_seconds()
    return a.drop(columns=["t0", "t1"])

def integrity_signals(fv, gps, base):
    from sklearn.cluster import DBSCAN

    tr = trail_features(gps)
    fv = fv.merge(tr, on="visit_id", how="left")
    b = base.rename(columns={"precision": "base_precision"})[["address_id", "geocoder_x", "geocoder_y", "base_precision"]]
    fv = fv.merge(b, on="address_id", how="left")

    fv["checkin_to_base_m"] = dist(fv["checkin_x"], fv["checkin_y"], fv["geocoder_x"], fv["geocoder_y"])
    fv["start_to_base_m"] = dist(fv["x0"], fv["y0"], fv["geocoder_x"], fv["geocoder_y"])
    fv["checkin_to_trail_end_m"] = dist(fv["checkin_x"], fv["checkin_y"], fv["x1"], fv["y1"])

    labels = np.full(len(fv), -1)
    valid = fv[["checkin_x", "checkin_y"]].notna().all(axis=1).values
    labels[valid] = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(
        fv.loc[valid, ["checkin_x", "checkin_y"]].values)
    fv["cl"] = labels

    rep = []
    for c in sorted(set(labels) - {-1}):
        sub = fv[fv["cl"] == c]
        u = sub.drop_duplicates("address_id")[["geocoder_x", "geocoder_y"]].dropna()
        if len(u):
            cx, cy = u["geocoder_x"].mean(), u["geocoder_y"].mean()
            spread = float(np.sqrt(np.mean((u["geocoder_x"] - cx) ** 2 + (u["geocoder_y"] - cy) ** 2)))
        else:
            spread = np.nan
        top = sub["agent_id"].value_counts()
        rep.append(dict(cluster=c, visits=len(sub), addresses=sub["address_id"].nunique(),
                        pin_spread_m=round(spread, 1), centre_x=round(sub["checkin_x"].mean(), 1),
                        centre_y=round(sub["checkin_y"].mean(), 1), n_agents=sub["agent_id"].nunique(),
                        top_agent=top.index[0], top_agent_share=round(top.iloc[0] / len(sub), 3)))
    cl_rep = pd.DataFrame(rep)
    hub_cl = set(cl_rep.loc[(cl_rep["visits"] >= HUB_MIN_VISITS) & (cl_rep["pin_spread_m"] >= HUB_MIN_SPREAD_M), "cluster"]) \
        if len(cl_rep) else set()
    cl_rep["is_hub_cluster"] = cl_rep["cluster"].isin(hub_cl) if len(cl_rep) else []
    spread_map = dict(zip(cl_rep["cluster"], cl_rep["pin_spread_m"])) if len(cl_rep) else {}
    fv["cl_pin_spread_m"] = fv["cl"].map(spread_map)

    fv["photo_addrs"] = fv.groupby("photo_hash")["address_id"].transform("nunique").fillna(1)
    fv["photo_reuse_n"] = fv.groupby("photo_hash")["visit_id"].transform("size").fillna(1)

    fv["_ts"] = pd.to_datetime(fv["checkin_ts"], errors="coerce")
    o = fv.sort_values(["agent_id", "_ts"])
    gb = o.groupby("agent_id")
    dt_h = gb["_ts"].diff().dt.total_seconds() / 3600.0
    dd_km = np.hypot(gb["checkin_x"].diff(), gb["checkin_y"].diff()) / 1000.0
    same_day = o["_ts"].dt.date == gb["_ts"].shift().dt.date
    fv["implied_speed_kmh"] = np.where((dt_h > 0) & same_day, dd_km / dt_h, np.nan)
    fv = fv.drop(columns=["_ts"])

    fv["hub_member"] = fv["cl"].isin(hub_cl)
    fv["tiny_trail"] = (fv["trail_len"] < 200) & (fv["start_to_base_m"] > 1500)
    fv["is_hub2"] = fv["hub_member"] | (fv["photo_addrs"] >= PHOTO_ADDR_THRESHOLD)
    fv["integrity_risk"] = (RISK_W["hub"] * fv["hub_member"].astype(float)
                            + RISK_W["photo"] * np.clip((fv["photo_addrs"] - 1) / 2.0, 0, 1)
                            + RISK_W["tiny"] * fv["tiny_trail"].astype(float)
                            + RISK_W["start"] * np.clip((fv["start_to_base_m"].fillna(0) - 500) / 1500.0, 0, 1))
    fv["tier"] = np.where(fv["is_hub2"], "flagged",
                          np.where(fv["outcome"] == "address_not_traceable", "not_traceable", "clean"))

    resid = np.full(len(fv), np.nan)
    pos = {idx: i for i, idx in enumerate(fv.index)}
    for _, g in fv.groupby("address_id"):
        c = g[g["tier"] == "clean"]
        if c.empty:
            continue
        for idx, row in g.iterrows():
            others = c[c["agent_id"] != row["agent_id"]]
            if len(others):
                resid[pos[idx]] = float(dist(row["checkin_x"], row["checkin_y"],
                                             others["checkin_x"].median(), others["checkin_y"].median()))
    fv["lao_resid_m"] = resid
    return fv, cl_rep, hub_cl

def build_pseudo_truth(fv, addr, splits):
    clean = fv[fv["tier"] == "clean"].copy()
    pt = clean.groupby("address_id").agg(pt_x=("checkin_x", "median"), pt_y=("checkin_y", "median"),
                                         pt_k=("visit_id", "size")).reset_index()
    clean = clean.merge(pt[["address_id", "pt_x", "pt_y"]], on="address_id")
    clean["_d"] = dist(clean["checkin_x"], clean["checkin_y"], clean["pt_x"], clean["pt_y"])
    pt = pt.merge(clean.groupby("address_id")["_d"].median().rename("pt_spread_m").reset_index(), on="address_id")
    acct = addr[["address_id", "account_id"]] if "account_id" in addr.columns else \
        fv[["address_id", "account_id"]].drop_duplicates("address_id")
    sp_col = pick_col(splits, ["split", "set", "subset", "partition"], "split")
    pt = pt.merge(acct, on="address_id", how="left").merge(
        splits[["account_id", sp_col]].rename(columns={sp_col: "split"}), on="account_id", how="left")
    return pt


def norm_num(v):
    v = str(v).strip().lower()
    if v in ("", "nan", "none", "na", "-"):
        return None
    v = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", v)
    v = re.sub(r"\.0$", "", v)
    if v.isdigit():
        v = v.lstrip("0") or "0"
    return v

def norm_set(v):
    v = str(v).strip().lower()
    if v in ("", "nan", "none"):
        return set()
    return {x.strip().replace(" ", "_") for x in v.split("|") if x.strip()}

def evaluate_parser(P, half):
    p = Path(OUT_DIR) / LABELS_FILLED
    if not p.exists():
        print(f"\n[info] {p} not found; skipping parser evaluation (label the sample first).")
        return
    lab = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "split_half" not in lab.columns:
        print("[warn] split_half column missing in the filled sheet")
        return
    lab = lab[lab["split_half"] == half].copy()
    hdr(f"PARSER EVALUATION on the {half.upper()} half (n={len(lab)})")
    if half == "test":
        print("!! You are scoring the LOCKED test half. Do not tune the parser on this afterwards.")
    m = lab.merge(P, on="address_id", how="left", suffixes=("_lab", ""))
    errors = []
    out = []

    for name, lcol, pcol in [("town", "label_town_id", "town_id"), ("locality", "label_locality_id", "locality_id")]:
        L = m[lcol].str.strip().str.upper().replace("", np.nan)
        Pp = m[pcol].astype(object).where(m[pcol].notna(), np.nan)
        Pp = Pp.map(lambda v: str(v).strip().upper() if pd.notna(v) else np.nan)
        n = len(m)
        n_lab, n_pred = int(L.notna().sum()), int(Pp.notna().sum())
        tp = int(((L == Pp) & L.notna() & Pp.notna()).sum())
        out.append(dict(field=name, n=n, labelled=n_lab, predicted=n_pred, correct=tp,
                        precision=tp / n_pred if n_pred else np.nan, recall=tp / n_lab if n_lab else np.nan))
        for i in m.index[(L != Pp) & ~(L.isna() & Pp.isna())]:
            errors.append(dict(address_id=m.at[i, "address_id"], field=name, label=L[i], pred=Pp[i],
                               text_clean=m.at[i, "text_clean"]))

    for name, lcol, pcol in [("house_no", "label_house_no", "house_no"), ("cross", "label_cross", "cross_no"),
                             ("main", "label_main", "main_no"), ("gali", "label_gali", "gali_no"),
                             ("road", "label_road", "road_no"), ("ward", "label_ward", "ward_no"),
                             ("block", "label_block", "block")]:
        L = m[lcol].map(norm_num)
        Pp = m[pcol].map(lambda v: norm_num(v) if pd.notna(v) else None)
        n_lab, n_pred = int(L.notna().sum()), int(Pp.notna().sum())
        tp = int(sum(1 for a, b in zip(L, Pp) if a is not None and a == b))
        out.append(dict(field=name, n=len(m), labelled=n_lab, predicted=n_pred, correct=tp,
                        precision=tp / n_pred if n_pred else np.nan, recall=tp / n_lab if n_lab else np.nan))
        for i, (a, b) in zip(m.index, zip(L, Pp)):
            if a != b:
                errors.append(dict(address_id=m.at[i, "address_id"], field=name, label=a, pred=b,
                                   text_clean=m.at[i, "text_clean"]))

    for name, pcol in [("landmark_all", "lm_types_all"), ("landmark_hc", "lm_types_hc")]:
        tp = fp = fn = 0
        for i in m.index:
            L = norm_set(m.at[i, "label_landmark_types"])
            Pp = norm_set(m.at[i, pcol] if pd.notna(m.at[i, pcol]) else "")
            tp += len(L & Pp)
            fp += len(Pp - L)
            fn += len(L - Pp)
            if L != Pp:
                errors.append(dict(address_id=m.at[i, "address_id"], field=name, label="|".join(sorted(L)),
                                   pred="|".join(sorted(Pp)), text_clean=m.at[i, "text_clean"]))
        out.append(dict(field=name, n=len(m), labelled=tp + fn, predicted=tp + fp, correct=tp,
                        precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan))

    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    res.to_csv(Path(OUT_DIR) / f"parser_eval_{half}.csv", index=False)
    pd.DataFrame(errors).to_csv(Path(OUT_DIR) / f"parser_errors_{half}.csv", index=False)
    print(f"\nsaved parser_eval_{half}.csv and parser_errors_{half}.csv (every disagreement, with the cleaned text)")


def run_parse():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    extra = load_extra()
    validate_all_datasets_present()
    init_nlp(extra)

    accounts = load("accounts.csv")
    agents = load("agents.csv")
    lenders = load("lenders.csv")
    
    addr = load("addresses.csv")
    text_col = detect_text_col(addr)
    type_col = pick_col(addr, ["address_type", "type"], "address type", required=False)
    town_col = pick_col(addr, ["town_id"], "town", required=False)
    print(f"[info] addresses columns: {list(addr.columns)} | text column = '{text_col}'")

    P = parse_addresses(addr, text_col, type_col, town_col)
    if "account_id" in addr.columns:
        P.insert(1, "account_id", addr["account_id"].values)
    print_parse_summary(P)

    resid_loc = P[P["locality_id"].isna()][["address_id", "address_type", "town_id", "text_clean"]]
    resid_loc.to_csv(out / "residual_locality.csv", index=False, encoding="utf-8-sig")
    resid_lm = P[P["rel_no_type"]][["address_id", "text_clean", "rel_context"]]
    resid_lm.to_csv(out / "residual_landmark.csv", index=False, encoding="utf-8-sig")
    known = R["known_tokens"]
    tok = Counter()
    for t in pd.concat([resid_loc["text_clean"], resid_lm["text_clean"]]).unique():
        for w in t.split():
            if not w.isdigit() and len(w) >= 3 and w not in known:
                tok[w] += 1
    pd.DataFrame(tok.most_common(300), columns=["token", "n_addresses"]).to_csv(
        out / "unmatched_tokens.csv", index=False, encoding="utf-8-sig")
    print(f"\nresidual files: residual_locality.csv ({len(resid_loc)}), residual_landmark.csv ({len(resid_lm)}), unmatched_tokens.csv")

    hdr("INTEGRITY SIGNALS")
    fv = load("field_visits.csv")
    gps = load("visit_gps_points.csv")
    base = load("baseline_geocodes.csv")
    splits = load("splits.csv")
    fv, cl_rep, hub_cl = integrity_signals(fv, gps, base)

    print("DBSCAN clusters (legit ones have a small pin spread; hubs have a large one):")
    print(cl_rep.sort_values("visits", ascending=False).to_string(index=False))
    print(f"\nhub clusters used: {sorted(hub_cl)}")
    print(f"check-in vs trail end (m): median {np.nanmedian(fv['checkin_to_trail_end_m']):.1f}, "
          f"p75 {np.nanpercentile(fv['checkin_to_trail_end_m'], 75):.1f}, max {np.nanmax(fv['checkin_to_trail_end_m']):.1f}   (expect ~15.9 / 28.4 / 155.7)")
    print("\nTier counts:")
    print(fv["tier"].value_counts().to_string())
    ag = fv.groupby("agent_id").agg(
        visits=("visit_id", "size"), flagged=("is_hub2", "sum"), flagged_share=("is_hub2", "mean"),
        hub_member=("hub_member", "sum"), photo_flag=("photo_addrs", lambda s: int((s >= PHOTO_ADDR_THRESHOLD).sum())),
        tiny_trail=("tiny_trail", "sum"), mean_risk=("integrity_risk", "mean"),
        lao_resid_median_clean=("lao_resid_m", lambda s: np.nan))
    clean_res = fv[fv["tier"] == "clean"].groupby("agent_id")["lao_resid_m"].agg(
        lao_resid_median_clean="median", lao_resid_p90_clean=lambda s: s.quantile(0.9))
    ag = ag.drop(columns=["lao_resid_median_clean"]).join(clean_res)
    print("\nPer-agent integrity report (expect FA009 ~231 flagged; the others near zero):")
    print(ag.round(3).to_string())
    print("\nFlagged visits outside FA009 (inspect these by hand):")
    cols = ["visit_id", "agent_id", "address_id", "outcome", "cl", "photo_addrs", "trail_len", "start_to_base_m", "integrity_risk"]
    print(fv[fv["is_hub2"] & (fv["agent_id"] != "FA009")][cols].round(1).to_string(index=False))
    ag.reset_index().to_csv(out / "agent_integrity_report.csv", index=False)
    cl_rep.to_csv(out / "integrity_cluster_report.csv", index=False)

    pt = build_pseudo_truth(fv, addr, splits)
    surv = load("surveyed_addresses.csv")
    hdr("PSEUDO-TRUTH SANITY (against surveyed truth, reporting only)")
    j = pt.merge(surv, on="address_id")
    print(f"addresses with pseudo-truth: {len(pt)}  (by split: {pt['split'].value_counts().to_dict()})")
    print("pseudo-truth k distribution:", pt["pt_k"].value_counts().sort_index().to_dict())
    s = err_stats(dist(j["pt_x"], j["pt_y"], j["surveyed_x"], j["surveyed_y"]))
    print(f"pseudo-truth vs surveyed on {s['n']} overlapping addresses: median {s['median']:.1f} m, "
          f"p90 {s['p90']:.1f} m, max {s['max']:.1f} m   (earlier EDA: 14 / 68 / 252)")
    jv = fv.merge(surv, on="address_id")
    jv["err_to_truth"] = dist(jv["checkin_x"], jv["checkin_y"], jv["surveyed_x"], jv["surveyed_y"])
    rows = []
    for tier, g in jv.groupby("tier"):
        e = err_stats(g["err_to_truth"])
        rows.append(dict(tier=tier, **e))
    print("\nVisit check-in error vs surveyed truth by tier (earlier EDA: clean 24/129, flagged 5368/7829, not_traceable 681/2349):")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    hdr("LOCALITY ASSIGNMENT SANITY (rank of the assigned locality by distance to pseudo-truth)")
    L = R["locs"]
    m = P.merge(pt[["address_id", "pt_x", "pt_y", "pt_k"]], on="address_id")
    m = m[m["locality_id"].notna() & m["town_id"].notna()]
    ranks, d_assigned = [], []
    for r in m.itertuples():
        g = L[L["town_id"] == r.town_id]
        d = dist(g["centroid_x"].values, g["centroid_y"].values, r.pt_x, r.pt_y)
        order = g["locality_id"].values[np.argsort(d)]
        if r.locality_id in order:
            ranks.append(int(np.where(order == r.locality_id)[0][0]) + 1)
            cx, cy, _ = R["loc_centroid"][r.locality_id]
            d_assigned.append(float(dist(cx, cy, r.pt_x, r.pt_y)))
        else:
            ranks.append(np.nan)
            d_assigned.append(np.nan)
    m = m.assign(loc_rank=ranks, dist_to_assigned_centroid_m=d_assigned)
    print(f"addresses checked: {len(m)}")
    if len(m):
        print("rank of assigned locality (1 = nearest centroid to the pseudo-truth):")
        print(m["loc_rank"].value_counts(normalize=True).sort_index().round(3).head(8).to_string())
        print("\nby method: share with rank <= 3, and median distance to the assigned centroid (m)")
        print(m.groupby("locality_method").agg(n=("address_id", "size"), rank_le3=("loc_rank", lambda s: (s <= 3).mean()),
                                               med_dist=("dist_to_assigned_centroid_m", "median")).round(3).to_string())
        m[m["loc_rank"] > 3][["address_id", "text_clean", "locality_id", "locality_method", "loc_rank"]].to_csv(
            out / "locality_suspects.csv", index=False, encoding="utf-8-sig")
        print("saved locality_suspects.csv (assigned locality ranked > 3: look at these by eye)")

    P.to_csv(out / "address_parsed.csv", index=False, encoding="utf-8-sig")
    keep = ["visit_id", "account_id", "address_id", "agent_id", "outcome", "checkin_x", "checkin_y", "gps_accuracy_m",
            "dwell_s", "n_points", "trail_len", "net_disp", "path_eff", "trail_dur_s", "start_to_base_m",
            "checkin_to_base_m", "checkin_to_trail_end_m", "base_precision", "cl", "cl_pin_spread_m", "hub_member",
            "photo_addrs", "photo_reuse_n", "tiny_trail", "implied_speed_kmh", "is_hub2", "integrity_risk",
            "tier", "lao_resid_m"]
    fv[[c for c in keep if c in fv.columns]].to_csv(out / "visit_features.csv", index=False)
    ev = P[["address_id", "town_id", "locality_id", "locality_method", "lm_primary", "poi_id", "poi_x", "poi_y"]] \
        .merge(base.rename(columns={"precision": "base_precision"}), on="address_id", how="left") \
        .merge(pt, on="address_id", how="left")
    ev["is_surveyed"] = ev["address_id"].isin(surv["address_id"])
    ev.to_csv(out / "address_evidence.csv", index=False)
    hdr("FILES WRITTEN")
    for f in ["address_parsed.csv", "visit_features.csv", "address_evidence.csv", "integrity_cluster_report.csv",
              "agent_integrity_report.csv", "residual_locality.csv", "residual_landmark.csv", "unmatched_tokens.csv",
              "locality_suspects.csv"]:
        print(" ", out / f)

    evaluate_parser(P, EVAL_HALF)

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="ps3_out")
    args=ap.parse_args(argv)
    global DATA_DIR, OUT_DIR
    DATA_DIR=args.data_dir
    OUT_DIR=args.out_dir
    run_parse()

if __name__ == "__main__":
    main()
