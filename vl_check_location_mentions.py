"""
vl_check_location_mentions.py -- How often do BERD captions mention anatomical location?

Supports (or refutes) the statement that location information is largely absent from the
captions. Three levels are checked for every image (train / val / test):
  1. exact   : the image's own location string (normalised) appears in its caption
  2. any_term: the caption contains any term of a fixed anatomical list (the BERD location
               field also contains finding/procedure words, so it is not used as a term list)
  3. lateral : the caption contains "left" or "right"
Note: device phrases such as "tracheal tube" or "tracheal bronchial support" also count
as anatomical terms here; inspect results/location_mentions_examples.csv manually.

Outputs: results/location_mentions_summary.csv, location_mentions_by_location.csv,
         location_mentions_terms.csv, location_mentions_examples.csv
Run: python vl_check_location_mentions.py   (no GPU needed, a few seconds)
"""

import re
from collections import Counter

import pandas as pd

import vl_common as C
from vl_metrics import norm_text

ANATOMY_TERMS = {
    "trachea", "tracheal", "carina", "bronchus", "bronchi", "bronchial", "lobe", "lobar",
    "segment", "segmental", "subsegment", "main", "upper", "middle", "lower", "intermediate",
    "lingula", "lingular", "apical", "basal", "anterior", "posterior", "lateral", "medial",
    "superior", "inferior", "glottis", "larynx", "vocal", "left", "right",
}
GENERIC = {"the", "of", "and", "a", "an", "in", "to", "opening", "mouth", "orifice", ""}


def words(s):
    return set(norm_text(s).replace("-", " ").split())


if __name__ == "__main__":
    _, _, TR, VA, TE = C.get_splits()
    cap_key = C.detect_key(TR, C.CAPTION_KEYS)
    loc_key = C.detect_key(TR, C.LOCATION_KEYS)
    loc_vocab = set().union(*(words(r.get(loc_key, "")) for r in TR)) - GENERIC
    terms = ANATOMY_TERMS
    print(f"caption key '{cap_key}', location key '{loc_key}'")
    print(f"location-field vocabulary ({len(loc_vocab)} words): {sorted(loc_vocab)}\n")

    summary, by_loc, examples, term_cnt = [], [], [], Counter()
    for tag, recs in [("train", TR), ("val", VA), ("test", TE)]:
        n = len(recs); exact = anyt = lat = 0
        per = {}
        for r in recs:
            cap, loc = norm_text(r.get(cap_key, "")), norm_text(r.get(loc_key, ""))
            cw = words(cap)
            e = bool(loc) and re.search(rf"\b{re.escape(loc)}\b", cap) is not None
            hit = cw & terms
            a, l = bool(hit), bool(cw & {"left", "right"})
            exact += e; anyt += a; lat += l
            if tag == "train":
                term_cnt.update(hit)
                d = per.setdefault(loc, [0, 0, 0]); d[0] += 1; d[1] += e; d[2] += a
                if a and len(examples) < 200:
                    examples.append({"location": loc, "caption": cap, "terms": ", ".join(sorted(hit)),
                                     "exact_match": e})
        summary.append({"split": tag, "images": n,
                        "exact_location_in_caption_%": round(100 * exact / n, 1),
                        "any_anatomical_term_%": round(100 * anyt / n, 1),
                        "left_or_right_%": round(100 * lat / n, 1)})
        if tag == "train":
            by_loc = [{"location": k, "images": v[0], "exact_%": round(100 * v[1] / v[0], 1),
                       "any_term_%": round(100 * v[2] / v[0], 1)}
                      for k, v in sorted(per.items(), key=lambda x: -x[1][0])]

    df = pd.DataFrame(summary); print(df.to_string(index=False))
    df.to_csv(C.RESULTS_DIR / "location_mentions_summary.csv", index=False)
    pd.DataFrame(by_loc).to_csv(C.RESULTS_DIR / "location_mentions_by_location.csv", index=False)
    pd.DataFrame(term_cnt.most_common(), columns=["term", "train_captions"]).to_csv(
        C.RESULTS_DIR / "location_mentions_terms.csv", index=False)
    pd.DataFrame(examples).to_csv(C.RESULTS_DIR / "location_mentions_examples.csv", index=False)
    print("\nMost frequent anatomical terms in training captions:")
    print(term_cnt.most_common(15))
    print("\n✓ results/location_mentions_*.csv")
