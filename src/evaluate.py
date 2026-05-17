import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize(value):
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None

    val = str(value).strip()
    # strip trailing unit words the LLM might append (e.g. "3,701 million")
    val = re.sub(r"\s*(million|billion|thousand|mn|bn)\b.*$", "", val, flags=re.IGNORECASE)
    val = val.replace(",", "").replace("$", "").strip()

    try:
        return str(float(val))
    except ValueError:
        return val.lower()


def _is_match(extracted_value, gt_value):
    gt_norm = normalize(gt_value)
    if gt_norm is None:
        return False

    # split on ", " not "," so formatted numbers like "49,520" stay intact
    parts = [normalize(p) for p in str(extracted_value).split(", ")]
    return any(p == gt_norm for p in parts if p is not None)


def build_results(ground_truth_df, extracted, company, year):
    extracted_map = {r["variable"]: r for r in extracted}

    records = []
    for _, row in ground_truth_df.iterrows():
        variable = row["variable"]
        gt_value = row["value"]

        ext = extracted_map.get(variable, {})
        extracted_value = ext.get("value")
        context = ext.get("context")

        is_match = extracted_value is not None and _is_match(extracted_value, gt_value)

        records.append({
            "company": company,
            "year": year,
            "variable": variable,
            "ground_truth": gt_value,
            "extracted_value": extracted_value,
            "match": is_match,
            "context": context,
        })

    results_df = pd.DataFrame(records)
    matched = int(results_df["match"].sum())
    logger.info(f"Evaluation complete — {matched}/{len(results_df)} values matched")
    return results_df


def print_summary(results_df):
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS SUMMARY")
    print("=" * 60)

    for _, row in results_df.iterrows():
        status = "PASS" if row["match"] else "FAIL"
        print(f"[{status}] {row['variable']}")
        print(f"    Ground Truth : {row['ground_truth']}")
        print(f"    Extracted    : {row['extracted_value']}")
        print()

    total = len(results_df)
    matched = int(results_df["match"].sum())
    pct = 100 * matched / total if total > 0 else 0
    print(f"Overall accuracy: {matched}/{total} ({pct:.1f}%)")
    print("=" * 60 + "\n")
