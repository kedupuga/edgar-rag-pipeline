import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize(value):
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None

    val = str(value).strip()
    val = re.sub(r"\s*(million|billion|thousand|mn|bn)\b.*$", "", val, flags=re.IGNORECASE)
    val = val.replace(",", "").replace("$", "").strip()

    try:
        return str(float(val))
    except ValueError:
        return val.lower()


def _is_match(extracted_values, gt_value):
    gt_norm = normalize(gt_value)
    if gt_norm is None:
        return False
    return any(normalize(v) == gt_norm for v in extracted_values if normalize(v) is not None)


def build_results(ground_truth_df, extracted, company, year):
    extracted_map = {r["variable"]: r for r in extracted}

    records = []
    for _, row in ground_truth_df.iterrows():
        variable = row["variable"]
        gt_value = row["value"]

        ext = extracted_map.get(variable, {})
        extracted_values = ext.get("values", [])

        is_match = bool(extracted_values) and _is_match(extracted_values, gt_value)

        records.append({
            "company": company,
            "year": year,
            "variable": variable,
            "ground_truth": gt_value,
            "extracted_values": ", ".join(str(v) for v in extracted_values),
            "match": is_match,
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
        print(f"    Ground Truth     : {row['ground_truth']}")
        print(f"    Extracted Values : {row['extracted_values']}")
        print()

    total = len(results_df)
    matched = int(results_df["match"].sum())
    pct = 100 * matched / total if total > 0 else 0
    print(f"Overall accuracy: {matched}/{total} ({pct:.1f}%)")
    print("=" * 60 + "\n")
