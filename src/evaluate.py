import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize(value):
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None

    val = str(value).replace(",", "").replace("$", "").strip()

    try:
        return str(float(val))
    except ValueError:
        return val.lower()


def build_results(ground_truth_df, extracted, company, year):
    extracted_map = {r["variable"]: r for r in extracted}

    records = []
    for _, row in ground_truth_df.iterrows():
        variable = row["variable"]
        gt_value = row["value"]

        ext = extracted_map.get(variable, {})
        extracted_value = ext.get("value")
        context = ext.get("context")

        gt_norm = normalize(gt_value)
        ex_norm = normalize(extracted_value)

        is_match = False
        if gt_norm is not None and ex_norm is not None:
            try:
                gt_f, ex_f = float(gt_norm), float(ex_norm)
                # 1% tolerance handles rounding differences across fiscal year columns
                is_match = abs(gt_f - ex_f) / (abs(gt_f) + 1e-9) < 0.01
            except ValueError:
                is_match = gt_norm == ex_norm

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
