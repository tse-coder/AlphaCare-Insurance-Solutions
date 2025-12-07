"""Simple EDA utilities for the ACIS insurance dataset.

Usage:
    python src/eda.py data/sample_claims.csv --out results/eda_summary.csv
"""
import argparse
import pandas as pd
import os


def summary_by_group(df, group_cols, agg_col="total_claims"):
    out = df.groupby(group_cols)[agg_col].agg(["count", "sum", "mean", "median", "std"]).reset_index()
    return out


def run_eda(infile, outdir="results"):
    df = pd.read_csv(infile)
    os.makedirs(outdir, exist_ok=True)

    # basic dataset info
    summary = {
        "n_rows": len(df),
        "n_unique_policies": int(df["policy_id"].nunique()) if "policy_id" in df.columns else None,
        "columns": list(df.columns),
    }

    # summaries
    province_summary = summary_by_group(df, ["province"]) if "province" in df.columns else None
    zipcode_summary = summary_by_group(df, ["zipcode"]) if "zipcode" in df.columns else None

    # save
    if province_summary is not None:
        province_summary.to_csv(os.path.join(outdir, "province_summary.csv"), index=False)
    if zipcode_summary is not None:
        zipcode_summary.to_csv(os.path.join(outdir, "zipcode_summary.csv"), index=False)

    # overall summary CSV with sample rows
    df.head(100).to_csv(os.path.join(outdir, "sample_head.csv"), index=False)

    print("EDA complete. Results saved to:")
    print(f" - {outdir}/province_summary.csv")
    print(f" - {outdir}/zipcode_summary.csv")
    print(f" - {outdir}/sample_head.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("--out", default="results/eda")
    args = parser.parse_args()
    run_eda(args.infile, args.out)


if __name__ == "__main__":
    main()
