"""Hypothesis testing utilities for Task 2 (A/B tests and group comparisons).

Per the brief, the script runs the following tests:
 - Are there risk differences across provinces? (one-way ANOVA)
 - Are there risk differences across zipcodes? (one-way ANOVA)
 - Is there a significant margin (profit) difference between zipcodes? (one-way ANOVA)
 - Is there a significant risk difference between women and men? (two-sample t-test / Mann-Whitney if non-normal)

The script writes a JSON file with results and prints a concise report.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def anova_test(df, value_col, group_col):
    """Run one-way ANOVA using statsmodels and return p-value and table."""
    # drop missing
    sub = df[[value_col, group_col]].dropna()
    if sub[group_col].nunique() < 2:
        return {"error": "need at least two groups"}

    formula = f"{value_col} ~ C({group_col})"
    model = ols(formula, data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_val = anova_table["PR(>F)"].iloc[0]
    return {"anova_table": anova_table.to_dict(), "p_value": float(p_val)}


def ttest_two_groups(df, value_col, group_col, group_a, group_b, use_nonparametric=False):
    sub = df[[value_col, group_col]].dropna()
    a = sub[sub[group_col] == group_a][value_col]
    b = sub[sub[group_col] == group_b][value_col]
    if len(a) < 2 or len(b) < 2:
        return {"error": "not enough observations"}

    if use_nonparametric:
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        test = 'mannwhitneyu'
    else:
        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        test = 'ttest_ind'
    return {"test": test, "statistic": float(stat), "p_value": float(p), "n_a": int(len(a)), "n_b": int(len(b))}


def run_all_tests(df):
    results = {}

    # 1) Risk differences across provinces
    if "province" in df.columns and "total_claims" in df.columns:
        results['province_risk_anova'] = anova_test(df, 'total_claims', 'province')

    # 2) Risk differences across zipcodes
    if "zipcode" in df.columns and "total_claims" in df.columns:
        # if too many zipcodes, sample or aggregate small ones
        results['zipcode_risk_anova'] = anova_test(df, 'total_claims', 'zipcode')

    # 3) Margin difference between zipcodes (margin = premium - total_claims)
    if "zipcode" in df.columns and "premium" in df.columns and "total_claims" in df.columns:
        df = df.copy()
        df['margin'] = df['premium'] - df['total_claims']
        results['zipcode_margin_anova'] = anova_test(df, 'margin', 'zipcode')

    # 4) Risk difference between Women and Men
    if "gender" in df.columns and "total_claims" in df.columns:
        genders = df['gender'].dropna().unique()
        # try t-test between 'M' and 'F' if both present
        if set(['M', 'F']).issubset(set(genders)):
            # quick normality check: use nonparametric if distributions very skewed
            a = df[df['gender'] == 'M']['total_claims']
            b = df[df['gender'] == 'F']['total_claims']
            use_nonparam = (stats.skew(a.dropna()) > 2) or (stats.skew(b.dropna()) > 2)
            results['gender_risk_test'] = ttest_two_groups(df, 'total_claims', 'gender', 'M', 'F', use_nonparametric=use_nonparam)
        else:
            results['gender_risk_test'] = {"error": "M and F not both present"}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("--out", default="results/hypothesis_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.infile)
    results = run_all_tests(df)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: str(o))

    # Print a concise summary
    print("Hypothesis tests completed. Summary:")
    for k, v in results.items():
        print(f" - {k}: {('error: ' + v['error']) if isinstance(v, dict) and 'error' in v else 'done'}")

    # non-zero exit on fatal errors
    sys.exit(0)


if __name__ == "__main__":
    main()
