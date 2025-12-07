import click
import logging
import json
import sys
from ..eda.load_data import load_csv
from ..eda.preprocess import make_loss_ratio, coerce_dtypes
import scipy.stats as stats
import numpy as np




def test_province_anova(df):
# group LossRatio by province and apply ANOVA
    groups = [g['LossRatio'].dropna().values for _, g in df.groupby('Province') if g['LossRatio'].notna().any()]
    if len(groups) < 2:
        return None, None
    stat, p = stats.f_oneway(*groups)
    return stat, p




def test_gender_ttest(df):
    if 'Gender' not in df.columns:
        return None, None
    men = df[df['Gender'].astype(str).str.upper().str.startswith('M')]['LossRatio'].dropna().values
    women = df[df['Gender'].astype(str).str.upper().str.startswith('F')]['LossRatio'].dropna().values
    if len(men) < 2 or len(women) < 2:
        return None, None
    stat, p = stats.ttest_ind(men, women, equal_var=False, nan_policy='omit')
    return stat, p




def test_postalcode_kruskal(df, top_n=10):
# To avoid too many groups, we test top_n postal codes by count
    top = df['PostalCode'].value_counts().head(top_n).index.tolist()
    groups = [df.loc[df['PostalCode']==pc, 'LossRatio'].dropna().values for pc in top]
    if len(groups) < 2:
        return None, None
    stat, p = stats.kruskal(*groups)
    return stat, p




@click.command()
@click.option('--input', 'input_path', required=True, help='Path to input CSV')
@click.option('--alpha', default=0.05, help='Significance level')
def main(input_path: str, alpha: float):
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading data')
    df = load_csv(input_path)
    df = coerce_dtypes(df)
    df = make_loss_ratio(df)


    logging.info('Running ANOVA for provinces')
    s, p = test_province_anova(df)
    logging.info('Province ANOVA stat= %s p=%s -> reject H0? %s', s, p, p < alpha if p is not None else 'n/a')


    logging.info('Running t-test for gender')
    s2, p2 = test_gender_ttest(df)
    logging.info('Gender t-test stat= %s p=%s -> reject H0? %s', s2, p2, p2 < alpha if p2 is not None else 'n/a')


    logging.info('Running Kruskal for PostalCode (top 10)')
    s3, p3 = test_postalcode_kruskal(df, top_n=10)
    logging.info('PostalCode Kruskal stat= %s p=%s -> reject H0? %s', s3, p3, p3 < alpha if p3 is not None else 'n/a')

    # assemble results and print/save as JSON for downstream consumption
    results = {
        'province_anova': {'statistic': None if s is None else float(s), 'p_value': None if p is None else float(p)},
        'gender_ttest': {'statistic': None if s2 is None else float(s2), 'p_value': None if p2 is None else float(p2)},
        'postalcode_kruskal_top10': {'statistic': None if s3 is None else float(s3), 'p_value': None if p3 is None else float(p3)},
    }

    try:
        print(json.dumps(results, indent=2))
    except Exception:
        logging.info('Could not serialize results to JSON')

    # click commands shouldn't return values; exit explicitly
    sys.exit(0)