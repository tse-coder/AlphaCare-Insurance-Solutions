import click
import logging
from .load_data import load_csv
from .preprocess import make_loss_ratio, coerce_dtypes, basic_stats
from .visualizations import plot_lossratio_by_province, plot_monthly_trends, plot_top_makes_by_claim
import pandas as pd




@click.command()
@click.option('--input', 'input_path', required=True, help='Path to input CSV')
@click.option('--out', 'out_dir', default='plots', help='Directory to save plots')
@click.option('--sample', 'sample_frac', default=0.0, type=float, help='If >0, sample fraction of rows for quick runs')
def main(input_path: str, out_dir: str, sample_frac: float):
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading data')
    df = load_csv(input_path)
    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)


    logging.info('Preprocessing')
    df = coerce_dtypes(df)
    df = make_loss_ratio(df)


    logging.info('Saving descriptive stats')
    stats = basic_stats(df)
    stats_path = f'{out_dir}/descriptive_stats.csv'
    pd.DataFrame(stats).to_csv(stats_path)
    logging.info('Descriptive stats saved to %s', stats_path)


    logging.info('Generating plots...')
    p1 = plot_lossratio_by_province(df, out_dir)
    p2 = plot_monthly_trends(df, out_dir)
    p3 = plot_top_makes_by_claim(df, out_dir, top_n=20)
    logging.info('Plots generated: %s, %s, %s', p1, p2, p3)




if __name__ == '__main__':
    main()