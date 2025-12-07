import os
sns.set(style='whitegrid')




def plot_lossratio_by_province(df: pd.DataFrame, outdir: str):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    grp = df.groupby('Province')['LossRatio'].mean().dropna().sort_values()
    plt.figure(figsize=(10,6))
    sns.barplot(x=grp.values, y=grp.index)
    plt.xlabel('Average Loss Ratio')
    plt.title('Average Loss Ratio by Province')
    plt.tight_layout()
    path = os.path.join(outdir, 'lossratio_by_province.png')
    plt.savefig(path)
    plt.close()
    return path




def plot_monthly_trends(df: pd.DataFrame, outdir: str, date_col: str = 'TransactionDate'):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    if date_col not in df.columns:
        return None
    tmp = df.copy()
    tmp['month'] = tmp[date_col].dt.to_period('M')
    monthly = tmp.groupby('month')[['TotalClaims','TotalPremium']].sum()
    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']
    plt.figure(figsize=(10,6))
    monthly[['TotalClaims','TotalPremium']].plot(title='Monthly Totals', figsize=(10,6))
    plt.tight_layout()
    path1 = os.path.join(outdir, 'monthly_totals.png')
    plt.savefig(path1)
    plt.close()


    plt.figure(figsize=(10,5))
    monthly['LossRatio'].plot(title='Monthly Loss Ratio', marker='o')
    plt.xlabel('Month')
    plt.tight_layout()
    path2 = os.path.join(outdir, 'monthly_lossratio.png')
    plt.savefig(path2)
    plt.close()
    return path1, path2




def plot_top_makes_by_claim(df: pd.DataFrame, outdir: str, top_n: int = 20):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    if 'Make' not in df.columns:
        return None
    grp = df.groupby('Make')['TotalClaims'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,8))
    sns.barplot(x=grp.values, y=grp.index)
    plt.xlabel('Average TotalClaims')
    plt.title(f'Top {top_n} Vehicle Makes by Average Claims')
    plt.tight_layout()
    path = os.path.join(outdir, f'top_{top_n}_makes_by_claims.png')
    plt.savefig(path)
    plt.close()
    return path