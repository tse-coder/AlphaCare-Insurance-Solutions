import pandas as pd
import numpy as np

NUMERICAL_FILL = 0


def make_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create LossRatio column safely (TotalClaims / TotalPremium)."""
    df = df.copy()
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
    # avoid division by zero
        df['TotalPremium'] = df['TotalPremium'].replace(0, np.nan)
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    else:
        df['LossRatio'] = np.nan
    return df




def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce sensible dtypes for consistent EDA."""
    df = df.copy()
    # Postal code might be numeric but is categorical
    if 'PostalCode' in df.columns:
        df['PostalCode'] = df['PostalCode'].astype(str).fillna('')
    if 'Province' in df.columns:
        df['Province'] = df['Province'].astype('category')
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype('category')
    return df




def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple descriptive stats for numerical columns."""
    num = df.select_dtypes(include=[np.number])
    return num.describe().T