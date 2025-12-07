import pandas as pd

def load_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load CSV into DataFrame and attempt to parse dates.


    Args:
    path: CSV path
    nrows: optional to load sample
    Returns:
    pd.DataFrame
    """
    df = pd.read_csv(path, nrows=nrows)


    # detect common date columns and parse
    for col in ['TransactionDate', 'TransactionMonth', 'VehicleIntroDate']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                # leave as-is
                pass
    return df