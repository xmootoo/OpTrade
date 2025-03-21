import pandas as pd

def fill_open_zeros(group: pd.DataFrame) -> pd.DataFrame:
    """Fills any zero values in the 'mid_price' column with the first non-zero value that occurs
    before 9:35 AM within each group of data."""
    if group.iloc[0]['mid_price'] == 0:
        first_nonzero = group[
            (group['mid_price'] != 0) &
            (group['datetime'].dt.time <= pd.Timestamp('09:35').time())
        ]['mid_price'].iloc[0]
        group.loc[group['mid_price'] == 0, 'mid_price'] = first_nonzero
    return group
