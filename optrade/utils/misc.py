import pandas as pd
import random
import string
import time
import os


def format_time_dynamic(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}min {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}hr {minutes}min {seconds}s"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}hr {minutes}min {seconds}s"


def fill_open_zeros(group: pd.DataFrame) -> pd.DataFrame:
    """Fills any zero values in the 'mid_price' column with the first non-zero value that occurs
    before 9:35 AM within each group of data."""
    if group.iloc[0]["mid_price"] == 0:
        first_nonzero = group[
            (group["mid_price"] != 0)
            & (group["datetime"].dt.time <= pd.Timestamp("09:35").time())
        ]["mid_price"].iloc[0]
        group.loc[group["mid_price"] == 0, "mid_price"] = first_nonzero
    return group


def generate_random_id(length=10):
    # Seed the random number generator with current time and os-specific random data
    random.seed(int(time.time() * 1000) ^ os.getpid())

    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))
