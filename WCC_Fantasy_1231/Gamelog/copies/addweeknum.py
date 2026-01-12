import pandas as pd 
import numpy as np

# Define your season boundaries
WEEK0_END   = pd.Timestamp("2025-12-27")   # everything <= this is week 0
WEEK1_START = pd.Timestamp("2025-12-28")   # week 1 starts Sunday 12/28
SEASON_END  = pd.Timestamp("2026-02-28")   # season ends Saturday 2/28

def add_week_num_from_date(date_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series, errors="coerce")

    weeknum = pd.Series(pd.NA, index=d.index, dtype="Int64")

    # Week 0
    weeknum[d.le(WEEK0_END)] = 0

    # Weeks 1+ (Sunday->Saturday blocks)
    in_season = d.ge(WEEK1_START) & d.le(SEASON_END)
    weeknum[in_season] = ((d[in_season] - WEEK1_START).dt.days // 7) + 1

    return weeknum

df = pd.read_csv("Gamelog/gamelog-1228-0104.csv")
df["weeknum"] = add_week_num_from_date(df["Date"])
df.to_csv("Gamelog/gamelog-1228-0104-withweeknum.csv", index=False)