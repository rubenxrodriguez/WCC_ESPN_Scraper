import pandas as pd
import numpy as np

df = pd.read_csv("Gamelog/gamelog.csv")

POINTS_DICT_NEW = {
    "FGM": 2,
    "FGA": -1,
    "FTM": 1,
    "FTA": -1,
    "3PM": 2,
    "3PA": -1,
    "OREB": 1,
    "DREB": 1,
    "AST": 3,
    "STL": 3.5,
    "BLK": 3.5,
    "TO": -2.5,
    "PTS": 1.5,
    "didWin":1
}



# Ensure stats are numeric (coerce bad strings to NaN)
for col in POINTS_DICT_NEW:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["NewFP"] = 0.0
for stat, w in POINTS_DICT_NEW.items():
    if stat in df.columns:
        df["NewFP"] += df[stat].fillna(0) * w
        
df["NewFP"] = df["NewFP"].round(0)
df["NewFP"] = df["NewFP"].astype(int)

df2 = pd.read_csv("Gamelog/gamelog.csv")
df2['FantasyPts'] = df['NewFP']
print(df2[['Name','FantasyPts']].head(3))

df2.to_csv("Gamelog/gamelog_updated.csv", index=False)