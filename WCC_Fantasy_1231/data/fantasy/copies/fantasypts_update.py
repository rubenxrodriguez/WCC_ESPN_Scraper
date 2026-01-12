import pandas as pd
import numpy as np

df = pd.read_csv("Gamelog/copies/fantasy_stats_latest2.csv")

POINTS_DICT_NEW = {
    "FGM_sum": 2,
    "FGA_sum": -1,
    "FTM_sum": 1,
    "FTA_sum": -1,
    "3PM_sum": 2,
    "3PA_sum": -1,
    "OREB_sum": 1,
    "DREB_sum": 1,
    "AST_sum": 3,
    "STL_sum": 3.5,
    "BLK_sum": 3.5,
    "TO_sum": -2.5,
    "PTS_sum": 1.5,
}


'''
print(
    df[["Full Name","FantasyPts_sum","FantasyPts_mean"]]
    .sort_values(by="FantasyPts_mean", ascending=False)
    .head(20)
    .round({"FantasyPts_sum":1,"FantasyPts_mean":2})
)
'''
# Ensure stats are numeric (coerce bad strings to NaN)
for col in POINTS_DICT_NEW:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["NewFP_sum"] = 0.0
for stat, w in POINTS_DICT_NEW.items():
    if stat in df.columns:
        df["NewFP_sum"] += df[stat].fillna(0) * w
        df["NewFP_sum"] = df["NewFP_sum"].round(0)

df["NewFP_mean"] = np.where(df["GP"].fillna(0) > 0, df["NewFP_sum"] / df["GP"], np.nan)

# Helpful comparison columns
df["Delta_sum"]  = df["NewFP_sum"]  - df["FantasyPts_sum"]
df["Delta_mean"] = df["NewFP_mean"] - df["FantasyPts_mean"]

print(
    df[["Full Name","FullTeamName","FantasyPts_sum","NewFP_sum","FantasyPts_mean","NewFP_mean","PTS_mean"]]
    .sort_values(by="NewFP_mean", ascending=False)
    .head(35)
    .round({"FantasyPts_sum":1,"NewFP_sum":1,"FantasyPts_mean":2,"NewFP_mean":2,"PTS_mean":1})
)
#print(df['FantasyPts_mean'].describe())
print('\n\n')
#print(df['NewFP_mean'].describe())
df2 = pd.read_csv("Gamelog/copies/fantasy_stats_latest2.csv")
df2['FantasyPts_sum'] = df['NewFP_sum']
df2['FantasyPts_mean'] = df['NewFP_mean'].round(2)

df2.to_csv("Gamelog/copies/fantasy_stats_updated.csv", index=False)