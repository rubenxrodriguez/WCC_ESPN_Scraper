import pandas as pd 

df = pd.read_csv("data/fantasy/fantasy_stats_latest.csv")

#df = df.sort_values(by="FantasyPts_sum",ascending = False).head(100)
#print(df['Pos.'].value_counts())

df['fgpct'] = df['FGM_sum'] / df['FGA_sum']
df['fgpct'] = df['fgpct'].round(3)
df = df[df['FGA_sum'] >= 100]

df = df.sort_values(by="fgpct", ascending=False).head(20)
#print(df[['Full Name','FGM_sum','FGA_sum','fgpct']])

df2 = pd.read_csv("data/fantasy/fantasy_stats_latest.csv")
df2 = df2.sort_values(by="PTS_mean", ascending=False).head(20)
#print(df2[['Full Name','PTS_mean']])
print(df2['FullTeamName'].unique())