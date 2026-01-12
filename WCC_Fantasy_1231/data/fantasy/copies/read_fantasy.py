import pandas as pd 
fantasy = pd.read_csv("data/fantasy/fantasy_stats_copy.csv")

print(fantasy[['Full Name', 'userteam', 'FantasyPts_sum', 'rank_global']].sort_values('FantasyPts_sum',ascending=False).head(10))
