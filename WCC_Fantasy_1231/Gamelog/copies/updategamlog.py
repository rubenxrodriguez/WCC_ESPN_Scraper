import pandas as pd 
df = pd.read_csv("Gamelog/gamelog.csv")

'''

df['efg%'] = (df['FGM'] + 0.5 * df['3PM']) / df['FGA']
df['efg%'] = df['efg%'].round(3)
colorder =['Name', 'Jersey #', 'Team', 'Opponent', 'didWin', 'Date', 'MIN', 'OREB',
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'FGM', 'FGA', '3PM',
       '3PA', 'FTM', 'FTA','efg%', 'PTS', 'FantasyPts', 'isWCC']
df = df[colorder]'''

df['weeknum'] = 0
df2 = df[df['Team']=='Loyola Marymount Lions']
print(df2[['Name','Date','PTS','FGA','efg%','weeknum']].sample(8))
df.to_csv("Gamelog/gamelog2.csv", index=False)