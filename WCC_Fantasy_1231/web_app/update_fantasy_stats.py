import pandas as pd
import unidecode
from datetime import date,timedelta
import os
import numpy as np

# --- 0️⃣ Paths ---
DRAFT_PATH = "data/wcc_concat_updated.csv"
RESULTS_PATH = "data/draft_results2.csv"
GAMELOG_PATH = "Gamelog/gamelog.csv" ########## change this 
OUTPUT_DIR = "data/fantasy/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1️⃣ Load core sources ---
draft = pd.read_csv(DRAFT_PATH)
draft_results = pd.read_csv(RESULTS_PATH)
gamelogs = pd.read_csv(GAMELOG_PATH)

# --- 2️⃣ Map player → userteam ---
userteam_map = dict(zip(draft_results['Player'], draft_results['Team']))


# Assign userteam or mark as FreeAgent
draft['userteam'] = draft['Full Name'].map(userteam_map).fillna('FreeAgent')

# --- 3️⃣ Clean gamelog player names ---
def make_clean_name(name):
    if pd.isna(name):
        return ""
    return unidecode.unidecode(name.strip().lower())

gamelogs['clean_name'] = gamelogs['Name'].apply(make_clean_name)

# --- 4️⃣ Aggregate gamelogs ---
agg = gamelogs.groupby('clean_name').agg({
    'FantasyPts': ['sum', 'mean', 'count'],
    'MIN': ['sum', 'mean'],
    'OREB': ['sum', 'mean'],
    'DREB': ['sum', 'mean'],
    'REB': ['sum', 'mean'],
    'AST': ['sum', 'mean'],
    'STL': ['sum', 'mean'],
    'BLK': ['sum', 'mean'],
    'TO': ['sum', 'mean'],
    'FGM': ['sum', 'mean'],
    'FGA': ['sum', 'mean'],
    '3PM': ['sum', 'mean'],
    '3PA': ['sum', 'mean'],
    'FTM': ['sum', 'mean'],
    'FTA': ['sum', 'mean'],
    'PTS': ['sum', 'mean'],
    'didWin':['sum']
}).reset_index()

# Flatten column names
agg.columns = ['_'.join(col).rstrip('_') for col in agg.columns.values]

# --- 5️⃣ Merge aggregated data into draft board ---

fantasy_stats = draft.merge(agg, how='left', on='clean_name')
num_cols = fantasy_stats.select_dtypes(include='number').columns
fantasy_stats[num_cols] = fantasy_stats[num_cols].fillna(0)
#fantasy_stats = draft.merge(agg, how='left', on='clean_name').fillna(0)

fantasy_stats["FGM_sum"] = pd.to_numeric(fantasy_stats["FGM_sum"], errors="coerce")
fantasy_stats["3PM_sum"] = pd.to_numeric(fantasy_stats["3PM_sum"], errors="coerce")
fantasy_stats["FGA_sum"] = pd.to_numeric(fantasy_stats["FGA_sum"], errors="coerce")

fantasy_stats["efg%"] = np.where(
    fantasy_stats["FGA_sum"] > 0,
    (fantasy_stats["FGM_sum"] + 0.5 * fantasy_stats["3PM_sum"]) / fantasy_stats["FGA_sum"],
    np.nan
)

fantasy_stats["efg%"] = fantasy_stats["efg%"].round(3)

# --- 6️⃣ Rankings ---
fantasy_stats['rank_pos'] = fantasy_stats.groupby('Pos.')['FantasyPts_sum'].rank(ascending=False)
fantasy_stats['rank_global'] = fantasy_stats['FantasyPts_sum'].rank(ascending=False)
fantasy_stats['GP'] = fantasy_stats['FantasyPts_count']
# --- 7️⃣ Reorder columns ---
fantasy_stats = fantasy_stats[[
    '#', 'Full Name', 'FullTeamName', 'Pos.', 'Ht.', 'Year', 'Hometown',
    'userteam', 'FantasyPts_sum', 'FantasyPts_mean', 'rank_pos', 'rank_global',
    'GP', 'MIN_mean', 'PTS_mean','efg%', 'REB_mean', 'AST_mean', 'STL_mean',
    'BLK_mean', '3PM_mean', '3PA_mean', 'FTM_mean', 'TO_mean', 'FGM_mean', 'FGA_mean',
    'OREB_sum', 'OREB_mean', 'DREB_sum', 'DREB_mean', 'MIN_sum', 'PTS_sum', 'REB_sum',
    'AST_sum', 'STL_sum', 'BLK_sum', 'TO_sum', 'FGM_sum', 'FGA_sum', '3PM_sum', '3PA_sum',
    'FTM_sum', 'FTA_sum', 'FTA_mean','didWin_sum', 'Previous School', 'clean_name', 'ImageURL'
]]

# --- 8️⃣ Round numbers ---
cols_to_round = fantasy_stats.columns.difference(['efg%'])
fantasy_stats[cols_to_round] = fantasy_stats[cols_to_round].round(2)



# --- 9️⃣ Save outputs ---


fantasy_stats.to_csv(os.path.join(OUTPUT_DIR, "fantasy_stats_latest.csv"), index=False) ########## change this 
#fantasy_stats.to_csv(os.path.join(OUTPUT_DIR, "fantasy_stats_2025-12-28.csv"), index=False)
print('\n'*3)

print(fantasy_stats[['Full Name', 'userteam', 'FantasyPts_sum', 'rank_global']].sort_values('FantasyPts_sum',ascending=False).head(10))
# Round all columns except 'efg%'
