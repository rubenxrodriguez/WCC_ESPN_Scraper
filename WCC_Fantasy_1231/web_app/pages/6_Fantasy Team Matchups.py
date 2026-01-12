# 6_UserTeam_Matchups.py
# Future schedule (Weeks 1..9) independent of gamelog.
# Weekly scoring pulls from gamelog when available (weeknum), otherwise 0.
# Global standings uses gamelog season-to-date totals.

import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="User Team Matchups", page_icon="🗓️", layout="wide")
st.title("🗓️ Fantasy League Matchups & Scoreboard")

# -----------------------------
# Config: season length
# -----------------------------
TOTAL_WEEKS = 9  # future schedule weeks 1..9

# -----------------------------
# Load data
# -----------------------------
fantasystats = pd.read_csv("data/fantasy/fantasy_stats_latest.csv")
gamelog = pd.read_csv("Gamelog/gamelog.csv")


# fantasy_stats_latest mapping columns
fs_player = "Full Name"
fs_userteam = "userteam"

# gamelog columns
gl_player = "Name"
gl_fpts   = "FantasyPts"
gl_week   = "weeknum"


# Normalize
fantasystats[fs_player] = fantasystats[fs_player].astype(str).str.strip()
fantasystats[fs_userteam] = fantasystats[fs_userteam].astype(str).str.strip()

gamelog[gl_player] = gamelog[gl_player].astype(str).str.strip()
gamelog[gl_week] = pd.to_numeric(gamelog[gl_week], errors="coerce").astype("Int64")
gamelog[gl_fpts] = pd.to_numeric(gamelog[gl_fpts], errors="coerce").fillna(0)

# User teams list (exclude FreeAgent)

user_teams = sorted([t for t in fantasystats[fs_userteam].unique().tolist() if t != "FreeAgent"])
if len(user_teams) < 2:
    st.error("Not enough user teams (need at least 2 non-FreeAgent teams).")
    st.stop()

# -----------------------------
# Map Player -> userteam (from fantasy_stats_latest)
# -----------------------------
# we want this to depend on the weekly rostered team 
player_to_userteam = dict(
    fantasystats[fantasystats[fs_userteam] != "FreeAgent"]
    .dropna(subset=[fs_player, fs_userteam])
    .drop_duplicates(subset=[fs_player])[[fs_player, fs_userteam]]
    .values
)

# Attach userteam to gamelog (unrostered -> FreeAgent)
gamelog["userteam"] = gamelog[gl_player].map(player_to_userteam).fillna("FreeAgent")

# -----------------------------
# Global standings (season-to-date from gamelog)
# -----------------------------
st.subheader("🌍 Global Standings (Season-to-date)")

season_totals = (
    gamelog[gamelog["userteam"] != "FreeAgent"]
    .groupby("userteam", dropna=False)[gl_fpts]
    .sum()
    .reset_index()
    .rename(columns={gl_fpts: "FantasyPts_total"})
    .sort_values("FantasyPts_total", ascending=False)
)

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Season weeks", TOTAL_WEEKS)
with c2:
    if len(season_totals):
        st.metric("Current #1 team", f"{season_totals.iloc[0]['userteam']} ({season_totals.iloc[0]['FantasyPts_total']:.1f})")
    else:
        st.metric("Current #1 team", "N/A")

st.dataframe(season_totals.style.format({"FantasyPts_total": "{:,.1f}"}), use_container_width=True)

st.divider()

# -----------------------------
# Generate FULL future schedule (Weeks 1..9) independent of gamelog
# -----------------------------
st.subheader("📋 Full Matchup Schedule (All 9 Weeks)")

if "matchups_full" not in st.session_state:
    schedule = []
    matchup_id = 1

    for wk in range(1, TOTAL_WEEKS + 1):
        teams = list(user_teams)
        random.shuffle(teams)

        # If odd number of teams -> add BYE
        if len(teams) % 2 == 1:
            teams.append("BYE")

        for i in range(0, len(teams), 2):
            schedule.append({
                "Matchup #": matchup_id,
                "Week": wk,
                "Team A": teams[i],
                "Team B": teams[i + 1],
            })
            matchup_id += 1

    st.session_state.matchups_full = pd.DataFrame(schedule)

matchups_df = st.session_state.matchups_full.copy()

# -----------------------------
# Weekly scoring from gamelog (if week exists yet, else 0)
# -----------------------------
def team_points_for_week(team_name: str, week: int) -> float:
    if team_name in (None, "", "BYE"):
        return 0.0
    df = gamelog[(gamelog["userteam"] == team_name) & (gamelog[gl_week] == week)]
    return float(df[gl_fpts].sum())

matchups_df["Team A Pts"] = matchups_df.apply(lambda r: team_points_for_week(r["Team A"], int(r["Week"])), axis=1)
matchups_df["Team B Pts"] = matchups_df.apply(lambda r: team_points_for_week(r["Team B"], int(r["Week"])), axis=1)

def pick_winner(row):
    # If no games yet, you can optionally show None instead of Tie
    if row["Team A"] == "BYE":
        return row["Team B"]
    if row["Team B"] == "BYE":
        return row["Team A"]
    if row["Team A Pts"] > row["Team B Pts"]:
        return row["Team A"]
    if row["Team B Pts"] > row["Team A Pts"]:
        return row["Team B"]
    # If both are 0 and it’s future, you might prefer blank:
    if row["Team A Pts"] == 0 and row["Team B Pts"] == 0:
        return ""
    return "Tie"

matchups_df["Winner"] = matchups_df.apply(pick_winner, axis=1)

# Full schedule table
st.dataframe(
    matchups_df[["Matchup #", "Week", "Team A", "Team B", "Team A Pts", "Team B Pts", "Winner"]]
    .style.format({"Team A Pts": "{:,.1f}", "Team B Pts": "{:,.1f}"}),
    use_container_width=True,
    height=500
)

st.divider()

# -----------------------------
# Scoreboard by week
# -----------------------------
week_input = st.selectbox("Select week to view scoreboard", list(range(1, TOTAL_WEEKS + 1)), index=0)

scoreboard = matchups_df[matchups_df["Week"] == week_input].copy()

st.subheader(f"🏀 Scoreboard — Week {week_input}")
st.dataframe(
    scoreboard[["Matchup #", "Week", "Team A", "Team B", "Team A Pts", "Team B Pts", "Winner"]]
    .style.format({"Team A Pts": "{:,.1f}", "Team B Pts": "{:,.1f}"}),
    use_container_width=True
)

# Weekly standings for selected week (from gamelog)
st.subheader(f"📈 Weekly Standings — Week {week_input}")

weekly_totals = (
    gamelog[(gamelog["userteam"] != "FreeAgent") & (gamelog[gl_week] == week_input)]
    .groupby("userteam", dropna=False)[gl_fpts]
    .sum()
    .reset_index()
    .rename(columns={gl_fpts: "FantasyPts_week"})
    .sort_values("FantasyPts_week", ascending=False)
)

st.dataframe(weekly_totals.style.format({"FantasyPts_week": "{:,.1f}"}), use_container_width=True)






# -----------------------------
# My Players — by Week + User Team (means + sums)
# -----------------------------
st.divider()
st.subheader("👥 My Players — Weekly Boxscore (means + sums)")

# Let user change BOTH filters here
team_input = st.selectbox("Select user team", user_teams, index=0, key="myplayers_team")

# Week options: only weeks that exist in gamelog (plus 1..TOTAL_WEEKS if you want)
week_options = sorted([int(x) for x in gamelog[gl_week].dropna().unique().tolist() if int(x) >= 1])
# Fallback if weeknum missing / empty:
if not week_options:
    week_options = list(range(1, TOTAL_WEEKS + 1))

week_pick = st.selectbox("Select weeknum", week_options, index=0, key="myplayers_week")

# --- Rostered players for that team (from fantasy_stats_latest) ---
rostered = (
    fantasystats[fantasystats[fs_userteam].eq(team_input)]
    [[fs_player]]
    .dropna()
    .drop_duplicates()
    .rename(columns={fs_player: "Player"})
)

# --- Filter gamelog to that team + week ---
week_rows = gamelog[
    (gamelog["userteam"].eq(team_input)) &
    (gamelog[gl_week].eq(int(week_pick)))
].copy()

# If there are no rows yet, still show roster with zeros
if week_rows.empty:
    out = rostered.copy()
    for c in [
        "MIN_mean","PTS_mean","REB_mean","OREB_mean","AST_mean","eFG%","FTM_mean","FTA_mean",
        "3PM_mean","3PA_mean","STL_mean","BLK_mean","TO_mean","FantasyPts_mean","FantasyPts_sum"
    ]:
        out[c] = 0
    st.metric(f"{team_input} — Week {week_pick} total", "0.0")
    st.dataframe(out, use_container_width=True)
    st.stop()

# --- Aggregate per player: means + sums needed for derived rates ---
# We compute eFG% and FT% as *weighted* rates using SUMS in the week.
agg = (
    week_rows.groupby(gl_player, dropna=False)
    .agg(
        games=("Date", "count") if "Date" in week_rows.columns else (gl_fpts, "count"),

        MIN_mean=("MIN", "mean") if "MIN" in week_rows.columns else (gl_fpts, "mean"),
        PTS_mean=("PTS", "mean") if "PTS" in week_rows.columns else (gl_fpts, "mean"),
        REB_mean=("REB", "mean") if "REB" in week_rows.columns else (gl_fpts, "mean"),
        OREB_mean=("OREB", "mean") if "OREB" in week_rows.columns else (gl_fpts, "mean"),
        AST_mean=("AST", "mean") if "AST" in week_rows.columns else (gl_fpts, "mean"),

        FTM_mean=("FTM", "mean") if "FTM" in week_rows.columns else (gl_fpts, "mean"),
        FTA_mean=("FTA", "mean") if "FTA" in week_rows.columns else (gl_fpts, "mean"),
        TPM_mean=("3PM", "mean") if "3PM" in week_rows.columns else (gl_fpts, "mean"),
        TPA_mean=("3PA", "mean") if "3PA" in week_rows.columns else (gl_fpts, "mean"),

        STL_mean=("STL", "mean") if "STL" in week_rows.columns else (gl_fpts, "mean"),
        BLK_mean=("BLK", "mean") if "BLK" in week_rows.columns else (gl_fpts, "mean"),
        TO_mean=("TO", "mean") if "TO" in week_rows.columns else (gl_fpts, "mean"),

        FantasyPts_mean=(gl_fpts, "mean"),
        FantasyPts_sum=(gl_fpts, "sum"),

        # sums for derived rates
        FGM_sum=("FGM", "sum") if "FGM" in week_rows.columns else (gl_fpts, "sum"),
        FGA_sum=("FGA", "sum") if "FGA" in week_rows.columns else (gl_fpts, "sum"),
        TPM_sum=("3PM", "sum") if "3PM" in week_rows.columns else (gl_fpts, "sum"),
        FTM_sum=("FTM", "sum") if "FTM" in week_rows.columns else (gl_fpts, "sum"),
        FTA_sum=("FTA", "sum") if "FTA" in week_rows.columns else (gl_fpts, "sum"),
    )
    .reset_index()
    .rename(columns={gl_player: "Player", "TPM_mean": "3PM_mean", "TPA_mean": "3PA_mean"})
)

# Derived %s (weighted by attempts using sums)
# eFG% = (FGM + 0.5*3PM) / FGA
agg["eFG%"] = 0.0
mask_fga = agg["FGA_sum"].fillna(0).ne(0)
agg.loc[mask_fga, "eFG%"] = (agg.loc[mask_fga, "FGM_sum"] + 0.5 * agg.loc[mask_fga, "TPM_sum"]) / agg.loc[mask_fga, "FGA_sum"]


# Keep only requested columns (and join to roster so players with 0 games show)
keep_cols = [
    "Player",
    "MIN_mean", "PTS_mean", "REB_mean", "OREB_mean", "AST_mean",
    "eFG%",
    "FTM_mean", "FTA_mean",
    "3PM_mean", "3PA_mean",
    "STL_mean", "BLK_mean", "TO_mean",
    "FantasyPts_mean", "FantasyPts_sum",
]

out = (
    rostered.merge(agg[keep_cols], on="Player", how="left")
    .fillna({c: 0 for c in keep_cols if c != "Player"})
    .sort_values("FantasyPts_sum", ascending=False)
    .reset_index(drop=True)
)

st.metric(f"{team_input} — Week {week_pick} total", f"{out['FantasyPts_sum'].sum():.1f}")

st.dataframe(
    out.style.format({
        "MIN_mean": "{:,.1f}",
        "PTS_mean": "{:,.1f}",
        "REB_mean": "{:,.1f}",
        "OREB_mean": "{:,.1f}",
        "AST_mean": "{:,.1f}",
        "eFG%": "{:.3f}",
        "FTM_mean": "{:,.1f}",
        "FTA_mean": "{:,.1f}",
        "3PM_mean": "{:,.1f}",
        "3PA_mean": "{:,.1f}",
        "STL_mean": "{:,.1f}",
        "BLK_mean": "{:,.1f}",
        "TO_mean": "{:,.1f}",
        "FantasyPts_mean": "{:,.1f}",
        "FantasyPts_sum": "{:,.1f}",
    }),
    use_container_width=True
)







# -----------------------------
# Randomize schedule button
# -----------------------------

#if st.button("🔄 Randomize Matchups Again"):
 #   st.session_state.pop("matchups_full", None)
  #  st.rerun()
