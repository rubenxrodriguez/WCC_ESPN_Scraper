# 7_WCC_Team_Rosters.py
# CSV-only version with TWO stat views (toggle):
#   1) "Fantasy" (computed from Gamelog aggregates, season-to-date or selected weeks)
#   2) "Season averages" (directly from fantasy_stats_latest.csv)
#
# Uses:
#   - Gamelog/gamelog.csv
#   - data/fantasy/fantasy_stats_latest.csv

import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
GAMELOG_CSV = "Gamelog/gamelog.csv"
FANTASY_LATEST_CSV = "data/fantasy/fantasy_stats_latest.csv"

WCC_FULLTEAMNAMES = {
    "Santa Clara Broncos",
    "Loyola Marymount Lions",
    "Oregon State Beavers",
    "Gonzaga Bulldogs",
    "San Francisco Dons",
    "Washington State Cougars",
    "Pacific Tigers",
    "Pepperdine Waves",
    "Saint Mary's Gaels",
    "Seattle U Redhawks",
    "Portland Pilots",
    "San Diego Toreros",
}

# Week rules
WEEK0_END = pd.Timestamp("2025-12-27")
WEEK1_START = pd.Timestamp("2025-12-28")  # Sunday
SEASON_END = pd.Timestamp("2026-02-28")   # Saturday

st.set_page_config(page_title="WCC Team Rosters", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_weeknum_from_date(date_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series, errors="coerce")
    weeknum = pd.Series(pd.NA, index=d.index, dtype="Int64")

    weeknum[d.le(WEEK0_END)] = 0

    in_season = d.ge(WEEK1_START) & d.le(SEASON_END)
    weeknum[in_season] = ((d[in_season] - WEEK1_START).dt.days // 7) + 1

    return weeknum


def aggregate_gamelog(
    gl: pd.DataFrame,
    player_col: str,
    team_col: str,
    fpts_col: str,
    mp_col: str | None,
    weeks: list[int] | None = None,
) -> pd.DataFrame:
    df = gl.copy()

    if weeks is not None:
        df = df[df["weeknum"].isin(weeks)]

    df[fpts_col] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0)

    if mp_col:
        df[mp_col] = pd.to_numeric(df[mp_col], errors="coerce")

    group_cols = [player_col, team_col]

    agg_dict = {fpts_col: "sum"}
    if mp_col:
        agg_dict[mp_col] = "sum"

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={
            fpts_col: "FantasyPts_sum",
            **({mp_col: "MP_sum"} if mp_col else {}),
        })
    )

    games = df.groupby(group_cols, dropna=False).size().reset_index(name="Games")
    out = out.merge(games, on=group_cols, how="left")

    out["FantasyPts_per_game"] = (out["FantasyPts_sum"] / out["Games"]).where(out["Games"] > 0, 0)

    if mp_col:
        out["MP_per_game"] = (out["MP_sum"] / out["Games"]).where(out["Games"] > 0, 0)

    return out


# -----------------------------
# Load CSVs
# -----------------------------
st.title("WCC Team Rosters")
st.caption("Browse rosters by team and view either Fantasy totals (from gamelog) or Season averages (from fantasy_stats_latest).")

try:
    gamelog = pd.read_csv(GAMELOG_CSV)
except Exception as e:
    st.error(f"Could not read {GAMELOG_CSV}: {e}")
    st.stop()

try:
    latest = pd.read_csv(FANTASY_LATEST_CSV)
except Exception as e:
    st.error(f"Could not read {FANTASY_LATEST_CSV}: {e}")
    st.stop()

# Identify key columns
gl_player = _pick_col(gamelog, [ "Name"])
gl_team   = _pick_col(gamelog, ["Team"])
gl_date   = _pick_col(gamelog, ["Date"])
gl_fpts   = _pick_col(gamelog, ["FantasyPts"])
gl_mp     = _pick_col(gamelog, ["MP"])

lt_player = _pick_col(latest, ["Full Name"])
lt_team   = _pick_col(latest, ["FullTeamName" ])

missing = []
if not gl_player: missing.append("gamelog player column")
if not gl_team:   missing.append("gamelog team column")
if not gl_date:   missing.append("gamelog date column")
if not gl_fpts:   missing.append("gamelog fantasy points column")
if not lt_player: missing.append("fantasy_stats_latest player column")
if not lt_team:   missing.append("fantasy_stats_latest team column")

if missing:
    st.error("Missing required columns:\n- " + "\n- ".join(missing))
    st.stop()

# Normalize strings
gamelog[gl_player] = gamelog[gl_player].astype(str).str.strip()
gamelog[gl_team] = gamelog[gl_team].astype(str).str.strip()
latest[lt_player] = latest[lt_player].astype(str).str.strip()
latest[lt_team] = latest[lt_team].astype(str).str.strip()

# weeknum (use existing if present, else compute)
wk_col = _pick_col(gamelog, ["weeknum", "week_num", "Week", "week"])
if wk_col:
    if wk_col != "weeknum":
        gamelog = gamelog.rename(columns={wk_col: "weeknum"})
else:
    gamelog["weeknum"] = add_weeknum_from_date(gamelog[gl_date])

# Build roster list for each team from fantasy_stats_latest
rosters = latest[[lt_player, lt_team]].dropna().drop_duplicates().rename(
    columns={lt_player: "Player", lt_team: "Team"}
)
rosters["Player"] = rosters["Player"].astype(str).str.strip()
rosters["Team"] = rosters["Team"].astype(str).str.strip()

# -----------------------------
# Controls
# -----------------------------
c1, c2, c3, c4, c5 = st.columns([1.1, 1.9, 1.6, 1.8, 2.2])

with c1:
    wcc_only = st.checkbox("WCC only", value=True)

with c2:
    show_season_avgs = st.checkbox("Show season averages (from fantasy_stats_latest)", value=False)
    week_mode = st.toggle("This week only", value=False, disabled=show_season_avgs)

team_list = sorted(rosters["Team"].unique().tolist())
if wcc_only:
    team_list = [t for t in team_list if t in WCC_FULLTEAMNAMES]
    if not team_list:
        st.warning("No teams matched WCC filter from fantasy_stats_latest; showing all teams.")
        team_list = sorted(rosters["Team"].unique().tolist())

with c3:
    selected_team = st.selectbox("Team", team_list, index=0 if team_list else None)

weeks_available = sorted([int(x) for x in gamelog["weeknum"].dropna().unique().tolist()])
with c4:
    selected_weeks = None
    if (not show_season_avgs) and week_mode:
        default_weeks = weeks_available[-1:] if weeks_available else []
        selected_weeks = st.multiselect("Week(s)", weeks_available, default=default_weeks)

# Metric options depend on view
if show_season_avgs:
    # Pick likely season-average columns that might exist in your fantasy_stats_latest
    candidates = [
        "FantasyPts", "FantasyPts/G", "FantasyPts_per_game",
        "PTS", "REB", "AST", "MP",
        "FG%", "eFG%", "FT%", "3P%", "2P%",
        "STL", "BLK", "TO",
    ]
    metric_options = [c for c in candidates if c in latest.columns]
    if not metric_options:
        # fallback: just let user pick any numeric column besides identifiers
        metric_options = [c for c in latest.columns if c not in {lt_player, lt_team, "Player", "Team"}]
else:
    metric_options = ["FantasyPts_sum", "FantasyPts_per_game", "Games"]
    if gl_mp:
        metric_options += ["MP_sum", "MP_per_game"]

with c5:
    sort_metric = st.selectbox("Sort by", metric_options, index=0 if metric_options else None)
    sort_desc = st.toggle("Descending", value=True)

# -----------------------------
# Build output table
# -----------------------------
if show_season_avgs:
    # Season averages view from fantasy_stats_latest
    team_latest = latest[latest[lt_team] == selected_team].copy()
    team_latest = team_latest.rename(columns={lt_player: "Player", lt_team: "Team"})
    team_latest["Player"] = team_latest["Player"].astype(str).str.strip()
    team_latest["Team"] = team_latest["Team"].astype(str).str.strip()

    team_roster = rosters[rosters["Team"] == selected_team].copy()

    # Merge roster list + season stats
    out = team_roster.merge(team_latest, on=["Player", "Team"], how="left")

else:
    # Fantasy view from gamelog aggregates
    gl_team_df = gamelog[gamelog[gl_team] == selected_team].copy()

    agg = aggregate_gamelog(
        gl=gl_team_df,
        player_col=gl_player,
        team_col=gl_team,
        fpts_col=gl_fpts,
        mp_col=gl_mp,
        weeks=selected_weeks if week_mode else None,
    ).rename(columns={gl_player: "Player", gl_team: "Team"})

    team_roster = rosters[rosters["Team"] == selected_team].copy()
    out = team_roster.merge(agg, on=["Player", "Team"], how="left")

    # Fill missing for players with no games in the window
    for col in ["FantasyPts_sum", "FantasyPts_per_game", "Games", "MP_sum", "MP_per_game"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)

# Sort (if column exists)
if sort_metric and sort_metric in out.columns:
    # try numeric sort when possible
    out[sort_metric] = pd.to_numeric(out[sort_metric], errors="ignore")
    out = out.sort_values(by=sort_metric, ascending=not sort_desc)

# -----------------------------
# Display
# -----------------------------
st.subheader(f"{selected_team} — {'Season averages' if show_season_avgs else ('Weekly view' if week_mode else 'Season to date')}")
if (not show_season_avgs) and week_mode:
    st.caption(f"Weeks selected: {selected_weeks if selected_weeks else 'None'}")


# -----------------------------
# Derived shooting percentages (season averages view)
# -----------------------------
if show_season_avgs:

    # 3P%
    if {"3PM_mean", "3PA_mean"}.issubset(out.columns):
        out["3P%"] = out["3PM_mean"] / out["3PA_mean"]
        out.loc[out["3PA_mean"] == 0, "3P%"] = pd.NA

    # FT%
    if {"FTM_mean", "FTA_mean"}.issubset(out.columns):
        out["FT%"] = out["FTM_mean"] / out["FTA_mean"]
        out.loc[out["FTA_mean"] == 0, "FT%"] = pd.NA

# Decide which columns to show (Team hidden because team is chosen already)
if show_season_avgs:
    avg_cols = [c for c in [
        "MIN_mean", "PTS_mean", "REB_mean","OREB_mean", "AST_mean",
        "efg%","FTM_mean", "FTA_mean", "FT%","3PM_mean", "3PA_mean", "3P%",
        "STL_mean", "BLK_mean", "TO_mean",
        "FantasyPts_mean","FantasyPts_sum",
    ] if c in out.columns]

    # If none of those exist, show all columns except Team (still hide Team)
    if not avg_cols:
        avg_cols = [c for c in out.columns if c not in {"Team"}]

    display_cols = ["Player"] + [c for c in avg_cols if c != "Player" and c != "Team"]

else:
    display_cols = ["Player"] + [c for c in ["Games", "FantasyPts_sum", "FantasyPts_per_game", "MP_sum", "MP_per_game"] if c in out.columns]

show_df = out[display_cols].copy()

# Formatting
fmt = {}

# Fantasy view formats
if "FantasyPts_sum" in show_df.columns:
    fmt["FantasyPts_sum"] = "{:,.1f}"
if "FantasyPts_per_game" in show_df.columns:
    fmt["FantasyPts_per_game"] = "{:,.2f}"
if "Games" in show_df.columns:
    fmt["Games"] = "{:.0f}"
if "MP_sum" in show_df.columns:
    fmt["MP_sum"] = "{:,.0f}"
if "MP_per_game" in show_df.columns:
    fmt["MP_per_game"] = "{:,.1f}"

# Season averages formats
for pct_col in ["3P%", "FT%", "efg%"]:
    if pct_col in show_df.columns:
        fmt[pct_col] = "{:.1%}"   # because values are 0–1

for col in [
    "MIN_mean", "PTS_mean", "REB_mean", "OREB_mean", "AST_mean",
    "STL_mean", "BLK_mean", "TO_mean", "FantasyPts_mean","FTM_mean", "FTA_mean",
    "3PM_mean", "3PA_mean"
]:
    if col in show_df.columns:
        fmt[col] = "{:,.2f}"


# Render
try:
    st.dataframe(show_df.style.format(fmt), use_container_width=True, height=650)
except Exception:
    st.dataframe(show_df, use_container_width=True, height=650)

# Totals
t1, t2, t3 = st.columns(3)
with t1:
    st.metric("Roster players", int(team_roster.shape[0]))
with t2:
    if show_season_avgs:
        # best-effort total if FantasyPts exists
        tot = pd.to_numeric(out.get("FantasyPts_mean", 0), errors="coerce").fillna(0).mean() if "FantasyPts_mean" in out.columns else 0.0
        st.metric("Average FantasyPts (if available)", round(float(tot),2))
    else:
        st.metric("Total FantasyPts", float(out["FantasyPts_sum"].sum()) if "FantasyPts_sum" in out.columns else 0.0)
with t3:
    if show_season_avgs:
        st.metric("Rows in team stats", int(out.shape[0]))
    else:
        st.metric("Total Games Logged", int(pd.to_numeric(out["Games"], errors="coerce").fillna(0).sum()) if "Games" in out.columns else 0)
