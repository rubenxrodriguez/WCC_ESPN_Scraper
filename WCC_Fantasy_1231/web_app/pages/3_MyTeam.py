import streamlit as st
import pandas as pd

st.set_page_config(page_title="My Team", page_icon="🧍", layout="wide")
st.title("🧍 My Team Roster")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    fantasy = pd.read_csv("data/fantasy/fantasy_stats_latest.csv")
    gamelog = pd.read_csv("Gamelog/gamelog.csv")
    teams = fantasy["userteam"].unique().tolist()
    if "FreeAgent" in teams:
        teams.remove("FreeAgent")
    return fantasy, gamelog, teams

df, gl, teams = load_data()

# -----------------------------
# Select Team
# -----------------------------
team = st.selectbox("Select your team:", teams)

roster = df[df["userteam"] == team].copy()
roster["FantasyPts_sum"] = pd.to_numeric(roster["FantasyPts_sum"], errors="coerce").fillna(0)
roster = roster.sort_values("FantasyPts_sum", ascending=False)

my_players = roster["Full Name"].astype(str).str.strip().unique().tolist()
my_players_set = set(my_players)

# -----------------------------
# (A) Team-level season averages from gamelog (for THIS userteam)
# -----------------------------
st.subheader("📊 Team Season Averages (from Game Log)")

# clean + parse
gl["Name"] = gl["Name"].astype(str).str.strip()
gl["Date_dt"] = pd.to_datetime(gl["Date"], errors="coerce")

# filter to your roster players
team_gl = gl[gl["Name"].isin(my_players_set)].copy()

# ensure numeric
num_cols = ["MIN","PTS","REB","AST","STL","BLK","TO","FGM","FGA","3PM","3PA","FTM","FTA","FantasyPts"]
for c in num_cols:
    if c in team_gl.columns:
        team_gl[c] = pd.to_numeric(team_gl[c], errors="coerce")

def safe_mean(col):
    if col in team_gl.columns and len(team_gl):
        return team_gl[col].dropna().mean()
    return None

def safe_sum(col):
    if col in team_gl.columns and len(team_gl):
        return team_gl[col].fillna(0).sum()
    return None

# weighted shooting % from totals (better than averaging % row-by-row)
fg_pct = None
tp_pct = None
ft_pct = None

if {"FGM","FGA"}.issubset(team_gl.columns):
    fgm_sum = team_gl["FGM"].fillna(0).sum()
    fga_sum = team_gl["FGA"].fillna(0).sum()
    fg_pct = (fgm_sum / fga_sum) if fga_sum > 0 else None

if {"3PM","3PA"}.issubset(team_gl.columns):
    tpm_sum = team_gl["3PM"].fillna(0).sum()
    tpa_sum = team_gl["3PA"].fillna(0).sum()
    tp_pct = (tpm_sum / tpa_sum) if tpa_sum > 0 else None

if {"FTM","FTA"}.issubset(team_gl.columns):
    ftm_sum = team_gl["FTM"].fillna(0).sum()
    fta_sum = team_gl["FTA"].fillna(0).sum()
    ft_pct = (ftm_sum / fta_sum) if fta_sum > 0 else None

# metrics row
m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = st.columns(10)
m1.metric("GP rows", int(team_gl.shape[0]))
m2.metric("MIN", f"{safe_mean('MIN'):.1f}" if safe_mean("MIN") is not None else "—")
m3.metric("PTS", f"{safe_mean('PTS'):.1f}" if safe_mean("PTS") is not None else "—")
m4.metric("REB", f"{safe_mean('REB'):.1f}" if safe_mean("REB") is not None else "—")
m5.metric("AST", f"{safe_mean('AST'):.1f}" if safe_mean("AST") is not None else "—")
m6.metric("STL", f"{safe_mean('STL'):.1f}" if safe_mean("STL") is not None else "—")
m7.metric("BLK", f"{safe_mean('BLK'):.1f}" if safe_mean("BLK") is not None else "—")
m8.metric("TO",  f"{safe_mean('TO'):.1f}"  if safe_mean("TO")  is not None else "—")
m9.metric("Fantasy", f"{safe_mean('FantasyPts'):.1f}" if safe_mean("FantasyPts") is not None else "—")

# optional summary table
summary_df = pd.DataFrame([{
    "FantasyPts_mean": safe_mean("FantasyPts"),
    "FantasyPts_sum": safe_sum("FantasyPts")
}])

st.dataframe(
    summary_df.style.format({
        "FantasyPts_mean": "{:,.2f}",
        "FantasyPts_sum": "{:,.1f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.divider()

# -----------------------------
# Roster cards (your original layout)
# -----------------------------
st.subheader("👥 Roster")

cols = st.columns(4)
for i, (_, row) in enumerate(roster.iterrows()):
    with cols[i % 4]:
        st.image(row["ImageURL"], width=120)
        st.markdown(f"**{row['Full Name']}**")
        st.caption(f"{row['FullTeamName']} | {row['Pos.']}")
        st.write(f"**Fantasy Pts Mean:** {row['FantasyPts_mean']:.1f}")
        st.write(f"**Fantasy Pts Sum:** {row['FantasyPts_sum']:.1f}")
        st.write(f"Position Rank: `{row['rank_pos']}`")
        st.write(f"Global Rank: `{row['rank_global']}`")

st.divider()

# -----------------------------
# Roster table (your original table)
# -----------------------------
st.dataframe(
    roster[
        ["Full Name", "Pos.", "FullTeamName", "GP",
         "FantasyPts_sum", "FantasyPts_mean",
         "PTS_mean", 'efg%',"REB_mean", "AST_mean", "STL_mean", "BLK_mean",
         "rank_pos", "rank_global"]
    ],
    use_container_width=True
)

st.divider()

# -----------------------------
# (B) Individual game log tracker
# -----------------------------
st.subheader("📈 Player Game Log Tracker")

player = st.selectbox("Player", my_players)

player_logs = gl[gl["Name"] == player].copy()
player_logs["Date_dt"] = pd.to_datetime(player_logs["Date"], errors="coerce")

if not player_logs.empty and 'ImageURL' in df.columns:
    player_image_url = df[df['Full Name'] == player]['ImageURL'].iloc[0]
    if pd.notna(player_image_url):
        st.image(player_image_url, width=120)



if not player_logs.empty:
    st.subheader(f"📈 {player} — Game Log")

    st.dataframe(
        player_logs[
            ["Date", "Team", "Opponent", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TO",
             "FantasyPts", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA"]
        ].sort_values("Date", ascending=False),
        use_container_width=True
    )

    st.line_chart(
        player_logs.sort_values("Date"),
        x="Date", y="FantasyPts",
        use_container_width=True
    )
else:
    st.info("No game logs found for this player yet.")

st.divider()

# -----------------------------
# Export roster
# -----------------------------
csv = roster.to_csv(index=False).encode("utf-8")
st.download_button("Download Team CSV", csv, f"{team}_roster.csv", "text/csv")
