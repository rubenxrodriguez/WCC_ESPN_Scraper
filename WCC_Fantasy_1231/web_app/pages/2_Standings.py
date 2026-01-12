import streamlit as st
import pandas as pd

st.set_page_config(page_title="Standings", page_icon="📊")

st.title("📊 Global Standings")
#also want to add fantasy individual leaders
# --- Load Data ---
@st.cache_data
def load_fantasy_stats():
    return pd.read_csv("data/fantasy/fantasy_stats_latest.csv")

df = load_fantasy_stats()



# --- Sort and select top 5 ---
leaders = df.sort_values("FantasyPts_sum", ascending=False).head(5)

# --- Layout ---
st.subheader("🏀 Top 5 Fantasy Players")

# Create columns dynamically for visual layout
cols = st.columns(5)

for i, (idx, row) in enumerate(leaders.iterrows()):
    with cols[i]:
        st.image(row["ImageURL"], width=120)
        st.markdown(f"**{row['Full Name']}**")
        st.caption(f"{row['FullTeamName']} | {row['Pos.']}")
        st.write(f"**Fantasy Pts:** {row['FantasyPts_sum']:.1f}")
        st.write(f"User Team: `{row['userteam']}`")

# --- Remove FreeAgents before team aggregation ---
df_userteams = df[df["userteam"] != "FreeAgent"]

# --- Aggregate by userteam ---
team_stats = (
    df_userteams.groupby("userteam", as_index=False)
        .agg({
            "FantasyPts_sum": "sum",
            "FantasyPts_mean": "mean",
            "GP": "sum"
        })
        .sort_values("FantasyPts_sum", ascending=False)
)

team_stats.columns = ["Team", "Total Fantasy Points", "Avg FP/Game", "Games Played"]

# --- Display ---
st.dataframe(team_stats, use_container_width=True)

# Optional: Bar chart
st.bar_chart(data=team_stats, x="Team", y="Total Fantasy Points")

st.divider()
cols = ["Full Name","#", "Pos.", "FullTeamName","GP", "FantasyPts_sum", "FantasyPts_mean", 
         "PTS_mean", "REB_mean", "AST_mean", "STL_mean", "BLK_mean",'rank_pos', 'rank_global']
st.dataframe(df[cols],use_container_width=True)

st.divider()
st.subheader("🏷️ Position Ranks")

# --- Controls ---
pos_options = sorted(df["Pos."].dropna().unique().tolist())
default_positions = [p for p in ["G", "F", "C"] if p in pos_options] or pos_options[:3]

positions = st.multiselect(
    "Positions to display:",
    options=pos_options,
    default=default_positions
)

top_n = st.slider("Players per position:", min_value=5, max_value=25, value=10, step=1)

metric_label = st.selectbox(
    "Rank players by:",
    ["FantasyPts_sum (Total)", "FantasyPts_mean (FP/Game)"]
)
metric_col = "FantasyPts_sum" if "sum" in metric_label else "FantasyPts_mean"

drafted_only = st.checkbox("Drafted only (exclude FreeAgent)", value=False)

# --- Base filter ---
base = df.copy()
if drafted_only and "userteam" in base.columns:
    base = base[base["userteam"].ne("FreeAgent")]

# --- Tabs by position ---
if positions:
    tabs = st.tabs(positions)
    for i, pos in enumerate(positions):
        with tabs[i]:
            pos_df = base[base["Pos."] == pos].sort_values(metric_col, ascending=False).head(top_n)

            show_cols = [c for c in [
                "Full Name", '#','Ht.',"userteam", "FullTeamName", "GP",
                "FantasyPts_sum", "FantasyPts_mean",
                "PTS_mean", "REB_mean", "AST_mean", "STL_mean", "BLK_mean"
            ] if c in pos_df.columns]

            st.dataframe(pos_df[show_cols], use_container_width=True)
else:
    st.info("Select at least one position to display.")

