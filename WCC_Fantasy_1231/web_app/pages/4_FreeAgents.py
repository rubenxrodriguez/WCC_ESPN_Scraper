import streamlit as st
import pandas as pd

st.set_page_config(page_title="Free Agents", page_icon="🧾")

st.title("🧾 Available Free Agents")

# --- Load Data ---
@st.cache_data
def load_fantasy_stats():
    return pd.read_csv("data/fantasy/fantasy_stats_latest.csv")

df = load_fantasy_stats()

free_agents = df[df["userteam"] == "FreeAgent"].sort_values("FantasyPts_mean", ascending=False)
top_free_agents = free_agents.head(5)
st.markdown("### Top Available Free Agents")
# Create columns dynamically for visual layout
cols = st.columns(5)

for i, (idx, row) in enumerate(top_free_agents.iterrows()):
    with cols[i]:
        st.image(row["ImageURL"], width=120)
        st.markdown(f"**{row['Full Name']}**")
        st.caption(f"{row['FullTeamName']} | {row['Pos.']} | #{row['#']}")
        st.write(f"**Fantasy Pts:** {row['FantasyPts_sum']:.1f}")
        st.write(f"Fantasy Pts Per Game: `{row['FantasyPts_mean']:.1f}`")



st.divider()
st.markdown("### All Available Free Agents")
st.dataframe(
    free_agents[
        ["Full Name", "Pos.", "FullTeamName", "FantasyPts_sum", "FantasyPts_mean", 
         "PTS_mean", "REB_mean", "AST_mean",'rank_pos', 'rank_global']
    ],
    use_container_width=True
)
