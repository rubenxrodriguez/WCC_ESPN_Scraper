import streamlit as st
import pandas as pd
import glob
import random


st.set_page_config(page_title="WCC Analytics Dashboard", layout="wide")
st.title("🏀 WCC Analytics Dashboard")

# --- Load conference player list ---
all_players = pd.read_csv("WCC_Fantasy_1231/data/wcc_concat_updated.csv")
draft_board = pd.read_csv("WCC_Fantasy_1231/data/fantasy/fantasy_stats_2025-12-27.csv")


# --- ADP based on FantasyPts_mean up to 2025-12-27 ---
draft_board["FantasyPts_mean"] = pd.to_numeric(draft_board["FantasyPts_mean"], errors="coerce").fillna(0)

# Higher FantasyPts_mean = earlier pick => ADP 1 is best
draft_board["ADP"] = draft_board["FantasyPts_mean"].rank(ascending=False, method="min").astype(int)

# Make sure we can join reliably
for df in (all_players, draft_board):
    if "clean_name" in df.columns:
        df["clean_name"] = df["clean_name"].fillna("").astype(str).str.strip()

# Overwrite old ADP in all_players with the new one
# Prefer joining on clean_name if available, else fall back to Full Name
if "clean_name" in all_players.columns and "clean_name" in draft_board.columns:
    all_players = all_players.drop(columns=["ADP"], errors="ignore").merge(
        draft_board[["clean_name", "ADP", "FantasyPts_mean","PTS_mean"]],
        on="clean_name",
        how="left"
    )
else:
    all_players = all_players.drop(columns=["ADP"], errors="ignore").merge(
        draft_board[["Full Name", "ADP", "FantasyPts_mean","PTS_mean"]],
        on="Full Name",
        how="left"
    )

# Any players with no games yet go to the bottom
all_players["ADP"] = pd.to_numeric(all_players["ADP"], errors="coerce")
all_players = all_players.sort_values(["ADP", "Full Name"], na_position="last").reset_index(drop=True)




# --- Section 1: Filter by Team ---

# Convert Team column to string, replace NaN with empty string
all_players["Team"] = all_players["Team"].fillna("").astype(str)

teams_list = sorted(all_players["Team"].unique())

# --- Section 2: Top Draft Prospects ---
st.markdown("## 🌟 Top Players")
top15 = all_players.nsmallest(15, "ADP")  # 15 players

# Display in rows of 5
top_list = top15.reset_index(drop=True)
# Reduce column gap by using the 'gap' argument in st.columns (Streamlit >= 1.25)

for i in range(0, len(top_list), 5):
    row_players = top_list.iloc[i:i+5]  # slice 5 players for this row
    cols = st.columns(5, gap="small")
    for idx, (_, player) in enumerate(row_players.iterrows()):
        with cols[idx]:
            st.image(player["ImageURL"], width=100)
            st.markdown(f"**{player['Full Name']}**")
            st.markdown(f"ADP: {player['ADP']}")
            st.markdown(f"{player['Pos.']} | {player['Ht.']} | {player['Year']}")
            st.markdown(f"Fantasy Pts Average: {player['FantasyPts_mean'] if 'FantasyPts_mean' in player else 'N/A'}")
st.markdown("---")


# --- Section 3: Special Categories ---
st.markdown("---")
st.markdown("## ⚡ Notable Players by Category")

# Define categories
power5transfers = ['Irune Orio', 'Charlece Ohiaeri', 'Lizzy Williamson', 'Néné Sow',
                   'Lova Lagerlid', 'Lucija Milkovic', 'Ivana Krajina']

nostat_upperclassmen = ['Hannah Burg', 'Fia Proctor', 'Lucy Larson', 'Sydnie Rodriguez',
                        'Tamia Stricklin', 'Florence Dallow', 'Paula Tirado', 'Kenlee Durrill',
                        'Mackenzie Shivers', 'Addi Wedin', 'Caitlin Monahan', 'Lauren Whittaker',
                        'Keandra Koorits', 'Shorna Preston', 'Kayla Jones', 'Meghan McIntyre',
                         'Bella Green']

preseason_team = ['Dyani Ananiev','Zeryhia Aokuso','Tiara Bolden','Catarina Ferreira',
                  'Sophie Glancey','Maya Hernandez','Maia Jones','Kennedie Shuler','Allie Turner','Eleonora Villa']

above63 = ['Alex Covill', 'Marina Radocaj', 'Georgia Grigoropoulou', 'Carly Heidger',
           'Lauren Glazier', 'Edie Clarke', 'Andjela Bigovic', 'Ella Zimmerman',
           'Lizzy Williamson', 'Néné Sow', 'LaMiracle Lebon', 'Natasa Tausova',
           'Tahara Magassa', 'Jada Kennedy', 'Julia Dalan', 'Lara Alonso-Basurto',
           'Lauren Whittaker', 'Lucija Milkovic', 'Elisa Mehyar']

categories = {
    "Power 5 Transfers": power5transfers,
    "Preseason All-Conference Team": preseason_team,
    "No Stats Upperclassmen": nostat_upperclassmen,
    "Players Taller than 6-2": above63
}
# Create expanders vertically
for cat_name, player_list in categories.items():
    with st.expander(f"🔽 {cat_name}", expanded=False):
        filtered = all_players[all_players["Full Name"].isin(player_list)].copy()
        filtered["ADP"] = pd.to_numeric(filtered["ADP"], errors="coerce")
        filtered = filtered.sort_values("ADP", na_position="last")

        if filtered.empty:
            st.markdown("_No players found in this category._")
        else:
            # Create 3 columns inside each category
            cols = st.columns(3)
            for idx, (_, player_row) in enumerate(filtered.iterrows()):
                with cols[idx % 3]:
                    st.image(player_row["ImageURL"], width=50)
                    st.markdown(
                        f"**{player_row['Full Name']}**  \n"
                        f"{player_row['Pos.']} | {player_row['Ht.']} | {player_row['Year']}  \n"
                        f"Team: {player_row.get('FullTeamName', player_row.get('Team', 'N/A'))}  \n"
                        f"Fantasy Pts Average: {player_row['FantasyPts_mean'] if'FantasyPts_mean' in player else 'N/A' }  \n"
                        #st.markdown(f"Fantasy Pts Average: {player['FantasyPts_mean'] if 'FantasyPts_mean' in player else 'N/A'}")
                        f"**ADP:** {player_row['ADP']}"
                    )

#------------------------------------------------------------------------------------------------------------------
# --- Section 4: 🏀 Draft Room ---

st.markdown("## 🏀 Draft Room")

# --- Initialize Session State ---
if "drafted_players" not in st.session_state:
    st.session_state["drafted_players"] = {
        "UserTeam1": [],
        "UserTeam2": [],
        "UserTeam3": [],
        "UserTeam4": [],
        "RR": [],
        "UserTeam6": [],
    }



# --- Section 4.1: Clear Draft Board ---
if st.button("🧹 Clear Draft Board"):
    for team in st.session_state.drafted_players.keys():
        st.session_state.drafted_players[team] = []
    st.success("All drafted players cleared.")
    st.rerun()

# --- Section 4.2: Team Selector ---
selected_team = st.selectbox("Select which team is drafting:", list(st.session_state.drafted_players.keys()))

# --- Section 4.3: Top Available Players ---
st.markdown("### 🔥 Top 15 Available Players")

drafted_flat = [p for team in st.session_state.drafted_players.values() for p in team]
available_players = all_players[~all_players["Full Name"].isin(drafted_flat)].head(15)

for i in range(0, len(available_players), 5):
    row_players = available_players.iloc[i:i+5]
    cols = st.columns(5, gap="small")
    for idx, (_, player) in enumerate(row_players.iterrows()):
        with cols[idx]:
            st.image(player["ImageURL"], width=80)
            st.markdown(f"**{player['Full Name']}**")
            st.caption(f"ADP: {player['ADP']} | {player['Pos.']} | {player['Ht.']} | {player['Year']}")
            st.caption(f"Team: {player.get('FullTeamName', 'N/A')}")
            st.caption(f"Pts Avg: {player.get('PTS_mean', 'N/A')}")
            st.caption(f"Fantasy Pts: {player.get('FantasyPts_mean', 'N/A')}")
            
            # Draft Button
            if st.button(f"Draft ➕", key=f"draft_{player['Full Name']}"):
                st.session_state.drafted_players[selected_team].append(player["Full Name"])
                st.rerun()

st.markdown("---")

# --- Section 4.4: Manual Draft Selection ---
st.markdown("### 🔍 Draft a Player Manually")

# Compute undrafted players
drafted_flat = [p for team in st.session_state.drafted_players.values() for p in team]
undrafted_players = all_players[~all_players["Full Name"].isin(drafted_flat)]

manual_pick = st.selectbox(
    "Search for a player to draft:",
    options=[""] + undrafted_players["Full Name"].tolist(),
    index=0
)

team_choice = st.selectbox("Assign to which team:", list(st.session_state.drafted_players.keys()))

if st.button("➕ Add Player to Team"):
    if manual_pick:
        if manual_pick not in drafted_flat:
            st.session_state.drafted_players[team_choice].append(manual_pick)
            st.success(f"{manual_pick} added to {team_choice}!")
            st.rerun()
        else:
            st.warning(f"{manual_pick} has already been drafted.")

# --- Section 4.5: Display Draft Boards ---
st.markdown("## 📋 Draft Boards")

cols = st.columns(len(st.session_state.drafted_players))

for idx, (team_name, roster) in enumerate(st.session_state.drafted_players.items()):
    with cols[idx]:
        st.subheader(team_name)
        if roster:
            for player_name in roster:
                player_row = all_players[all_players["Full Name"] == player_name].iloc[0]
                st.image(player_row["ImageURL"], width=60)
                st.markdown(f"**{player_row['Full Name']}**")
                st.caption(f"{player_row['Pos.']} | {player_row['Ht.']} | {player_row['Year']}")
                st.caption(f"Team: {player_row.get('Team', 'N/A')}")
        else:
            st.markdown("_No players drafted yet_")



# --- Save Drafted Players to CSV ---
if st.button("💾 Save Draft Results to CSV"):
    drafted_data = []
    for team, players in st.session_state["drafted_players"].items():
        for player in players:
            drafted_data.append({"Team": team, "Player": player})

    if drafted_data:
        df_draft = pd.DataFrame(drafted_data)
        df_draft.to_csv("data/draft_results2.csv", index=False)
        st.success("✅ Draft results saved to draft_results.csv")
    else:
        st.warning("No players have been drafted yet.")
