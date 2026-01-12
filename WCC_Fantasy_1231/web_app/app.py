import streamlit as st

st.set_page_config(page_title="WCC Fantasy Dashboard", page_icon="🏀")

st.title("🏀 WCC Fantasy Basketball")
st.write("Welcome to your fantasy league! Use the sidebar or links below to navigate.")



st.markdown("""
### Quick Links
- [📊 View Weekly Standings](Standings)
- [🧍 Check Your Team](MyTeam)
- [🧾 Browse Free Agents](FreeAgents)
- [🧠 Enter Draft Board](Draft_board)
- [📅 Player Game Logs](Game_Logs)
""")