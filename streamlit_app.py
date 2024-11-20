import streamlit as st

# --- PAGE SETUP ---
about_page = st.Page(
    "views/about_me.py",
    title="About Me",
    icon=":material/account_circle:",
    default=True,
)

project_1_page = st.Page(
    "views/qaqc.py",
    title="QAQC",
    icon=":material/view_kanban:",
)

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
#pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "Duplicate": [project_1_page]
        #"Chat": [project_2_page],
        #"QaQc": [project_3_page],
    }
)

# --- RUN NAVIGATION ---
pg.run()