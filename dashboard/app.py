"""
Streamlit dashboard — User Page + Item Page, calling the FastAPI backend.
Run: streamlit run dashboard/app.py
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RS Course Project", layout="wide")

page = st.sidebar.radio("Navigate", ["User Page", "Item Page"])


def fetch(endpoint: str, params: dict | None = None):
    try:
        resp = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Backend request failed: {e}")
        return None


if "user_page_num" not in st.session_state:
    st.session_state.user_page_num = 1
if "item_page_num" not in st.session_state:
    st.session_state.item_page_num = 1


# ---------------- USER PAGE ----------------
if page == "User Page":
    st.title("User Recommendations")

    col1, col2 = st.columns([1, 1])
    with col1:
        user_id = st.number_input("Select User ID", min_value=1, step=1, value=1)
    with col2:
        n = st.number_input("Items per page (N)", min_value=1, max_value=50, value=10)

    st.divider()

    # User history
    history = fetch(f"/history/user/{user_id}")  # TODO: add this endpoint to backend
    if history:
        st.subheader("User History")
        st.dataframe(history.get("history", []), use_container_width=True)

    st.divider()
    st.subheader("Top-N Recommendations")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅ Previous", key="user_prev") and st.session_state.user_page_num > 1:
            st.session_state.user_page_num -= 1
    with nav_col3:
        if st.button("Next ➡", key="user_next"):
            st.session_state.user_page_num += 1
    with nav_col2:
        jump_page = st.number_input(
            "Jump to page", min_value=1, value=st.session_state.user_page_num, key="user_jump"
        )
        if jump_page != st.session_state.user_page_num:
            st.session_state.user_page_num = jump_page

    recs = fetch(
        f"/recommend/user/{user_id}",
        params={"page": st.session_state.user_page_num, "page_size": n},
    )
    if recs:
        st.caption(f"Page {recs['page']} — {recs['total_available']} items available")
        st.table(recs["items"])


# ---------------- ITEM PAGE ----------------
elif page == "Item Page":
    st.title("Item Similarity")

    item_id = st.number_input("Select Item ID", min_value=1, step=1, value=1)
    n = st.number_input("Similar items per page (N)", min_value=1, max_value=50, value=10)

    st.divider()

    profile = fetch(f"/profile/item/{item_id}")  # TODO: add this endpoint to backend
    if profile:
        st.subheader("Item Profile")
        st.write(f"**{profile.get('title', '')}**")
        st.write(", ".join(profile.get("genres", [])))

    st.divider()
    st.subheader("Top-N Similar Items")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("⬅ Previous", key="item_prev") and st.session_state.item_page_num > 1:
            st.session_state.item_page_num -= 1
    with nav_col3:
        if st.button("Next ➡", key="item_next"):
            st.session_state.item_page_num += 1
    with nav_col2:
        jump_page = st.number_input(
            "Jump to page", min_value=1, value=st.session_state.item_page_num, key="item_jump"
        )
        if jump_page != st.session_state.item_page_num:
            st.session_state.item_page_num = jump_page

    sims = fetch(
        f"/similar/item/{item_id}",
        params={"page": st.session_state.item_page_num, "page_size": n},
    )
    if sims:
        st.caption(f"Page {sims['page']} — {sims['total_available']} items available")
        st.table(sims["items"])
