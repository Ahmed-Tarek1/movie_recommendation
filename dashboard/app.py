"""
Streamlit dashboard — User Page + Item Page.
Run: streamlit run dashboard/app.py
"""
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="CineSense", page_icon="🎬", layout="wide")
st.title("🎬 CineSense — Movie Recommender")

page = st.sidebar.radio("Navigate", ["👤 User Page", "🎥 Item Page", "📊 Model Stats"])


# ── helpers ────────────────────────────────────────────────────────────────────

def fetch(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


@st.cache_data(ttl=300)
def get_user_ids() -> list[int]:
    data = fetch("/users")
    return data["user_ids"] if data else []


@st.cache_data(ttl=300)
def get_items() -> list[dict]:
    data = fetch("/items")
    return data["items"] if data else []


@st.cache_data(ttl=600)
def get_stats():
    users = fetch("/users")
    items = fetch("/items")
    return {
        "users" : len(users["user_ids"]) if users else 0,
        "movies": len(items["items"])    if items else 0,
    }


# ── SUMMARY STATS PANEL (visible on every page) ────────────────────────────────
stats = get_stats()
m1, m2, m3 = st.columns(3)
m1.metric("Total Users",   stats["users"])
m2.metric("Total Movies",  stats["movies"])
m3.metric("Total Ratings", "100,836")

st.divider()

# session state for pagination
for key in ("user_page", "item_page"):
    if key not in st.session_state:
        st.session_state[key] = 1


# ── USER PAGE ──────────────────────────────────────────────────────────────────

if page == "👤 User Page":
    st.header("User Recommendations")

    user_ids = get_user_ids()
    if not user_ids:
        st.warning("Could not load user list — is the backend running?")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.selectbox("Select User", user_ids)
    with col2:
        n = st.number_input("Items per page (N)", min_value=1, max_value=50, value=10)

    genre_filter = st.multiselect(
        "Filter recommendations by genre",
        options=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
        default=[]
    )

    # movie_id -> genres lookup, built once from the cached /items list
    genre_lookup = {it["item_id"]: it.get("genres", []) for it in get_items()}

    st.divider()

    # User history
    with st.expander("📋 User Watch History", expanded=True):
        history = fetch(f"/history/user/{user_id}")
        if history and history.get("history"):
            hist_data = [
                {"Title": h["title"], "Rating": h["rating"],
                 "Timestamp": h["timestamp"], "Movie ID": h["item_id"]}
                for h in history["history"]
            ]
            st.dataframe(hist_data, width="stretch", hide_index=True)
        else:
            st.info("No history found for this user.")

    st.divider()
    st.subheader("🎯 Top-N Recommendations")

    # pagination
    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
    with p_col1:
        if st.button("⬅ Prev") and st.session_state.user_page > 1:
            st.session_state.user_page -= 1
    with p_col3:
        if st.button("Next ➡"):
            st.session_state.user_page += 1
    with p_col2:
        jump = st.number_input("Page", min_value=1, value=st.session_state.user_page, key="ujump")
        if jump != st.session_state.user_page:
            st.session_state.user_page = jump

    recs = fetch(
        f"/recommend/user/{user_id}",
        params={"page": st.session_state.user_page, "page_size": n},
    )
    if recs:
        st.caption(f"Page {recs['page']} · {recs['total_available']} items available")

        rec_items = recs["items"]
        if genre_filter:
            rec_items = [
                r for r in rec_items
                if set(genre_lookup.get(r["item_id"], [])) & set(genre_filter)
            ]
            if not rec_items:
                st.info("No recommendations on this page match the selected genres. Try another page.")

        rec_data = [
            {"Rank": i + 1 + (recs['page'] - 1) * n,
             "Title": r["title"], "Score": r["score"], "Movie ID": r["item_id"]}
            for i, r in enumerate(rec_items)
        ]
        st.dataframe(rec_data, width="stretch", hide_index=True)


# ── ITEM PAGE ──────────────────────────────────────────────────────────────────

elif page == "🎥 Item Page":
    st.header("Item Similarity")

    items = get_items()
    if not items:
        st.warning("Could not load items list — is the backend running?")
        st.stop()

    item_options = {f"{it['title']} (ID: {it['item_id']})": it['item_id'] for it in items}
    selected_label = st.selectbox("Select Movie", list(item_options.keys()))
    item_id = item_options[selected_label]
    n = st.number_input("Similar items per page (N)", min_value=1, max_value=50, value=10)

    st.divider()

    # Item profile
    profile = fetch(f"/profile/item/{item_id}")
    if profile:
        st.subheader(f"🎬 {profile['title']}")
        st.write("**Genres:**", " · ".join(profile.get("genres", [])) or "N/A")

    st.divider()
    st.subheader("🔗 Similar Movies")

    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
    with p_col1:
        if st.button("⬅ Prev") and st.session_state.item_page > 1:
            st.session_state.item_page -= 1
    with p_col3:
        if st.button("Next ➡"):
            st.session_state.item_page += 1
    with p_col2:
        jump = st.number_input("Page", min_value=1, value=st.session_state.item_page, key="ijump")
        if jump != st.session_state.item_page:
            st.session_state.item_page = jump

    sims = fetch(
        f"/similar/item/{item_id}",
        params={"page": st.session_state.item_page, "page_size": n},
    )
    if sims:
        st.caption(f"Page {sims['page']} · {sims['total_available']} items available")
        sim_data = [
            {"Rank": i + 1 + (sims['page'] - 1) * n,
             "Title": s["title"], "Similarity": s["similarity"], "Movie ID": s["item_id"]}
            for i, s in enumerate(sims["items"])
        ]
        st.dataframe(sim_data, width="stretch", hide_index=True)


# ── MODEL STATS PAGE ────────────────────────────────────────────────────────────

elif page == "📊 Model Stats":
    st.header("Model Comparison — FM vs DeepFM")
    import pandas as pd
    from pathlib import Path

    try:
        comparison_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "comparison.csv"
        )

        comparison = pd.read_csv(comparison_path)

        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption("Winner: DeepFM (Best Test RMSE: 1.0063)")
        
    except FileNotFoundError:
        st.warning("Could not find `../models/comparison.csv` — check the file path relative to where you run `streamlit run`.")