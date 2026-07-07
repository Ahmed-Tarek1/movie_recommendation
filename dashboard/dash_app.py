"""
Dash-based Movie Recommender Dashboard — CineSense
Run: python dashboard/dash_app.py

Performance design:
- All movie metadata (title + genres) loaded ONCE at startup into a module-level
  dict. Zero per-item HTTP calls anywhere in the dashboard.
- Dash long_callback / flask-caching not required — the startup preload is enough.
- Backend /items now returns genres in the same payload, so the dashboard never
  hits /profile/item in a loop.
"""
import os
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── App init ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="CineSense",
)

# ── Module-level cache — loaded ONCE on startup ────────────────────────────────
# {item_id: {"title": str, "genres": list[str]}}
_ITEM_CACHE: dict[int, dict] = {}
_USER_IDS: list[int] = []
_ALL_GENRES: list[str] = []


def _load_cache():
    """Fetch /users and /items once and populate module-level caches."""
    global _ITEM_CACHE, _USER_IDS, _ALL_GENRES
    try:
        u = requests.get(f"{BACKEND_URL}/users", timeout=10)
        u.raise_for_status()
        _USER_IDS = u.json().get("user_ids", [])
    except Exception:
        _USER_IDS = []

    try:
        r = requests.get(f"{BACKEND_URL}/items", timeout=30)
        r.raise_for_status()
        for it in r.json().get("items", []):
            _ITEM_CACHE[it["item_id"]] = {"title": it["title"], "genres": it.get("genres", [])}
        genre_set: set[str] = set()
        for meta in _ITEM_CACHE.values():
            genre_set.update(meta["genres"])
        _ALL_GENRES = sorted(genre_set)
    except Exception:
        _ITEM_CACHE = {}
        _ALL_GENRES = []


_load_cache()


# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y")


def item_title(item_id: int) -> str:
    return _ITEM_CACHE.get(item_id, {}).get("title", f"Item {item_id}")


def item_genres(item_id: int) -> list[str]:
    return _ITEM_CACHE.get(item_id, {}).get("genres", [])


def empty_fig(height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ── Navbar ─────────────────────────────────────────────────────────────────────

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [html.I(className="bi bi-film me-2", style={"fontSize": "1.5rem"}),
                 html.Span("CineSense", className="navbar-brand mb-0 h1")],
                className="d-flex align-items-center",
            ),
            dbc.Nav(
                [
                    dbc.NavLink("User",  href="/",      active="exact", className="px-3"),
                    dbc.NavLink("Item",  href="/item",  active="exact", className="px-3"),
                    dbc.NavLink("Stats", href="/stats", active="exact", className="px-3"),
                ],
                pills=True,
                className="ms-auto",
            ),
        ],
        fluid=True,
        className="d-flex align-items-center",
    ),
    color="dark",
    dark=True,
    className="mb-4",
    style={"boxShadow": "0 2px 4px rgba(0,0,0,.3)"},
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(id="page-content"),
    ],
    fluid=True,
    style={"backgroundColor": "#1a1a1a", "minHeight": "100vh"},
)


# ── User Page layout ───────────────────────────────────────────────────────────

def user_page():
    if not _USER_IDS:
        return dbc.Alert("Backend unavailable — start it with: uvicorn backend.main:app", color="danger")

    genre_options = [{"label": g, "value": g} for g in _ALL_GENRES]

    return dbc.Container([
        html.H3([html.I(className="bi bi-person-circle me-2"), "User Recommendations"], className="mb-3"),

        # Controls row
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                dbc.Label("Select User", className="fw-bold"),
                dcc.Dropdown(
                    id="user-select",
                    options=[{"label": f"User {uid}", "value": uid} for uid in _USER_IDS],
                    value=_USER_IDS[0],
                    clearable=False,
                    style={"color": "#000"},
                ),
            ], md=6),
            dbc.Col([
                dbc.Label("Items per page", className="fw-bold"),
                dbc.Input(id="user-page-size", type="number", min=1, max=50, value=10),
            ], md=3),
            dbc.Col([
                dbc.Label("Filter by Genre", className="fw-bold"),
                dcc.Dropdown(
                    id="genre-filter",
                    options=genre_options,
                    placeholder="All genres",
                    clearable=True,
                    style={"color": "#000"},
                ),
            ], md=3),
        ], className="g-3")), className="mb-4"),

        # Charts
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Genre Preferences", className="card-title"),
                dcc.Loading(dcc.Graph(id="user-genre-chart", config={"displayModeBar": False})),
            ])), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Rating Distribution", className="card-title"),
                dcc.Loading(dcc.Graph(id="user-rating-chart", config={"displayModeBar": False})),
            ])), md=6),
        ], className="g-3 mb-4"),

        # History
        dbc.Card([
            dbc.CardHeader(html.H5([html.I(className="bi bi-clock-history me-2"), "Watch History"], className="mb-0")),
            dbc.CardBody(dcc.Loading(html.Div(id="user-history-table"))),
        ], className="mb-4"),

        # Recommendations
        dbc.Card([
            dbc.CardHeader(html.H5([html.I(className="bi bi-stars me-2"), "Top Recommendations"], className="mb-0")),
            dbc.CardBody([
                dcc.Loading(html.Div(id="user-recommendations")),
                dbc.Row([
                    dbc.Col(dbc.Button([html.I(className="bi bi-chevron-left me-1"), "Prev"],
                                       id="user-prev-btn", color="secondary", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="user-page-info", className="text-center text-muted small"),
                            className="d-flex align-items-center justify-content-center"),
                    dbc.Col(dbc.Button(["Next", html.I(className="bi bi-chevron-right ms-1")],
                                       id="user-next-btn", color="secondary", size="sm"),
                            width="auto", className="text-end"),
                ], className="mt-3"),
            ]),
        ]),

        dcc.Store(id="user-page-store", data=1),
    ], fluid=True)


# ── Item Page layout ───────────────────────────────────────────────────────────

def item_page():
    if not _ITEM_CACHE:
        return dbc.Alert("Backend unavailable — start it with: uvicorn backend.main:app", color="danger")

    item_options = [
        {"label": f"{meta['title']} (ID: {iid})", "value": iid}
        for iid, meta in _ITEM_CACHE.items()
    ]
    first_id = next(iter(_ITEM_CACHE))

    return dbc.Container([
        html.H3([html.I(className="bi bi-film me-2"), "Item Similarity"], className="mb-3"),

        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([
                dbc.Label("Select Movie", className="fw-bold"),
                dcc.Dropdown(
                    id="item-select",
                    options=item_options,
                    value=first_id,
                    clearable=False,
                    style={"color": "#000"},
                ),
            ], md=9),
            dbc.Col([
                dbc.Label("Items per page", className="fw-bold"),
                dbc.Input(id="item-page-size", type="number", min=1, max=50, value=10),
            ], md=3),
        ], className="g-3")), className="mb-4"),

        # Profile
        dbc.Card([
            dbc.CardHeader(html.H5([html.I(className="bi bi-info-circle me-2"), "Movie Profile"], className="mb-0")),
            dbc.CardBody(html.Div(id="item-profile")),
        ], className="mb-4"),

        # Similar movies
        dbc.Card([
            dbc.CardHeader(html.H5([html.I(className="bi bi-link-45deg me-2"), "Similar Movies"], className="mb-0")),
            dbc.CardBody([
                dcc.Loading(html.Div(id="item-similar")),
                dbc.Row([
                    dbc.Col(dbc.Button([html.I(className="bi bi-chevron-left me-1"), "Prev"],
                                       id="item-prev-btn", color="secondary", size="sm"), width="auto"),
                    dbc.Col(html.Div(id="item-page-info", className="text-center text-muted small"),
                            className="d-flex align-items-center justify-content-center"),
                    dbc.Col(dbc.Button(["Next", html.I(className="bi bi-chevron-right ms-1")],
                                       id="item-next-btn", color="secondary", size="sm"),
                            width="auto", className="text-end"),
                ], className="mt-3"),
            ]),
        ]),

        dcc.Store(id="item-page-store", data=1),
    ], fluid=True)


# ── Stats Page layout ──────────────────────────────────────────────────────────

def stats_page():
    return dbc.Container([
        html.H3([html.I(className="bi bi-bar-chart-line me-2"), "System Statistics"], className="mb-3"),
        dbc.Card(dbc.CardBody(dcc.Loading(html.Div(id="stats-content")))),
    ], fluid=True)


# ── Router callback ────────────────────────────────────────────────────────────

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/item":
        return item_page()
    if pathname == "/stats":
        return stats_page()
    return user_page()


# ── User callbacks ─────────────────────────────────────────────────────────────
# Charts + history: 1 HTTP call total (just /history/user/{id}).
# Genres come from _ITEM_CACHE — no per-item HTTP round-trips.

@app.callback(
    [
        Output("user-genre-chart",   "figure"),
        Output("user-rating-chart",  "figure"),
        Output("user-history-table", "children"),
    ],
    Input("user-select", "value"),
)
def update_user_stats(user_id):
    ef = empty_fig()
    if not user_id:
        return ef, ef, html.P("No data")

    history_data = fetch(f"/history/user/{user_id}")
    if not history_data or "error" in history_data or not history_data.get("history"):
        return ef, ef, html.P("No history found for this user", className="text-muted")

    history = history_data["history"]

    # Genre counts — purely from _ITEM_CACHE (zero extra HTTP calls)
    genre_counts: dict[str, int] = {}
    for h in history:
        for g in item_genres(h["item_id"]):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    if genre_counts:
        genre_fig = px.pie(
            names=list(genre_counts.keys()),
            values=list(genre_counts.values()),
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma,
            hole=0.35,
        )
        genre_fig.update_layout(paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
                                 showlegend=True, height=280,
                                 margin=dict(l=10, r=10, t=10, b=10))
    else:
        genre_fig = ef

    # Rating histogram
    ratings = [h["rating"] for h in history]
    rating_fig = px.histogram(
        x=ratings, nbins=9,
        labels={"x": "Rating", "y": "Count"},
        template="plotly_dark",
        color_discrete_sequence=["#636EFA"],
    )
    rating_fig.update_layout(paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
                              showlegend=False, height=280,
                              margin=dict(l=40, r=10, t=10, b=40))

    # History table (last 20)
    rows = []
    for i, h in enumerate(history[:20]):
        rows.append(html.Tr([
            html.Td(i + 1, className="text-muted"),
            html.Td(h["title"]),
            html.Td(dbc.Badge(f"⭐ {h['rating']}",
                              color="warning" if h["rating"] >= 4 else "secondary")),
            html.Td(fmt_ts(h["timestamp"]), className="text-muted small"),
        ]))

    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("#"), html.Th("Title"), html.Th("Rating"), html.Th("Date")])),
         html.Tbody(rows)],
        bordered=True, hover=True, responsive=True, className="table-dark mb-0",
    )
    return genre_fig, rating_fig, table


# Recommendations — 1 HTTP call, genre filter from _ITEM_CACHE

@app.callback(
    [Output("user-recommendations", "children"),
     Output("user-page-info",       "children")],
    [Input("user-select",    "value"),
     Input("user-page-size", "value"),
     Input("user-page-store","data"),
     Input("genre-filter",   "value")],
)
def update_user_recommendations(user_id, page_size, page, genre_filter):
    if not user_id or not page_size:
        return html.P("Select a user", className="text-muted"), ""

    # Request enough items to fill the page after potential genre filtering.
    # When a genre filter is active we over-fetch so the page looks full.
    fetch_size = int(page_size) * (5 if genre_filter else 1)
    recs_data = fetch(
        f"/recommend/user/{user_id}",
        params={"page": 1, "page_size": min(fetch_size * page, 200)},
    )
    if not recs_data or "error" in recs_data:
        return html.P("Error loading recommendations", className="text-danger"), ""

    all_items = recs_data.get("items", [])

    # Genre filter — purely from cache, no HTTP calls
    if genre_filter:
        all_items = [it for it in all_items if genre_filter in item_genres(it["item_id"])]

    total = len(all_items)
    start = (page - 1) * int(page_size)
    items = all_items[start: start + int(page_size)]

    if not items:
        return html.P("No results for this page / filter", className="text-muted"), ""

    cards = []
    for i, rec in enumerate(items):
        rank = start + i + 1
        score_pct = min(float(rec["score"]) * 100, 100)
        cards.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div([
                    dbc.Badge(f"#{rank}", color="primary", className="me-2"),
                    html.Strong(rec["title"], style={"fontSize": "0.9rem"}),
                ], className="mb-2"),
                html.Div([
                    html.Small(f"Score: {rec['score']:.4f}", className="text-muted me-3"),
                    html.Small(
                        " · ".join(item_genres(rec["item_id"])[:3]) or "—",
                        className="text-muted",
                    ),
                ]),
                html.Div([
                    html.Div(style={
                        "width": f"{score_pct:.1f}%",
                        "height": "5px",
                        "backgroundColor": "#00d9ff",
                        "borderRadius": "3px",
                    })
                ], style={"backgroundColor": "#333", "borderRadius": "3px", "marginTop": "8px"}),
            ]),
            style={"backgroundColor": "#2b2b2b", "borderColor": "#444"},
            className="h-100"),
            md=6, lg=4, className="mb-3",
        ))

    page_info = f"Page {page} · {total} results"
    return dbc.Row(cards, className="g-3"), page_info


@app.callback(
    Output("user-page-store", "data"),
    [Input("user-prev-btn", "n_clicks"), Input("user-next-btn", "n_clicks")],
    State("user-page-store", "data"),
    prevent_initial_call=True,
)
def user_paginate(prev_n, next_n, page):
    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if btn == "user-prev-btn" and page > 1:
        return page - 1
    if btn == "user-next-btn":
        return page + 1
    return page


# ── Item callbacks ─────────────────────────────────────────────────────────────

@app.callback(Output("item-profile", "children"), Input("item-select", "value"))
def update_item_profile(item_id):
    if not item_id:
        return html.P("Select a movie", className="text-muted")
    meta = _ITEM_CACHE.get(item_id)
    if not meta:
        return html.P("Not found", className="text-danger")

    badges = [dbc.Badge(g, color="info", className="me-1 mb-1") for g in meta["genres"]]
    return html.Div([
        html.H5(meta["title"], className="mb-2"),
        html.Div([html.Strong("Genres: "), html.Span(badges or html.Small("N/A", className="text-muted"))]),
    ])


@app.callback(
    [Output("item-similar",   "children"),
     Output("item-page-info", "children")],
    [Input("item-select",    "value"),
     Input("item-page-size", "value"),
     Input("item-page-store","data")],
)
def update_item_similar(item_id, page_size, page):
    if not item_id or not page_size:
        return html.P("Select a movie", className="text-muted"), ""

    data = fetch(f"/similar/item/{item_id}",
                 params={"page": page, "page_size": page_size})
    if not data or "error" in data:
        return html.P("Error loading similar items", className="text-danger"), ""

    items = data.get("items", [])
    if not items:
        return html.P("No similar items found", className="text-muted"), ""

    # Horizontal bar chart
    titles = [
        (it["title"][:45] + "…") if len(it["title"]) > 45 else it["title"]
        for it in items
    ]
    sims = [it["similarity"] for it in items]

    fig = go.Figure(go.Bar(
        x=sims, y=titles,
        orientation="h",
        marker=dict(color=sims, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sim", thickness=10)),
        text=[f"{s:.4f}" for s in sims],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        height=max(300, len(items) * 38),
        margin=dict(l=10, r=80, t=10, b=40),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Cosine Similarity"),
    )

    page_info = f"Page {page} · {data.get('total_available', 0)} total"
    return dcc.Graph(figure=fig, config={"displayModeBar": False}), page_info


@app.callback(
    Output("item-page-store", "data"),
    [Input("item-prev-btn", "n_clicks"), Input("item-next-btn", "n_clicks")],
    State("item-page-store", "data"),
    prevent_initial_call=True,
)
def item_paginate(prev_n, next_n, page):
    btn = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if btn == "item-prev-btn" and page > 1:
        return page - 1
    if btn == "item-next-btn":
        return page + 1
    return page


# ── Stats callback ─────────────────────────────────────────────────────────────

@app.callback(Output("stats-content", "children"), Input("url", "pathname"))
def update_stats(pathname):
    if pathname != "/stats":
        return html.Div()

    n_users = len(_USER_IDS)
    n_items = len(_ITEM_CACHE)

    # Genre distribution across entire catalog
    catalog_genres: dict[str, int] = {}
    for meta in _ITEM_CACHE.values():
        for g in meta["genres"]:
            catalog_genres[g] = catalog_genres.get(g, 0) + 1

    genre_fig = px.bar(
        x=list(catalog_genres.keys()),
        y=list(catalog_genres.values()),
        labels={"x": "Genre", "y": "# Movies"},
        template="plotly_dark",
        color=list(catalog_genres.values()),
        color_continuous_scale="Viridis",
    )
    genre_fig.update_layout(paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
                             showlegend=False, coloraxis_showscale=False,
                             height=320, margin=dict(l=40, r=10, t=10, b=80),
                             xaxis_tickangle=-40)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-people-fill", style={"fontSize": "3rem", "color": "#00d9ff"}),
                html.H2(f"{n_users:,}", className="mt-2 mb-0"),
                html.P("Total Users", className="text-muted mb-0"),
            ], className="text-center"), style={"backgroundColor": "#2b2b2b", "borderColor": "#444"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-film", style={"fontSize": "3rem", "color": "#00d9ff"}),
                html.H2(f"{n_items:,}", className="mt-2 mb-0"),
                html.P("Total Movies", className="text-muted mb-0"),
            ], className="text-center"), style={"backgroundColor": "#2b2b2b", "borderColor": "#444"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.I(className="bi bi-tags-fill", style={"fontSize": "3rem", "color": "#00d9ff"}),
                html.H2(f"{len(catalog_genres):,}", className="mt-2 mb-0"),
                html.P("Unique Genres", className="text-muted mb-0"),
            ], className="text-center"), style={"backgroundColor": "#2b2b2b", "borderColor": "#444"}), md=4),
        ], className="g-3 mb-4"),
        dbc.Card(dbc.CardBody([
            html.H5("Movies per Genre", className="card-title"),
            dcc.Graph(figure=genre_fig, config={"displayModeBar": False}),
        ])),
    ])


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
