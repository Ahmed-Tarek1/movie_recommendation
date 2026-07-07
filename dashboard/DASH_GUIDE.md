# CineSense Dash Dashboard Guide

## What's New

The new Dash dashboard replaces the basic Streamlit interface with a rich, modern UI featuring:

### 🎨 Design
- **Dark cinema theme** with professional styling
- **Responsive layout** that adapts to screen size
- **Bootstrap icons** and smooth transitions
- **Plotly charts** with interactive tooltips

### 📊 Features

#### User Page
- **Genre Preferences Pie Chart** — see what genres a user watches most
- **Rating Distribution Histogram** — visualize how the user rates movies
- **Watch History Table** — last 20 movies with formatted dates and rating badges
- **Top-N Recommendations** — displayed as cards with:
  - Score bars (visual representation of prediction score)
  - Ranking badges
  - Genre filter dropdown to narrow recommendations
- **Pagination** with page navigation

#### Item Page
- **Movie Profile** — title with genre badges
- **Similar Movies** — horizontal bar chart ranked by cosine similarity score
- **Pagination** for large similarity results

#### Stats Page
- **System Overview** — total users and movies in the database

---

## Installation

Install the new dependencies:
```bash
pip install dash dash-bootstrap-components plotly
```

Or update from requirements.txt:
```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Make sure the backend is running first:
```bash
uvicorn backend.main:app --reload --port 8000
```

Then start the Dash app:
```bash
python dashboard/dash_app.py
```

Visit: **http://localhost:8050**

---

## Environment Variables

The dashboard reads `BACKEND_URL` from environment or defaults to `http://localhost:8000`.

Set it in `.env`:
```bash
BACKEND_URL=http://localhost:8000
```

---

## Architecture

- **Dash + Plotly** — Python-based web framework with native chart support
- **Dash Bootstrap Components** — prebuilt UI components (cards, buttons, nav)
- **Multi-page routing** — `/` (User), `/item` (Item), `/stats` (Stats)
- **Callback-driven** — reactive updates when user interacts with dropdowns/buttons
- **Client-side stores** — `dcc.Store` for pagination state

---

## Customization

### Change Theme
Edit line 18 in `dash_app.py`:
```python
external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP]
```
Replace `DARKLY` with another Bootstrap theme: `SOLAR`, `SLATE`, `CYBORG`, etc.

### Change Port
Edit line 788:
```python
app.run(debug=True, host="0.0.0.0", port=8050)
```

### Add More Charts
Follow the callback pattern:
1. Add a `dcc.Graph` in the layout
2. Create a callback with `@app.callback` to populate it
3. Use Plotly Express (`px`) or Graph Objects (`go`) for the figure

---

## Comparison: Dash vs Streamlit

| Feature | Streamlit | Dash |
|---------|-----------|------|
| Layout control | Limited (columns/expanders) | Full (Bootstrap grid + CSS) |
| Charts | Basic Plotly support | Native Plotly integration |
| Interactivity | Rerun entire script | Callback-based (efficient) |
| Styling | Minimal customization | Full CSS/theme control |
| Production | Single-user friendly | Multi-user ready |
| Learning curve | Very low | Moderate |

---

## Troubleshooting

**Backend unavailable error:**
- Make sure `uvicorn backend.main:app --port 8000` is running
- Check `BACKEND_URL` environment variable

**ModuleNotFoundError:**
- Run `pip install -r requirements.txt`

**Port already in use:**
- Change the port in line 788 or kill the process using port 8050

**Charts not loading:**
- Check browser console for errors
- Verify backend endpoints return data (visit `http://localhost:8000/docs`)

---

## Next Steps

Potential enhancements:
- Add year filter (extract from movie titles)
- Show average rating per movie
- Add a "surprise me" random recommendation button
- Export recommendations to CSV
- Model comparison view (if you train both FM and DeepFM)
- User-to-user similarity explorer
- Real-time prediction confidence intervals

