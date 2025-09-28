# AI Recommender for Travel

A lightweight demo that blends **content-based (TF-IDF)** and **collaborative (item–item)** signals to recommend activities, then **composes feasible itineraries** under budget/time & policy constraints.

---

## Quickstart

### Requirements
- Python 3.10 or above

```bash
# 1) create a new conda environment
conda create -n hybrid_reco python=3.10 -y

# 2) activate the environment
conda activate hybrid_reco

# 3) install dependencies
pip install -r requirements.txt

# 4) run the app
streamlit run streamlit_app.py
````

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## Project structure

```
.
├─ data/
│  ├─ items.csv            # catalog: item_id, title, tags, category, price, duration_hr, family_friendly, nightlife
│  └─ interactions.csv     # implicit feedback: user_id, item_id, weight
├─ recommender.py          # TF-IDF (CB) + item–item CF + hybrid ranker (MMR) + normalization
├─ planner.py              # deterministic composer (budget/hours/policy/preferences)
└─ streamlit_app.py        # UI (embedded engine only)
```

---

## How it works (short)

* **Content-based (CB):** TF-IDF on `title + tags`, cosine similarity to seed item or query.
* **Collaborative (CF):** build item×user matrix from `interactions.csv`, compute item–item cosine, score candidates by similarity to user’s history.
* **Normalization:** per request, CB & CF scores are min–max scaled to **[0,1]**.
* **Hybrid ranker:** `alpha * cb + beta * cf`, then **MMR** for diversity and a **guards hook** (place for policy filters).
* **Itinerary composer:** beam-style expansion picks a small set of items that satisfies:

  * `sum(price) ≤ max_budget`
  * `sum(duration_hr) ≤ max_hours`
  * optional flags: `family_friendly`, `avoid_nightlife`
  * soft preferences via `prefer_tags`, plus a small diversity penalty

---

## Using the app

### 1) Hybrid Recommendations

* **User (optional):** choose a user to enable CF.
* **Seed item (optional)** or **Query:** drives CB.
* **Sliders:** `alpha` (content weight), `beta` (collab weight), `Top-K`.
* Click **Recommend** to see a table with `item_id`, `title`, `tags`, `cb_score`, `cf_score`.

### 2) Persona Itineraries

* Set **Max budget**, **Max hours**, **Min #activities**.
* Toggle **Family-friendly only**, **Avoid nightlife**.
* **Prefer tags:** e.g., `snorkel, reef, family`.
* Click **Compose Itineraries** to get up to **Top-10** itineraries.
  Each result shows total cost/hours and a numbered list of items.

> Tip: Selecting a **seed item** (e.g., *Island Snorkel Tour*) before composing often yields better plans.

---

## Data format

**`data/items.csv`**

```
item_id,title,tags,category,price,duration_hr,family_friendly,nightlife
A,Island Snorkel Tour,"snorkel, reef, boat, family",activity,120,4,1,0
...
```

**`data/interactions.csv`**

```
user_id,item_id,weight
u1,A,1
u1,B,1
u2,G,2
...
```

* `weight` = strength of implicit feedback (e.g., clicks=1, wishlist=2, booking=5).

---

## Troubleshooting

* **“No feasible itineraries”**

  * Start with lenient constraints: `budget ≥ 250`, `hours ≥ 8`, `min activities = 2`.
  * Pick a friendly **seed item** and click **Recommend** first.
  * Ensure your `tags` actually include your “Prefer tags” words.

* **CF shows zeros**

  * You didn’t pick a **user**, or that user has no interactions.

* **Python errors about `str | None`**

  * Use Python **3.10+**, or replace `str | None` with `Optional[str]`.

---

## Future Works

* Add **must-cover categories** (e.g., include at least one `food` item).
* Add **time windows/overlap checks** with real scheduling.
* Swap TF-IDF for **bi-encoder embeddings** and ANN search (FAISS/Annoy).
* Add **metrics/logging** (latency, hit rate, selection reasons).


