import streamlit as st
import pandas as pd
from recommender import RecoService
from planner import UserProfile, compose_topk

st.set_page_config(page_title="Hybrid Recommender | Itinerary Planner", layout="wide")
st.title("AI Recommender for Travel")
st.subheader("1. Hybrid Recommendation : Content-Based Filtering + Collaborative Filtering")

# Engine (embedded)
svc = RecoService("data/items.csv", "data/interactions.csv")

# Controls
alpha = st.sidebar.slider("alpha (content weight)", 0.0, 1.0, 0.5, 0.05)
beta  = st.sidebar.slider("beta (collaborative weight)", 0.0, 1.0, 0.5, 0.05)
top_k = st.sidebar.slider("Top-K recommendations", 1, 20, 10, 1)

# UI data
items = svc.catalog.items.to_dict(orient="records")
users = svc.cf.users
item_map = {row["title"]: row["item_id"] for row in items}

# Inputs
col1, col2 = st.columns(2)
with col1:
    sel_user = st.selectbox("User", [""] + users)
with col2:
    seed_title = st.selectbox("Item (Activity)", [""] + list(item_map.keys()))
query_text = st.text_input("Type a content query", "")


def coerce_numeric(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def build_policy_mask(df, fam_only: bool, avoid_night: bool):
    f = (df["family_friendly"] == 1) if fam_only else True
    n = (df["nightlife"] == 0)        if avoid_night else True
    # when one side is True (bool), pandas broadcasts fine
    return f & n

def ensure_candidates(seed_item_id, query_text, user_id, alpha, beta, topk_reco,
                      fam_only: bool, avoid_night: bool) -> list[str]:
    """Return a large, policy-compliant candidate set with sensible fallbacks."""
    # 1) Try recommender first (large pool)
    recos = svc.recommend(user_id=user_id or None,
                          seed_item_id=seed_item_id or None,
                          query_text=query_text or None,
                          alpha=alpha, beta=beta, top_k=topk_reco)
    cand = [r["item_id"] for r in recos]

    # 2) If too few, fall back to all items that pass policy
    df = coerce_numeric(svc.catalog.items, ["price","duration_hr","family_friendly","nightlife"])
    mask = build_policy_mask(df, fam_only, avoid_night)
    fallback_ids = df.loc[mask, "item_id"].tolist()

    # 3) Union, preserve order (reco first)
    seen = set()
    merged = []
    for i in cand + fallback_ids:
        if i not in seen:
            merged.append(i); seen.add(i)
    return merged

def coerce_numeric(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def build_policy_mask(df, fam_only: bool, avoid_night: bool):
    f = (df["family_friendly"] == 1) if fam_only else True
    n = (df["nightlife"] == 0)        if avoid_night else True
    # when one side is True (bool), pandas broadcasts fine
    return f & n

def ensure_candidates(seed_item_id, query_text, user_id, alpha, beta, topk_reco,
                      fam_only: bool, avoid_night: bool) -> list[str]:
    """Return a large, policy-compliant candidate set with sensible fallbacks."""
    # 1) Try recommender first (large pool)
    recos = svc.recommend(user_id=user_id or None,
                          seed_item_id=seed_item_id or None,
                          query_text=query_text or None,
                          alpha=alpha, beta=beta, top_k=topk_reco)
    cand = [r["item_id"] for r in recos]

    # 2) If too few, fall back to all items that pass policy
    df = coerce_numeric(svc.catalog.items, ["price","duration_hr","family_friendly","nightlife"])
    mask = build_policy_mask(df, fam_only, avoid_night)
    fallback_ids = df.loc[mask, "item_id"].tolist()

    # 3) Union, preserve order (reco first)
    seen = set()
    merged = []
    for i in cand + fallback_ids:
        if i not in seen:
            merged.append(i); seen.add(i)
    return merged

def compose_with_backoff(candidates, prefs, k=10):
    """Try to compose; if empty, relax constraints progressively."""
    plans = compose_topk(candidates, svc.catalog.items, prefs, k=k)
    if plans: return plans

    # backoff 1: +2 hours
    p1 = UserProfile(**{**prefs.__dict__, "max_hours": prefs.max_hours})
    plans = compose_topk(candidates, svc.catalog.items, p1, k=k)
    if plans: return plans

    # backoff 2: +20% budget
    p2 = UserProfile(**{**p1.__dict__, "max_budget": prefs.max_budget * 1.2})
    plans = compose_topk(candidates, svc.catalog.items, p2, k=k)
    if plans: return plans

    # backoff 3: min_activities -> max(1, current-1)
    p3 = UserProfile(**{**p2.__dict__, "min_activities": max(1, prefs.min_activities - 1)})
    plans = compose_topk(candidates, svc.catalog.items, p3, k=k)
    return plans



# Recommend (embedded)
if st.button("Recommend"):
    seed_item_id = item_map.get(seed_title) if seed_title else None
    # Build candidates from the recommender (bigger pool)
    recos = svc.recommend(
        user_id=sel_user or None,
        seed_item_id=item_map.get(seed_title) if seed_title else None,
        query_text=query_text or None,
        alpha=alpha, beta=beta, top_k=100   # <- larger pool for composing
    )
    candidates = [r["item_id"] for r in recos]

    # Fallback: if recommender returned nothing (or too few), use policy-filtered catalog
    if len(candidates) < 10:
        df = svc.catalog.items.copy()
        df["family_friendly"] = pd.to_numeric(df["family_friendly"], errors="coerce").fillna(0).astype(int)
        df["nightlife"] = pd.to_numeric(df["nightlife"], errors="coerce").fillna(0).astype(int)
        mask = ( (~st.session_state.get("fam_only", False)) | (df["family_friendly"] == 1) ) & \
            ( (~st.session_state.get("avoid_night", False)) | (df["nightlife"] == 0) )
        candidates = df.loc[mask, "item_id"].tolist()

    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame(recos), use_container_width=True)

st.divider()
st.header("2. Persona Itineraries")

# Composer inputs
colA, colB, colC = st.columns(3)
with colA: max_budget = st.number_input("Max budget", min_value=0.0, value=250.0, step=10.0)
with colB: max_hours  = st.number_input("Max total hours", min_value=1.0, value=8.0, step=1.0)
with colC: min_acts   = st.number_input("Min #activities", min_value=1, value=2, step=1)
colD, colE = st.columns(2)
with colD: fam = st.checkbox("Family-friendly only", value=False)
with colE: avoid_night = st.checkbox("Avoid nightlife", value=True)
prefer = st.text_input("Prefer tags (comma-separated)", "snorkel, reef")

# Compose (embedded)
# ---------- Compose ----------
if st.button("Compose Itineraries"):
    seed_item_id = item_map.get(seed_title) if seed_title else None

    # Use cached candidates if available; otherwise build now with policy
    candidates = st.session_state.get("candidates")
    if not candidates:
        candidates = ensure_candidates(
            seed_item_id=seed_item_id,
            query_text=query_text or None,
            user_id=sel_user or None,
            alpha=alpha, beta=beta,
            topk_reco=200,
            fam_only=fam, avoid_night=avoid_night
        )

    prefs = UserProfile(
        max_budget=max_budget, max_hours=max_hours, min_activities=min_acts,
        family_friendly=fam, avoid_nightlife=avoid_night,
        prefer_tags=tuple(t.strip() for t in (prefer or "").split(",") if t.strip())
    )


    # Try compose, then relax if needed
    plans = compose_with_backoff(candidates, prefs, k=10)

    if not plans:
        df_ = coerce_numeric(svc.catalog.items, ["family_friendly","nightlife"])
        eligible = df_.loc[build_policy_mask(df_, fam, avoid_night)]
        st.warning(f"No feasible itineraries. Eligible items under policy: {len(eligible)}. "
                   f"Try increasing hours/budget or unchecking 'Avoid nightlife'.")
    else:
        for i, plan in enumerate(plans, 1):
                with st.expander(f"Itinerary #{i} — ${plan['total_cost']:,.0f}, "
                                f"{plan['total_hours']:.0f} hrs, score {plan['score']:.3f}"):
                    # insert these two lines ↓ and update the expander title as above
                    dfp = pd.DataFrame(plan["items"])
                    dfp.insert(0, "#", range(1, len(dfp) + 1))
                    st.dataframe(dfp, use_container_width=True)