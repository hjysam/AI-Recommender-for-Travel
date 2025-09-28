
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

@dataclass
class UserProfile:
    max_budget: float = 300.0
    max_hours: float = 12.0
    min_activities: int = 2
    family_friendly: bool = False
    avoid_nightlife: bool = False
    prefer_tags: tuple[str, ...] = tuple()

def score_item(row: pd.Series, prefer_tags: set[str]) -> float:
    price_term = 1.0 / (1.0 + float(row["price"]))
    tags = set(str(row["tags"]).replace(",", " ").split())
    tag_hits = len(prefer_tags & tags)
    return 0.7*price_term + 0.3*(tag_hits/3.0)

def feasible(plan: list[dict], p: UserProfile) -> bool:
    cost = sum(x["price"] for x in plan); hrs = sum(x["duration_hr"] for x in plan)
    if cost > p.max_budget or hrs > p.max_hours: return False
    if p.family_friendly and any(x.get("family_friendly",0)==0 for x in plan): return False
    if p.avoid_nightlife and any(x.get("nightlife",0)==1 for x in plan): return False
    return True

def compose_topk(candidates: list[str], catalog: pd.DataFrame, profile: UserProfile, k:int=10) -> list[dict]:
    df = catalog.set_index("item_id").loc[[c for c in candidates if c in set(catalog["item_id"])]]         .reset_index().copy()
    prefer = set(profile.prefer_tags or [])
    df["base_score"] = df.apply(lambda r: score_item(r, prefer), axis=1)
    beams: list[tuple[list[dict], float]] = []
    # seed beams
    for _, r in df.sort_values("base_score", ascending=False).head(20).iterrows():
        item = r.to_dict(); plan = [item]
        if feasible(plan, profile):
            beams.append((plan, float(r["base_score"])))
    # expand
    def red_penalty(tags_a: set[str], tags_b: set[str]) -> float:
        if not tags_a or not tags_b: return 0.0
        j = len(tags_a & tags_b) / max(1, len(tags_a | tags_b))
        return 0.25 * j
    for _ in range(3):
        nxt=[]
        for plan, s in beams:
            used = {x["item_id"] for x in plan}
            plan_tags=set()
            for x in plan: plan_tags |= set(str(x["tags"]).replace(","," ").split())
            for _, r in df.iterrows():
                if r["item_id"] in used: continue
                cand = r.to_dict()
                new = plan + [cand]
                if not feasible(new, profile): continue
                sc = s + float(r["base_score"]) - red_penalty(plan_tags, set(str(cand["tags"]).replace(","," ").split()))
                nxt.append((new, sc))
        nxt.sort(key=lambda x:-x[1])
        beams = nxt[:30]
        if not beams: break
    beams = [b for b in beams if len(b[0])>=profile.min_activities]
    seen=set(); uniq=[]
    for plan, sc in beams:
        key = tuple(sorted(x["item_id"] for x in plan))
        if key in seen: continue
        seen.add(key); uniq.append((plan,sc))
    uniq.sort(key=lambda x:-x[1])
    out=[]
    for plan, sc in uniq[:k]:
        out.append({
            "items":[{"item_id":x["item_id"],"title":x["title"],"price":float(x["price"]), "duration_hr":float(x["duration_hr"])} for x in plan],
            "total_cost": float(sum(x["price"] for x in plan)),
            "total_hours": float(sum(x["duration_hr"] for x in plan)),
            "score": float(sc)
        })
    return out
