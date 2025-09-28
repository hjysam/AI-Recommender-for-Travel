
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def _minmax01(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    vmin = min(vals)
    vmax = max(vals)
    if vmax <= vmin:
        # all equal → map to 0.0
        return {k: 0.0 for k in d}
    rng = vmax - vmin
    return {k: max(0.0, (v - vmin) / rng) for k, v in d.items()}



@dataclass
class Catalog:
    items: pd.DataFrame
    interactions: pd.DataFrame
    @classmethod
    def from_csv(cls, items_csv: str, interactions_csv: str) -> "Catalog":
        return cls(pd.read_csv(items_csv), pd.read_csv(interactions_csv))

class ContentIndex:
    def __init__(self, catalog: Catalog):
        txt = (catalog.items["title"].fillna("") + " " + catalog.items["tags"].fillna("")).tolist()
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self.M = normalize(self.vectorizer.fit_transform(txt))
        self.id2idx = {iid: i for i, iid in enumerate(catalog.items["item_id"])}
        self.items = catalog.items
    def _vec_item(self, item_id): return self.M[self.id2idx[item_id]]
    def _vec_text(self, text):   return normalize(self.vectorizer.transform([text]))
    def similar_item(self, item_id, k=50):
        q = self._vec_item(item_id); s = (self.M @ q.T).toarray().ravel()
        s[self.id2idx[item_id]] = 0.0
        idx = np.argsort(-s)[:k]; ids = self.items.iloc[idx]["item_id"].tolist()
        return list(zip(ids, s[idx].tolist()))
    def similar_text(self, text, k=50):
        q = self._vec_text(text); s = (self.M @ q.T).toarray().ravel()
        idx = np.argsort(-s)[:k]; ids = self.items.iloc[idx]["item_id"].tolist()
        return list(zip(ids, s[idx].tolist()))

class CollabIndex:
    def __init__(self, catalog: Catalog):
        items = catalog.items["item_id"].tolist()
        users = catalog.interactions["user_id"].unique().tolist()
        self.item_index = {iid:i for i,iid in enumerate(items)}
        self.user_index = {u:i for i,u in enumerate(users)}
        I,U = len(items), len(users)
        MU = np.zeros((I,U), float)
        for _,r in catalog.interactions.iterrows():
            i = self.item_index.get(r["item_id"]); u = self.user_index.get(r["user_id"])
            if i is not None and u is not None:
                MU[i,u] += float(r.get("weight",1.0))
        MU = MU / (np.linalg.norm(MU, axis=0, keepdims=True) + 1e-12)
        self.item_user = MU
        sims = MU @ MU.T
        norms = np.linalg.norm(MU, axis=1, keepdims=True) + 1e-12
        self.item_sim = sims / (norms @ norms.T); np.fill_diagonal(self.item_sim, 0.0)
        self.items = items; self.users = users
    def for_user(self, user_id: str, k=50):
        if user_id not in self.user_index: return []
        u = self.user_index[user_id]; pref = self.item_user[:,u]
        score = self.item_sim @ pref
        idx = np.argsort(-score)[:k]
        return [(self.items[i], float(score[i])) for i in idx]

class Guard:
    def __init__(self, blocked: Optional[Set[str]]=None): self.blocked=set(blocked or [])
    def filter(self, ids): return [i for i in ids if i not in self.blocked]

def _jacc(a:Set[str], b:Set[str]): 
    return 0.0 if not a and not b else len(a & b)/max(1,len(a|b))

class HybridRanker:
    def __init__(self, catalog: Catalog):
        self.tags = {r.item_id:set(str(r.tags).replace(","," ").split()) for _,r in catalog.items.iterrows()}
    def rank(self, candidates, cb, cf, guards, alpha=0.6, beta=0.4, k=10, mmr_lambda=0.7):
        pool = guards.filter(candidates)
        base = {i: alpha*cb.get(i,0)+beta*cf.get(i,0) for i in pool}
        picked = []
        while base and len(picked)<k:
            if not picked:
                x = max(base, key=lambda i: base[i])
            else:
                def red(i): 
                    return max(_jacc(self.tags.get(i,set()), self.tags.get(j,set())) for j in picked)
                x = max(base, key=lambda i: mmr_lambda*base[i] - (1-mmr_lambda)*red(i))
            picked.append(x); base.pop(x, None)
        return picked

class RecoService:
    def __init__(self, items_csv: str, interactions_csv: str):
        self.catalog = Catalog.from_csv(items_csv, interactions_csv)
        self.cb = ContentIndex(self.catalog)
        self.cf = CollabIndex(self.catalog)
        self.rank = HybridRanker(self.catalog)
        self._items_by_id = self.catalog.items.set_index("item_id")
    def recommend(self, user_id=None, seed_item_id=None, query_text=None,
                alpha=0.6, beta=0.4, top_k=10, blocked=None):
        # 1) Get content-based pairs depending on seed/query/popularity
        if seed_item_id:
            cb_pairs = self.cb.similar_item(seed_item_id, k=100)
        elif query_text:
            cb_pairs = self.cb.similar_text(query_text, k=100)
        else:
            pop = (self.catalog.interactions
                .groupby("item_id")["weight"].sum()
                .sort_values(ascending=False))
            cb_pairs = [(i, float(s)) for i, s in pop.head(100).items()]

        # 2) Build dicts/lists from pairs (UNCONDITIONAL)
        cb_scores = dict(cb_pairs)
        cb_ids    = [i for i, _ in cb_pairs]

        cf_pairs  = self.cf.for_user(user_id, k=100) if user_id else []
        cf_scores = dict(cf_pairs)
        cf_ids    = [i for i, _ in cf_pairs]

        # 3) Normalize both signals to [0,1] per request
        cb_scores = _minmax01(cb_scores)
        cf_scores = _minmax01(cf_scores)

        # 4) Rank candidates
        candidates = list(dict.fromkeys(cb_ids + cf_ids))
        guards = Guard(blocked)
        ranked = self.rank.rank(candidates, cb_scores, cf_scores, guards,
                                alpha, beta, k=top_k)

        # 5) Build output rows
        out = []
        for iid in ranked:
            row = self._items_by_id.loc[iid]
            out.append({
                "item_id": iid,
                "title": row["title"],
                "tags": row["tags"],
                "cb_score": round(cb_scores.get(iid, 0.0), 4),
                "cf_score": round(cf_scores.get(iid, 0.0), 4),
            })
        return out
