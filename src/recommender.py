"""
Netflix Content Intelligence Dashboard
=======================================
Module 3 — Content-Based Recommendation System
TF-IDF + Cosine Similarity
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "netflix_cleaned.csv")


class NetflixRecommender:
    """Content-based filtering recommender using TF-IDF on genre + description."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._title_index = {}
        self._similarity_matrix = None
        self._build_model()

    def _build_feature_string(self, row: pd.Series) -> str:
        # Repeat genres 3x to give them stronger weight
        genres  = " ".join(row["listed_in"].replace(",", " ").split() * 3)
        desc    = str(row.get("description", ""))
        dtype   = str(row.get("type", ""))
        country = str(row.get("country", ""))
        return f"{genres} {dtype} {country} {desc}"

    def _build_model(self) -> None:
        print("  Building TF-IDF feature matrix ...")
        self.df["_feature_str"] = self.df.apply(self._build_feature_string, axis=1)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8_000,
            min_df=2,
        )
        tfidf_matrix = vectorizer.fit_transform(self.df["_feature_str"])
        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

        self._similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("  Cosine similarity matrix computed.")

        self._title_index = {
            title.lower(): idx
            for idx, title in enumerate(self.df["title"])
        }

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        key = title.lower()
        if key not in self._title_index:
            candidates = [t for t in self._title_index if key in t]
            if not candidates:
                raise ValueError(f"Title '{title}' not found.")
            key = candidates[0]
            print(f"  Using closest match: '{key}'")

        idx = self._title_index[key]
        sim_scores = list(enumerate(self._similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

        indices = [i for i, _ in sim_scores]
        scores  = [round(s, 4) for _, s in sim_scores]

        recs = self.df.iloc[indices][
            ["title", "type", "primary_genre", "listed_in",
             "rating", "release_year", "country", "duration"]
        ].copy()
        recs["similarity_score"] = scores
        return recs.reset_index(drop=True)

    def recommend_by_genre(self, genre: str, content_type: str = None,
                            top_n: int = 10) -> pd.DataFrame:
        mask = self.df["listed_in"].str.contains(genre, case=False, na=False)
        if content_type:
            mask &= self.df["type"].str.lower() == content_type.lower()
        return (
            self.df[mask][["title", "type", "primary_genre",
                            "listed_in", "rating", "release_year", "country"]]
            .drop_duplicates(subset=["title"])
            .head(top_n)
            .reset_index(drop=True)
        )

    def similar_to_multiple(self, titles: list, top_n: int = 10) -> pd.DataFrame:
        combined = np.zeros(len(self.df))
        for title in titles:
            key = title.lower()
            if key not in self._title_index:
                print(f"  Skipping unknown title: '{title}'")
                continue
            combined += self._similarity_matrix[self._title_index[key]]

        seed_indices = {
            self._title_index[t.lower()]
            for t in titles if t.lower() in self._title_index
        }
        ranked = sorted(enumerate(combined), key=lambda x: x[1], reverse=True)
        ranked = [(i, s) for i, s in ranked if i not in seed_indices][:top_n]

        indices = [i for i, _ in ranked]
        scores  = [round(s / max(len(titles), 1), 4) for _, s in ranked]

        recs = self.df.iloc[indices][
            ["title", "type", "primary_genre", "listed_in",
             "rating", "release_year", "country"]
        ].copy()
        recs["blended_score"] = scores
        return recs.reset_index(drop=True)


def run_demo(clean_path: str = CLEAN_PATH):
    print("\n" + "=" * 55)
    print("  Netflix Recommender Demo")
    print("=" * 55)

    df  = pd.read_csv(clean_path, low_memory=False)
    rec = NetflixRecommender(df)

    sample_title = df["title"].iloc[0]
    print(f"\n  Recommendations similar to: '{sample_title}'")
    recs = rec.recommend(sample_title, top_n=5)
    print(recs[["title", "type", "primary_genre", "similarity_score"]].to_string(index=False))

    print(f"\n  Top Drama Movies:")
    drama = rec.recommend_by_genre("Drama", content_type="Movie", top_n=5)
    print(drama[["title", "primary_genre", "rating", "release_year"]].to_string(index=False))

    print("\n  Recommender demo complete!\n")
    return rec


if __name__ == "__main__":
    run_demo()