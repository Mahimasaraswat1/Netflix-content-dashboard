"""
Netflix Content Intelligence Dashboard
=======================================
Module 2 — Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "netflix_cleaned.csv")


def load_clean_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"  Cleaned data loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def content_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df.groupby("type").size().reset_index(name="count")
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    return dist.sort_values("count", ascending=False)


def year_wise_additions(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby(["added_year", "type"]).size()
          .reset_index(name="count")
          .dropna(subset=["added_year"])
    )
    yearly["added_year"] = yearly["added_year"].astype(int)
    return yearly.sort_values("added_year")


def release_year_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["release_year", "type"]).size()
          .reset_index(name="count")
          .sort_values("release_year")
    )


def genre_analysis(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    genre_series = df["listed_in"].str.split(",").explode().str.strip()
    genre_counts = genre_series.value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]
    genre_counts["percentage"] = (
        genre_counts["count"] / genre_counts["count"].sum() * 100
    ).round(2)
    return genre_counts.head(top_n)


def genre_by_type(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for content_type, group in df.groupby("type"):
        genres = group["listed_in"].str.split(",").explode().str.strip()
        top = genres.value_counts().head(top_n).reset_index()
        top.columns = ["genre", "count"]
        top["type"] = content_type
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def country_analysis(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    country_series = (
        df[df["country"] != "Unknown"]["country"]
        .str.split(",").explode().str.strip()
    )
    counts = country_series.value_counts().reset_index()
    counts.columns = ["country", "count"]
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts.head(top_n)


def ratings_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rating_seg = df[["rating", "audience_segment"]].drop_duplicates()
    dist = (
        df.groupby("rating").size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    dist = dist.merge(rating_seg, on="rating", how="left")
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    return dist


def duration_analysis_movies(df: pd.DataFrame) -> dict:
    movies = df[df["type"] == "Movie"]["duration_minutes"].dropna()

    if len(movies) == 0:
        movies = (
            df[df["type"] == "Movie"]["duration"]
            .str.extract(r"(\d+)")[0].astype(float).dropna()
        )

    mode_val = movies.mode()
    stats = {
        "mean_minutes":   round(movies.mean(), 1),
        "median_minutes": round(movies.median(), 1),
        "mode_minutes":   int(mode_val.iloc[0]) if len(mode_val) > 0 else 90,
        "std_minutes":    round(movies.std(), 1),
        "min_minutes":    int(movies.min()) if len(movies) > 0 else 0,
        "max_minutes":    int(movies.max()) if len(movies) > 0 else 0,
        "count":          len(movies),
    }
    bins   = [0, 60, 90, 120, 150, 300]
    labels = ["<60 min", "60-90 min", "90-120 min", "120-150 min", "150+ min"]
    buckets = pd.cut(movies, bins=bins, labels=labels)
    stats["buckets"] = buckets.value_counts().sort_index().to_dict()
    return stats


def duration_analysis_shows(df: pd.DataFrame) -> pd.DataFrame:
    shows = df[df["type"] == "TV Show"]["duration_seasons"].dropna()
    dist = shows.astype(int).value_counts().sort_index().reset_index()
    dist.columns = ["seasons", "count"]
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    return dist


def content_age_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["content_age_group", "type"], observed=True)
          .size().reset_index(name="count")
          .dropna(subset=["content_age_group"])
    )


def monthly_additions(df: pd.DataFrame) -> pd.DataFrame:
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = (
        df.dropna(subset=["added_month"])
          .groupby(["added_month_name", "type"]).size()
          .reset_index(name="count")
    )
    monthly["added_month_name"] = pd.Categorical(
        monthly["added_month_name"], categories=month_order, ordered=True
    )
    return monthly.sort_values("added_month_name")


def top_directors(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    return (
        df[df["director"] != "Unknown"]
          .groupby(["director", "type"]).size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .head(top_n)
    )


def generate_kpis(df: pd.DataFrame) -> dict:
    movies = df[df["type"] == "Movie"]
    shows  = df[df["type"] == "TV Show"]
    return {
        "total_titles":       len(df),
        "total_movies":       len(movies),
        "total_shows":        len(shows),
        "unique_countries":   df[df["country"] != "Unknown"]["country"]
                                .str.split(",").explode().str.strip().nunique(),
        "unique_genres":      df["listed_in"].str.split(",").explode().str.strip().nunique(),
        "unique_directors":   df[df["director"] != "Unknown"]["director"].nunique(),
        "avg_movie_runtime":  round(movies["duration_minutes"].mean(), 1),
        "avg_show_seasons":   round(shows["duration_seasons"].mean(), 1),
        "year_range":         f"{int(df['release_year'].min())} - {int(df['release_year'].max())}",
        "most_common_rating": df["rating"].mode()[0],
    }


def run_full_eda(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  Netflix Content Intelligence — Full EDA")
    print("=" * 55)

    results = {}
    results["kpis"]               = generate_kpis(df)
    results["type_dist"]          = content_type_distribution(df)
    results["yearly_additions"]   = year_wise_additions(df)
    results["release_trend"]      = release_year_trend(df)
    results["genre_top15"]        = genre_analysis(df, top_n=15)
    results["genre_by_type"]      = genre_by_type(df, top_n=10)
    results["country_top15"]      = country_analysis(df, top_n=15)
    results["ratings_dist"]       = ratings_distribution(df)
    results["movie_duration_stats"] = duration_analysis_movies(df)
    results["show_seasons"]       = duration_analysis_shows(df)
    results["content_age"]        = content_age_distribution(df)
    results["monthly_additions"]  = monthly_additions(df)
    results["top_directors"]      = top_directors(df, top_n=15)

    kpis = results["kpis"]
    print(f"\n  Total Titles     : {kpis['total_titles']:,}")
    print(f"  Movies           : {kpis['total_movies']:,}")
    print(f"  TV Shows         : {kpis['total_shows']:,}")
    print(f"  Avg Movie Runtime: {kpis['avg_movie_runtime']} min")
    print(f"  Avg Show Seasons : {kpis['avg_show_seasons']}")

    print("\n  EDA complete!\n")
    return results


if __name__ == "__main__":
    df = load_clean_data()
    results = run_full_eda(df)