"""
Netflix Content Intelligence Dashboard
=======================================
Module 1 — Data Cleaning & Feature Engineering
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw",       "netflix_titles.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "netflix_cleaned.csv")


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'─'*55}")
    print(f"  Raw data loaded: {path}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'─'*55}")
    return df


def audit_missing_values(df: pd.DataFrame) -> None:
    total = len(df)
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("  No missing values found.")
        return
    print("\n  Missing Value Audit:")
    print(f"  {'Column':<20} {'Missing':>8}  {'%':>7}")
    print(f"  {'─'*38}")
    for col, cnt in miss.items():
        pct = cnt / total * 100
        print(f"  {col:<20} {cnt:>8,}  {pct:>6.1f}%")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["director", "cast", "country"]:
        df[col] = df[col].fillna("Unknown")
    df["date_added"] = df["date_added"].fillna("Not Available")
    rating_mode = df["rating"].mode()[0]
    df["rating"] = df["rating"].fillna(rating_mode)
    print(f"\n  Missing values handled. Mode rating used: '{rating_mode}'")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"  Duplicates removed: {removed:,}  ({before:,} -> {len(df):,} rows)")
    return df


def parse_date_added(df: pd.DataFrame) -> pd.DataFrame:
    df["date_added_parsed"] = pd.to_datetime(
        df["date_added"].replace("Not Available", np.nan),
        format="%B %d, %Y",
        errors="coerce",
    )
    df["added_year"]       = df["date_added_parsed"].dt.year.astype("Int64")
    df["added_month"]      = df["date_added_parsed"].dt.month.astype("Int64")
    df["added_month_name"] = df["date_added_parsed"].dt.strftime("%b")
    print("  Date columns parsed and extracted.")
    return df


def engineer_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    df["duration_minutes"] = np.nan
    df["duration_seasons"] = np.nan

    movie_mask = df["type"] == "Movie"
    df.loc[movie_mask, "duration_minutes"] = (
        df.loc[movie_mask, "duration"]
          .str.extract(r"(\d+)")[0]
          .astype(float)
    )

    show_mask = df["type"] == "TV Show"
    df.loc[show_mask, "duration_seasons"] = (
        df.loc[show_mask, "duration"]
          .str.extract(r"(\d+)")[0]
          .astype(float)
    )

    df["duration_minutes"] = pd.array(df["duration_minutes"], dtype="Int64")
    df["duration_seasons"] = pd.array(df["duration_seasons"], dtype="Int64")
    print("  Duration features engineered.")
    return df


def engineer_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    df["genre_count"]   = df["listed_in"].apply(lambda x: len(x.split(",")))
    df["primary_genre"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())
    print("  Genre features engineered.")
    return df


def engineer_content_age(df: pd.DataFrame) -> pd.DataFrame:
    reference_year = 2023
    df["content_age_years"] = reference_year - df["release_year"]
    bins   = [0, 2, 5, 10, 20, 100]
    labels = ["Very Recent (0-2y)", "Recent (3-5y)", "Moderate (6-10y)",
              "Classic (11-20y)", "Vintage (20y+)"]
    df["content_age_group"] = pd.cut(
        df["content_age_years"], bins=bins, labels=labels, right=True
    )
    print("  Content age features engineered.")
    return df


def classify_rating_audience(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "G": "Kids",    "TV-Y": "Kids",    "TV-Y7": "Kids",
        "PG": "Family", "TV-G": "Family",  "TV-PG": "Family",
        "PG-13": "Teens", "TV-14": "Teens",
        "R": "Adults",  "TV-MA": "Adults", "NC-17": "Adults",
        "NR": "Unrated", "UR": "Unrated",
    }
    df["audience_segment"] = df["rating"].map(mapping).fillna("Unrated")
    print("  Audience segments classified.")
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def run_cleaning_pipeline(raw_path: str = RAW_PATH,
                           clean_path: str = CLEAN_PATH) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  Netflix Content Intelligence — Data Cleaning")
    print("=" * 55)

    df = load_raw_data(raw_path)
    audit_missing_values(df)

    print("\n  Running cleaning steps ...")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = standardise_columns(df)
    df = parse_date_added(df)
    df = engineer_duration_features(df)
    df = engineer_genre_features(df)
    df = engineer_content_age(df)
    df = classify_rating_audience(df)

    df = df.drop(columns=["date_added_parsed"], errors="ignore")

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"\n  Cleaned data saved -> {clean_path}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n" + "=" * 55)
    print("  Cleaning pipeline complete!")
    print("=" * 55 + "\n")
    return df


if __name__ == "__main__":
    cleaned_df = run_cleaning_pipeline()