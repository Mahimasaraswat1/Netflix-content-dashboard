"""
Netflix Content Intelligence Dashboard
=======================================
main.py — Master Pipeline Runner

Usage:
  python src/main.py               # full pipeline
  python src/main.py --skip-gen    # skip dataset generation
"""

import os
import sys
import time
import argparse
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from data_cleaning import run_cleaning_pipeline
from eda_analysis   import load_clean_data, run_full_eda
from recommender    import NetflixRecommender

RAW_PATH      = os.path.join(BASE_DIR, "data", "raw",       "netflix_titles.csv")
CLEAN_PATH    = os.path.join(BASE_DIR, "data", "processed", "netflix_cleaned.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def export_analysis_csvs(results: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    mapping = {
        "type_distribution":    results["type_dist"],
        "yearly_additions":     results["yearly_additions"],
        "release_trend":        results["release_trend"],
        "genre_top15":          results["genre_top15"],
        "genre_by_type":        results["genre_by_type"],
        "country_top15":        results["country_top15"],
        "ratings_distribution": results["ratings_dist"],
        "show_seasons":         results["show_seasons"],
        "monthly_additions":    results["monthly_additions"],
        "top_directors":        results["top_directors"],
    }
    for name, frame in mapping.items():
        if isinstance(frame, pd.DataFrame):
            path = os.path.join(out_dir, f"{name}.csv")
            frame.to_csv(path, index=False)
            print(f"  Saved: {name}.csv  ({len(frame)} rows)")

    pd.DataFrame([results["kpis"]]).to_csv(
        os.path.join(out_dir, "kpi_summary.csv"), index=False
    )
    print("  Saved: kpi_summary.csv")

    dur = results["movie_duration_stats"]
    pd.DataFrame(list(dur["buckets"].items()), columns=["bucket", "count"]).to_csv(
        os.path.join(out_dir, "movie_duration_buckets.csv"), index=False
    )
    print("  Saved: movie_duration_buckets.csv")


def run_pipeline(skip_generation: bool = False) -> None:
    start = time.time()

    print("\n" + "=" * 55)
    print("  NETFLIX CONTENT INTELLIGENCE — Pipeline v1.0")
    print("=" * 55)

    # Step 0 — Generate dataset
    if not skip_generation and not os.path.exists(RAW_PATH):
        print("\n[Step 0] Generating raw dataset ...")
        from generate_dataset import generate_netflix_dataset
        df_raw = generate_netflix_dataset(8500)
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        df_raw.to_csv(RAW_PATH, index=False)
        print(f"  Raw dataset saved ({len(df_raw):,} rows)")
    else:
        print(f"\n[Step 0] Raw dataset found -> {RAW_PATH}")

    # Step 1 — Clean
    print("\n[Step 1] Data Cleaning & Feature Engineering")
    df_clean = run_cleaning_pipeline(RAW_PATH, CLEAN_PATH)

    # Step 2 — EDA
    print("\n[Step 2] Exploratory Data Analysis")
    df = load_clean_data(CLEAN_PATH)
    results = run_full_eda(df)

    # Step 3 — Export CSVs
    print("\n[Step 3] Exporting analysis CSVs for Tableau ...")
    export_analysis_csvs(results, PROCESSED_DIR)

    # Step 4 — Recommendation demo
    print("\n[Step 4] Recommendation System Demo")
    rec = NetflixRecommender(df)
    sample_title = df["title"].iloc[0]
    print(f"\n  Titles similar to: '{sample_title}'")
    recs = rec.recommend(sample_title, top_n=5)
    print(recs[["title", "type", "primary_genre", "similarity_score"]].to_string(index=False))

    elapsed = time.time() - start
    print(f"\n  Pipeline complete! Total time: {elapsed:.1f}s")
    print(f"  All outputs saved to: {PROCESSED_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip raw dataset generation")
    args = parser.parse_args()
    run_pipeline(skip_generation=args.skip_gen)