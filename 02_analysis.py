"""
working
CCT416 - Step 2: Analysis
Reads data/spotify_top50_raw.csv and produces:
  - Descriptive statistics per country
  - Genre frequency tables
  - Country-level audio feature summary (for heatmap)
  - Saved to data/summary_stats.csv and data/genre_counts.csv

Run after 01_collect_data.py
"""

import pandas as pd
import numpy as np
import os

AUDIO_FEATURES = [
    "danceability", "energy", "valence",
    "tempo", "acousticness", "speechiness",
    "loudness", "instrumentalness"
]

COUNTRIES_ORDER = [
    "United States", "Brazil", "Mexico",       # Americas / English+Spanish+Portuguese
    "Nigeria",                                  # Africa / Afrobeats
    "Sweden",                                   # Scandinavia
    "India", "South Korea", "Japan"             # Asia
]

def load_data(path="data/spotify_top50_raw.csv"):
    df = pd.read_csv(path)
    df["artist_genres"] = df["artist_genres"].fillna("unknown")
    return df

def descriptive_stats(df):
    """Mean ± std of audio features per country."""
    stats = df.groupby("country")[AUDIO_FEATURES].agg(["mean", "std"]).round(3)
    stats = stats.reindex([c for c in COUNTRIES_ORDER if c in stats.index])
    return stats

def genre_frequency(df, top_n=5):
    """Top N genres per country."""
    results = {}
    for country, group in df.groupby("country"):
        # Explode pipe-separated genres
        genres = (
            group["artist_genres"]
            .str.split(r"\s*\|\s*")
            .explode()
            .str.strip()
            .replace("unknown", np.nan)
            .dropna()
        )
        top = genres.value_counts().head(top_n)
        results[country] = top
    return results

def country_feature_summary(df):
    """Mean audio features per country — used for the heatmap."""
    summary = df.groupby("country")[AUDIO_FEATURES].mean().round(4)
    summary = summary.reindex([c for c in COUNTRIES_ORDER if c in summary.index])
    return summary

def popularity_summary(df):
    """Mean track popularity per country."""
    pop = df.groupby("country")["track_popularity"].agg(["mean","std","min","max"]).round(2)
    pop = pop.reindex([c for c in COUNTRIES_ORDER if c in pop.index])
    return pop

def pairwise_feature_diff(summary_df):
    """
    Pairwise Euclidean distance between countries based on normalized audio features.
    Returns a square DataFrame.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(summary_df),
        index=summary_df.index,
        columns=summary_df.columns
    )
    n = len(normed)
    dist = pd.DataFrame(np.zeros((n, n)), index=normed.index, columns=normed.index)
    for i in normed.index:
        for j in normed.index:
            dist.loc[i, j] = np.sqrt(((normed.loc[i] - normed.loc[j]) ** 2).sum())
    return dist.round(4)

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['country'].nunique()} countries\n")

    # ── Descriptive stats ──────────────────────────────────────────────────────
    print("=== Descriptive Statistics (mean ± std per country) ===")
    stats = descriptive_stats(df)
    print(stats.to_string())
    stats.to_csv("data/summary_stats.csv")

    # ── Genre frequency ────────────────────────────────────────────────────────
    print("\n=== Top 5 Genres per Country ===")
    genre_data = genre_frequency(df)
    genre_rows = []
    for country, series in genre_data.items():
        print(f"\n  {country}:")
        for genre, count in series.items():
            print(f"    {genre}: {count}")
            genre_rows.append({"country": country, "genre": genre, "count": count})
    genre_df = pd.DataFrame(genre_rows)
    genre_df.to_csv("data/genre_counts.csv", index=False)

    # ── Country feature summary ────────────────────────────────────────────────
    print("\n=== Country Audio Feature Summary (for heatmap) ===")
    summary = country_feature_summary(df)
    print(summary.to_string())
    summary.to_csv("data/country_feature_summary.csv")

    # ── Popularity ─────────────────────────────────────────────────────────────
    print("\n=== Track Popularity per Country ===")
    pop = popularity_summary(df)
    print(pop.to_string())

    # ── Pairwise distance ──────────────────────────────────────────────────────
    print("\n=== Pairwise Country Distance (lower = more similar) ===")
    dist = pairwise_feature_diff(summary)
    print(dist.to_string())
    dist.to_csv("data/country_pairwise_distance.csv")

    print("\nAnalysis complete. Files saved to data/")

if __name__ == "__main__":
    main()
