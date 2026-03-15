"""
working
CCT416 - Step 1c: Add Genre Labels via Artist Name Join
Joins genre dataset onto chart data by artist name (fuzzy-tolerant).
Much higher match rate than track ID join.

Requires: spotify_genres_dataset.csv in same folder
Run AFTER 01b_prepare_kaggle_data.py
Overwrites: data/spotify_top50_raw.csv
"""

import pandas as pd
import re

GENRE_FILE = "spotify_genres_dataset.csv"
CHART_FILE = "data/spotify_top50_raw.csv"

def normalize_name(name):
    """Lowercase, strip punctuation, trim whitespace."""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

print("Loading chart data...")
df_chart = pd.read_csv(CHART_FILE)
print(f"  {len(df_chart)} chart rows")

print("\nLoading genre dataset...")
df_genres = pd.read_csv(GENRE_FILE, usecols=["artists", "track_genre"])

# Build artist → most common genre map
df_genres["artist_norm"] = df_genres["artists"].apply(normalize_name)

# Some tracks have multiple artists separated by ;
# Explode so each artist gets their own row
df_genres["artist_norm"] = df_genres["artist_norm"].str.split(";")
df_genres = df_genres.explode("artist_norm")
df_genres["artist_norm"] = df_genres["artist_norm"].str.strip()
df_genres = df_genres[df_genres["artist_norm"] != ""]

# For each artist, pick their most common genre label
artist_genre_map = (
    df_genres.groupby("artist_norm")["track_genre"]
    .agg(lambda x: x.value_counts().index[0])  # most frequent genre
    .to_dict()
)
print(f"  Built genre map for {len(artist_genre_map)} unique artists")

# Normalize chart artist names
df_chart["artist_norm"] = df_chart["artist_name"].apply(normalize_name)

# Also try first artist only (some chart entries have "Artist1, Artist2")
def first_artist(name):
    if pd.isna(name):
        return ""
    # Split on comma, &, feat., x
    parts = re.split(r",|&|feat\.|ft\.|featuring|\sx\s", str(name), flags=re.IGNORECASE)
    return normalize_name(parts[0])

df_chart["artist_first"] = df_chart["artist_name"].apply(first_artist)

# Match: try full name first, then first artist
def lookup_genre(row):
    g = artist_genre_map.get(row["artist_norm"])
    if g:
        return g
    return artist_genre_map.get(row["artist_first"], None)

df_chart["matched_genre"] = df_chart.apply(lookup_genre, axis=1)

matched = df_chart["matched_genre"].notna().sum()
print(f"\nMatched {matched}/{len(df_chart)} tracks ({matched/len(df_chart)*100:.1f}%)")

# Fill artist_genres
df_chart["artist_genres"] = df_chart["matched_genre"].fillna("unknown")
df_chart = df_chart.drop(columns=["artist_norm", "artist_first", "matched_genre"])

# Save
df_chart.to_csv(CHART_FILE, index=False)
print(f"Saved updated data to {CHART_FILE}")

print("\nGenre match rate per country:")
for country, grp in df_chart.groupby("country"):
    matched = (grp["artist_genres"] != "unknown").sum()
    pct = matched / len(grp) * 100
    print(f"  {country}: {matched}/{len(grp)} ({pct:.0f}%)")

print("\nTop genres found:")
genre_counts = (
    df_chart[df_chart["artist_genres"] != "unknown"]["artist_genres"]
    .value_counts().head(20)
)
print(genre_counts.to_string())