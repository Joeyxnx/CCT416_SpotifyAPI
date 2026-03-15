"""
working
CCT416 - Step 3: Visualizations
Reads CSVs from data/ and produces all figures to figures/

Run after 02_analysis.py
Produces:
  figures/01_genre_bar_charts.png
  figures/02_boxplot_danceability.png
  figures/03_boxplot_energy.png
  figures/04_heatmap_audio_features.png
  figures/05_radar_chart.png
  figures/06_pairwise_distance_heatmap.png
  figures/07_valence_tempo_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings

warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────
PALETTE = [
    "#1DB954",  # Spotify green – US
    "#F7C948",  # Brazil
    "#E84855",  # Mexico
    "#5C6BC0",  # Nigeria
    "#26C6DA",  # Sweden
    "#FF7043",  # India
    "#AB47BC",  # South Korea
    "#EC407A",  # Japan
]

COUNTRIES_ORDER = [
    "United States", "Brazil", "Mexico",
    "Nigeria", "Sweden",
    "India", "South Korea", "Japan"
]

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw     = pd.read_csv("data/spotify_top50_raw.csv")
df_summary = pd.read_csv("data/country_feature_summary.csv", index_col=0)
df_dist    = pd.read_csv("data/country_pairwise_distance.csv", index_col=0)

# Genre data may be empty if dataset has no genre labels
try:
    df_genres = pd.read_csv("data/genre_counts.csv")
    has_genres = len(df_genres) > 0 and "genre" in df_genres.columns
except Exception:
    df_genres = pd.DataFrame()
    has_genres = False

# Enforce country order where possible
present = [c for c in COUNTRIES_ORDER if c in df_raw["country"].unique()]
color_map = {c: PALETTE[i] for i, c in enumerate(present)}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Top 5 Genres per Country (faceted bar charts)
# ─────────────────────────────────────────────────────────────────────────────
def fig_genre_bars():
    if not has_genres:
        print("  Skipping genre chart (no genre data available)", flush=True)
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Top 5 Genres per Country — Spotify Top 50", fontsize=15, fontweight="bold", y=1.01)

    for ax, country in zip(axes.flat, present):
        subset = df_genres[df_genres["country"] == country].nlargest(5, "count")
        color = color_map[country]
        bars = ax.barh(subset["genre"], subset["count"], color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(country, fontsize=11, fontweight="bold")
        ax.set_xlabel("Track count", fontsize=9)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/01_genre_bar_charts.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/01_genre_bar_charts.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 & 3: Boxplots — Danceability and Energy by Country
# ─────────────────────────────────────────────────────────────────────────────
def fig_boxplot(feature, filename, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    data_ordered = [df_raw[df_raw["country"] == c][feature].dropna().values for c in present]

    bp = ax.boxplot(
        data_ordered,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, linestyle="none", alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], [color_map[c] for c in present]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticklabels(present, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(feature.capitalize(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # Reference line at global mean
    global_mean = df_raw[feature].mean()
    ax.axhline(global_mean, color="grey", linestyle="--", linewidth=1, alpha=0.7, label=f"Global mean ({global_mean:.2f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/{filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Heatmap of Average Audio Features by Country
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap():
    # Normalize each feature 0–1 for visual comparability
    from sklearn.preprocessing import MinMaxScaler
    features = ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness"]

    summary_sub = df_summary[features].reindex([c for c in COUNTRIES_ORDER if c in df_summary.index])
    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(summary_sub),
        index=summary_sub.index,
        columns=summary_sub.columns
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        normed,
        annot=summary_sub.round(2),   # show raw values in cells
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Normalized value (0–1)"},
        annot_kws={"size": 9},
    )
    ax.set_title("Average Audio Features by Country\n(colour = normalized, numbers = raw mean)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("figures/04_heatmap_audio_features.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/04_heatmap_audio_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Radar / Spider Chart — Country Sonic Profiles
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar():
    from matplotlib.patches import FancyArrowPatch
    features = ["danceability", "energy", "valence", "acousticness", "speechiness"]
    N = len(features)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    from sklearn.preprocessing import MinMaxScaler
    summary_sub = df_summary[features].reindex([c for c in COUNTRIES_ORDER if c in df_summary.index])
    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(summary_sub),
        index=summary_sub.index,
        columns=summary_sub.columns
    )

    for i, (country, row) in enumerate(normed.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color = color_map.get(country, "#888888")
        ax.plot(angles, values, "o-", linewidth=1.8, color=color, label=country, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Sonic Profile Radar — 8 Countries\n(normalized audio features)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/05_radar_chart.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/05_radar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Pairwise Distance Heatmap (country similarity)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pairwise_heatmap():
    ordered = [c for c in COUNTRIES_ORDER if c in df_dist.index]
    dist_ordered = df_dist.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(dist_ordered), dtype=bool)  # mask diagonal
    sns.heatmap(
        dist_ordered,
        annot=True,
        fmt=".2f",
        cmap="Blues_r",
        mask=mask,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Euclidean distance (lower = more similar)"},
        annot_kws={"size": 9},
    )
    ax.set_title("Pairwise Country Similarity\n(based on normalized audio features)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("figures/06_pairwise_distance_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/06_pairwise_distance_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Scatter — Valence vs Tempo, coloured by country
# ─────────────────────────────────────────────────────────────────────────────
def fig_valence_tempo_scatter():
    fig, ax = plt.subplots(figsize=(11, 7))

    for country in present:
        sub = df_raw[df_raw["country"] == country]
        ax.scatter(
            sub["tempo"], sub["valence"],
            label=country,
            color=color_map[country],
            alpha=0.65,
            s=55,
            edgecolors="white",
            linewidth=0.4,
        )

    ax.set_xlabel("Tempo (BPM)", fontsize=11)
    ax.set_ylabel("Valence (positivity)", fontsize=11)
    ax.set_title("Valence vs. Tempo by Country\nEach point = one track in the Top 50",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Country", fontsize=9, title_fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/07_valence_tempo_scatter.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/07_valence_tempo_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig_genre_bars()
    fig_boxplot("danceability", "02_boxplot_danceability.png",
                "Danceability Distribution by Country — Spotify Top 50")
    fig_boxplot("energy",       "03_boxplot_energy.png",
                "Energy Distribution by Country — Spotify Top 50")
    fig_heatmap()
    fig_radar()
    fig_pairwise_heatmap()
    fig_valence_tempo_scatter()
    print("\nAll figures saved to figures/")