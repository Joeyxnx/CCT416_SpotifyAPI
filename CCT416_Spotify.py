import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd


CLIENT_ID = "94d7465f6e3f41aa84a80a42f7fce026"
CLIENT_SECRET = "c451528f51944fea925db72e7f0c6d2c"
REDIRECT_URI = "http://localhost:8888/callback"

scope = "user-top-read user-read-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=scope
))

# Get your top tracks
results = sp.current_user_top_tracks(limit=20, time_range="medium_term")

tracks = []
for item in results["items"]:
    tracks.append({
        "track_name": item["name"],
        "artist": item["artists"][0]["name"],
        "popularity": item["popularity"],
        "track_id": item["id"]
    })

df = pd.DataFrame(tracks)

print(df.head())
