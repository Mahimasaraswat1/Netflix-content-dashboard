"""
Netflix Content Intelligence Dashboard
=======================================
Dataset Generator
Generates a realistic Netflix-like CSV dataset with intentional imperfections
(missing values, duplicates) so the cleaning pipeline has something meaningful to do.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

random.seed(42)
np.random.seed(42)

COUNTRIES = [
    "United States", "India", "United Kingdom", "Canada", "France",
    "Germany", "Japan", "South Korea", "Spain", "Brazil", "Mexico",
    "Italy", "Australia", "Nigeria", "Turkey", "Argentina", "Sweden",
    "Norway", "Denmark", "Thailand", "Indonesia", "Philippines",
]

GENRES = [
    "Drama", "Comedy", "Action & Adventure", "Documentaries", "Horror",
    "Thrillers", "Romantic Movies", "Children & Family Movies",
    "Anime Features", "International Movies", "Crime TV Shows",
    "Stand-Up Comedy", "Reality TV", "Science & Nature TV",
    "Sports Movies", "Independent Movies", "Music & Musicals",
    "Teen TV Shows", "British TV Shows", "Korean TV Shows",
]

RATINGS = [
    "G", "PG", "PG-13", "R", "NC-17",
    "TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA", "NR", "UR",
]

DIRECTORS = [
    "Christopher Nolan", "Ava DuVernay", "Bong Joon-ho", "Greta Gerwig",
    "Martin Scorsese", "Spike Lee", "Alfonso Cuarón", "Kathryn Bigelow",
    "Steven Spielberg", "Jordan Peele", "Ryan Coogler", "Chloe Zhao",
    "David Fincher", "Denis Villeneuve", "Taika Waititi", "Barry Jenkins",
    "Wes Anderson", "Guillermo del Toro", "Sofia Coppola", "Park Chan-wook",
    None, None, None,
]

ACTORS = [
    "Leonardo DiCaprio", "Meryl Streep", "Denzel Washington", "Cate Blanchett",
    "Tom Hanks", "Viola Davis", "Brad Pitt", "Natalie Portman",
    "Ryan Gosling", "Zendaya", "Idris Elba", "Ana de Armas",
    "Anthony Mackie", "Florence Pugh", "Oscar Isaac", "Lupita Nyong'o",
]

MOVIE_TITLES = [
    "Midnight in Paris", "Dark Waters", "Crimson Peak", "The Silent House",
    "Beyond the Horizon", "A Beautiful Mind", "Shadow Protocol", "Glass Houses",
    "Neon Dreams", "The Forgotten Path", "Echoes of Tomorrow", "Broken Wings",
    "Steel Hearts", "The Last Train", "Ocean's Edge", "Fire & Ice",
    "Parallel Lives", "The Ghost Network", "Shattered Mirror", "Raw Power",
    "Velvet Underground", "The Iron Curtain", "Digital Nomad", "City of Lights",
    "The Final Frontier", "Quantum Leap", "Starfall", "Code Red",
    "The Phoenix Project", "Whispering Walls", "Frozen in Time", "Edge of Glory",
    "The Redemption Arc", "Silent Storm", "Golden Hour", "The Last Dance",
    "Paper Trail", "Hollow Ground", "Desert Rain", "Northern Lights",
]

SHOW_TITLES = [
    "Stranger Things", "Money Heist", "The Witcher", "Bridgerton", "Ozark",
    "Dark", "Lupin", "Squid Game", "Emily in Paris", "The Crown",
    "House of Cards", "Mindhunter", "Black Mirror", "Narcos", "Peaky Blinders",
    "Better Call Saul", "Altered Carbon", "The Haunting of Hill House",
    "You", "Locke & Key", "Warrior Nun", "The Umbrella Academy",
    "Sweet Tooth", "Jupiter's Legacy", "Shadow and Bone", "Cobra Kai",
    "Never Have I Ever", "On My Block", "Outer Banks",
    "Control Z", "Ginny & Georgia", "The Circle", "Too Hot to Handle",
]

DESCRIPTIONS = [
    "A gripping tale of power, betrayal, and survival in a world where nothing is as it seems.",
    "When secrets from the past resurface, a family must confront the truth before it destroys them.",
    "In a dystopian future, one person dares to challenge the system and ignite a revolution.",
    "Love, loss, and redemption collide in this emotionally charged journey across three continents.",
    "A detective with a troubled past investigates a series of mysterious disappearances.",
    "Five strangers are brought together by fate in a city that never sleeps.",
    "Based on true events, this documentary exposes the hidden world behind the headlines.",
    "An unlikely hero rises to face an impossible challenge in this action-packed adventure.",
    "A comedy that brilliantly captures the chaos and joy of modern family life.",
    "Set against the backdrop of history, a love story that transcends time and borders.",
]


def generate_netflix_dataset(n: int = 8500) -> pd.DataFrame:
    records = []
    counter = 1
    years = list(range(2000, 2023))
    weights = [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 10, 8]

    for i in range(n):
        content_type = random.choices(["Movie", "TV Show"], weights=[0.70, 0.30])[0]

        director = random.choice(DIRECTORS) if random.random() > 0.10 else None
        cast_list = random.sample(ACTORS, k=random.randint(2, 5))
        cast = ", ".join(cast_list) if random.random() > 0.05 else None
        country = random.choice(COUNTRIES) if random.random() > 0.08 else None
        rating = random.choice(RATINGS) if random.random() > 0.04 else None

        year = random.choices(years, weights=weights)[0]
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date_added = (
            datetime(year, month, day).strftime("%B %d, %Y")
            if random.random() > 0.07 else None
        )

        if content_type == "Movie":
            base = random.choice(MOVIE_TITLES)
            title = base + (f" {random.randint(2, 4)}" if random.random() > 0.75 else "")
            mins = random.randint(60, 180)
            duration = f"{mins} min"
        else:
            title = random.choice(SHOW_TITLES)
            num_seasons = random.randint(1, 8)
            duration = f"{num_seasons} Season{'s' if num_seasons > 1 else ''}"

        num_genres = random.randint(1, 3)
        listed_in = ", ".join(random.sample(GENRES, num_genres))

        row = {
            "show_id": f"s{counter}",
            "type": content_type,
            "title": title,
            "director": director,
            "cast": cast,
            "country": country,
            "date_added": date_added,
            "release_year": year,
            "rating": rating,
            "duration": duration,
            "listed_in": listed_in,
            "description": random.choice(DESCRIPTIONS),
        }
        records.append(row)
        counter += 1

        if records and random.random() < 0.02:
            records.append(records[random.randint(0, len(records) - 1)].copy())

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("  Netflix Content Intelligence — Dataset Generator")
    print("=" * 60)

    df = generate_netflix_dataset(8500)
    out = "data/raw/netflix_titles.csv"       # ← relative path for YOUR machine
    df.to_csv(out, index=False)

    print(f"\n✅  Raw dataset saved → {out}")
    print(f"    Rows   : {len(df):,}")
    print(f"    Columns: {df.shape[1]}")
    print(f"\nMissing value snapshot:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")