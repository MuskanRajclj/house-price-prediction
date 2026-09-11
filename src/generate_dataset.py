"""
generate_dataset.py
--------------------
Generates a house-price dataset that follows the EXACT same column
structure as Kaggle's popular "Housing Prices Dataset" (Housing.csv,
by yasserh): https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

Columns: price, area, bedrooms, bathrooms, stories, mainroad, guestroom,
basement, hotwaterheating, airconditioning, parking, prefarea,
furnishingstatus

Because this environment cannot reach kaggle.com directly, this script
creates a synthetic-but-realistic dataset with the SAME schema, so the
rest of the project (preprocessing, training, prediction) works
identically. If you download the real CSV from Kaggle, just replace
`data/Housing.csv` with it -- no code changes needed, since the column
names match exactly.

Run:
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 545  # same row count as the real Kaggle dataset


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area = rng.integers(1650, 16200, n_samples)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples)
    stories = rng.integers(1, 5, n_samples)
    parking = rng.integers(0, 4, n_samples)

    mainroad = rng.choice(["yes", "no"], n_samples, p=[0.85, 0.15])
    guestroom = rng.choice(["yes", "no"], n_samples, p=[0.3, 0.7])
    basement = rng.choice(["yes", "no"], n_samples, p=[0.35, 0.65])
    hotwaterheating = rng.choice(["yes", "no"], n_samples, p=[0.1, 0.9])
    airconditioning = rng.choice(["yes", "no"], n_samples, p=[0.4, 0.6])
    prefarea = rng.choice(["yes", "no"], n_samples, p=[0.25, 0.75])

    furnishingstatus = rng.choice(
        ["furnished", "semi-furnished", "unfurnished"], n_samples, p=[0.3, 0.4, 0.3]
    )

    furnishing_premium = {"furnished": 300_000, "semi-furnished": 150_000, "unfurnished": 0}

    price = (
        area * 250
        + bedrooms * 200_000
        + bathrooms * 300_000
        + stories * 150_000
        + parking * 100_000
        + np.where(mainroad == "yes", 250_000, 0)
        + np.where(guestroom == "yes", 150_000, 0)
        + np.where(basement == "yes", 200_000, 0)
        + np.where(hotwaterheating == "yes", 180_000, 0)
        + np.where(airconditioning == "yes", 350_000, 0)
        + np.where(prefarea == "yes", 400_000, 0)
        + np.array([furnishing_premium[f] for f in furnishingstatus])
        + rng.normal(0, 400_000, n_samples)
    )

    price = np.clip(price, 1_500_000, None).round(-4)

    df = pd.DataFrame(
        {
            "price": price.astype(int),
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "Housing.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
