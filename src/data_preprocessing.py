"""
data_preprocessing.py
----------------------
Loading and preprocessing utilities, matched to the column names of
Kaggle's "Housing Prices Dataset" (Housing.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]

CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

TARGET = "price"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df: pd.DataFrame):
    """Split a dataframe into feature matrix X and target vector y."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Scales numeric columns and one-hot-encodes categorical columns.
    Used as the first step of an sklearn Pipeline so training and
    prediction apply the exact same transformation.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
