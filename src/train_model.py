"""
train_model.py
---------------
Trains and evaluates Linear Regression and Random Forest on the housing
dataset, picks the better model, and saves it to
`models/house_price_model.joblib`.

Run:
    python src/train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import (
    load_data,
    split_features_target,
    build_preprocessor,
    train_test_split_data,
)

DATA_PATH = os.path.join("data", "Housing.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.joblib")
OUTPUT_DIR = "outputs"


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R^2  : {r2:.4f}")
    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def plot_predictions(y_test, preds, name: str, out_path: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5, edgecolor="k")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Ideal fit")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Run 'python src/generate_dataset.py' first, "
            f"or place a Kaggle Housing.csv file at that path."
        )

    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    preprocessor = build_preprocessor()

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    results = []
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        result = evaluate(pipeline, X_test, y_test, name)
        plot_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_').lower()}_fit.png")
        plot_predictions(y_test, result["preds"], name, plot_path)
        results.append(result)

    best = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best['name']} (R^2 = {best['r2']:.4f})")

    joblib.dump(best["model"], MODEL_PATH)
    print(f"Saved best model pipeline to {MODEL_PATH}")

    summary_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    pd.DataFrame(
        [{"model": r["name"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"]} for r in results]
    ).to_csv(summary_path, index=False)
    print(f"Saved model comparison metrics to {summary_path}")


if __name__ == "__main__":
    main()
