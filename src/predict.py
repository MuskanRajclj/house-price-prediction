"""
predict.py
----------
Loads the trained model and predicts the price of a new house.

Interactive mode:
    python src/predict.py

Scripted mode:
    python src/predict.py --area 6500 --bedrooms 3 --bathrooms 2 --stories 2 \
        --parking 1 --mainroad yes --guestroom no --basement no \
        --hotwaterheating no --airconditioning yes --prefarea yes \
        --furnishingstatus furnished
"""

import argparse
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "house_price_model.joblib")
YES_NO = ["yes", "no"]
FURNISHING = ["furnished", "semi-furnished", "unfurnished"]


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            f"Run 'python src/train_model.py' first."
        )
    return joblib.load(model_path)


def predict_price(model, features: dict) -> float:
    df = pd.DataFrame([features])
    return float(model.predict(df)[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a house price.")
    parser.add_argument("--area", type=float)
    parser.add_argument("--bedrooms", type=int)
    parser.add_argument("--bathrooms", type=int)
    parser.add_argument("--stories", type=int)
    parser.add_argument("--parking", type=int)
    parser.add_argument("--mainroad", type=str, choices=YES_NO)
    parser.add_argument("--guestroom", type=str, choices=YES_NO)
    parser.add_argument("--basement", type=str, choices=YES_NO)
    parser.add_argument("--hotwaterheating", type=str, choices=YES_NO)
    parser.add_argument("--airconditioning", type=str, choices=YES_NO)
    parser.add_argument("--prefarea", type=str, choices=YES_NO)
    parser.add_argument("--furnishingstatus", type=str, choices=FURNISHING)
    return parser.parse_args()


def prompt_choice(label: str, options: list) -> str:
    options_str = "/".join(options)
    while True:
        val = input(f"{label} ({options_str}): ").strip().lower()
        if val in options:
            return val
        print(f"Please enter one of: {options_str}")


def interactive_input() -> dict:
    print("Enter house details to predict its price:\n")
    features = {
        "area": float(input("Area (sqft): ")),
        "bedrooms": int(input("Bedrooms: ")),
        "bathrooms": int(input("Bathrooms: ")),
        "stories": int(input("Stories: ")),
        "parking": int(input("Parking spaces: ")),
        "mainroad": prompt_choice("On main road?", YES_NO),
        "guestroom": prompt_choice("Guest room?", YES_NO),
        "basement": prompt_choice("Basement?", YES_NO),
        "hotwaterheating": prompt_choice("Hot water heating?", YES_NO),
        "airconditioning": prompt_choice("Air conditioning?", YES_NO),
        "prefarea": prompt_choice("Preferred area?", YES_NO),
        "furnishingstatus": prompt_choice("Furnishing status", FURNISHING),
    }
    return features


def main():
    args = parse_args()
    model = load_model()

    arg_dict = vars(args)
    if all(v is not None for v in arg_dict.values()):
        features = arg_dict
    else:
        features = interactive_input()

    price = predict_price(model, features)
    print(f"\nPredicted house price: Rs. {price:,.2f}")


if __name__ == "__main__":
    main()
