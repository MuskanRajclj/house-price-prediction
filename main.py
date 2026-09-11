"""
main.py
-------
Entry point for the House Price Prediction project.
Run from the project root: `python main.py`
"""

import subprocess
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")


def run_script(script_name: str):
    script_path = os.path.join(SRC_DIR, script_name)
    subprocess.run([sys.executable, script_path], cwd=os.path.dirname(__file__))


def menu():
    while True:
        print("\n===== House Price Prediction =====")
        print("1. Generate sample dataset (Kaggle Housing.csv schema)")
        print("2. Train model")
        print("3. Predict a house price")
        print("4. Exit")
        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            run_script("generate_dataset.py")
        elif choice == "2":
            run_script("train_model.py")
        elif choice == "3":
            run_script("predict.py")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    menu()
