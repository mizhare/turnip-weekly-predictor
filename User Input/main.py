import sys
import os
from pathlib import Path

# Add the model_training folder to sys.path so we can import predict.py
model_training_path = Path(__file__).parent.parent / "model_training"
sys.path.append(str(model_training_path))

from predict import predict_from_partial

def get_prices_from_user():
    """
    Prompt the user to enter turnip prices for each day and period (AM/PM).
    User can enter:
     - integer prices between 0 and 660
     - 'skip' or empty to skip (stored as None)
     - 'quit' or 'stop' to exit input early
    Returns a list of prices or None for skipped values.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    periods = ["AM", "PM"]
    prices = []

    print("📥 Please enter turnip prices for each time period (AM/PM).")
    print("Enter a number between 0 and 660, 'skip' or empty to skip, 'quit' or 'stop' to finish input.\n")

    for day in days:
        for period in periods:
            while True:
                user_input = input(f"{day} {period}: ").strip().lower()

                if user_input in ['quit', 'stop']:
                    print("⚠️ Input stopped by user.")
                    return prices

                elif user_input in ['skip', '']:
                    prices.append(None)
                    break

                else:
                    try:
                        value = int(user_input)
                        if 0 <= value <= 660:
                            prices.append(value)
                            break
                        else:
                            print("❌ Invalid number. Please enter 0-660, 'skip' or 'quit'.")
                    except ValueError:
                        print("❌ Invalid input. Please enter an integer, 'skip' or 'quit'.")

    return prices

def main():
    partial_prices = get_prices_from_user()

    # If user entered no data, exit early
    if len(partial_prices) == 0:
        print("No data entered. Exiting.")
        return

    # Clean input: take only up to first 11 values (max prediction is for 12)
    partial_prices = partial_prices[:11]

    # Filter out None values at the end (optional, but keep order)
    while partial_prices and partial_prices[-1] is None:
        partial_prices.pop()

    if len(partial_prices) == 0:
        print("No valid prices entered. Exiting.")
        return

    # Call prediction function
    predict_from_partial(partial_prices, model_folder=str(model_training_path))

if __name__ == "__main__":
    main()