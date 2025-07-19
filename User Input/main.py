import sys
from pathlib import Path

# Add model_training folder to sys.path to import predict.py
model_training_path = Path(__file__).parent.parent / "model_training"
sys.path.append(str(model_training_path))

from predict import predict_from_partial

def get_previous_pattern():
    valid_patterns = ["large_spike", "small_spike", "decreasing", "random", "unknown"]
    while True:
        user_input = input(f"Enter the previous week's pattern {valid_patterns} (or 'unknown'): ").strip().lower()
        if user_input in valid_patterns:
            return user_input
        print(f"Invalid input. Please enter one of {valid_patterns}.")

def get_prices_from_user():
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
    previous_pattern = get_previous_pattern()
    partial_prices = get_prices_from_user()

    if len(partial_prices) == 0:
        print("No data entered. Exiting.")
        return

    # Trim input to max 12 slots (full week)
    partial_prices = partial_prices[:12]

    # Remove trailing Nones for cleaner input
    while partial_prices and partial_prices[-1] is None:
        partial_prices.pop()

    if len(partial_prices) == 0:
        print("No valid prices entered. Exiting.")
        return

    predict_from_partial(partial_prices, model_folder=str(model_training_path), previous_pattern=previous_pattern)

if __name__ == "__main__":
    main()