import sys
from pathlib import Path

# Add model_training folder to sys.path to allow importing predict
model_training_path = Path(__file__).parent.parent / "model_training"
sys.path.append(str(model_training_path))

from predict import predict_from_partial


def get_previous_pattern():
    """Ask the user to input last week's pattern."""
    valid_patterns = ["large_spike", "small_spike", "decreasing", "fluctuating", "unknown"]
    while True:
        user_input = input(f"Enter the previous week's pattern {valid_patterns} (or 'unknown'): ").strip().lower()
        if user_input in valid_patterns:
            return user_input
        print(f"Invalid input. Please enter one of {valid_patterns}.")


def get_prices_from_user():
    """Collect turnip prices for the current week from the user."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    periods = ["AM", "PM"]
    prices = []

    print("Enter turnip prices for each time period (AM/PM). Type 'skip' or press Enter to skip a price, 'quit' or 'stop' to finish input.\n")

    for day in days:
        for period in periods:
            while True:
                user_input = input(f"{day} {period}: ").strip().lower()

                if user_input in ['quit', 'stop']:
                    print("Input stopped by user.")
                    return prices

                elif user_input in ['', 'skip']:
                    prices.append(None)
                    break

                else:
                    try:
                        value = int(user_input)
                        if 0 <= value <= 660:
                            prices.append(value)
                            break
                        else:
                            print("Please enter a number between 0 and 660, or 'skip', 'quit'.")
                    except ValueError:
                        print("Invalid input. Enter an integer, 'skip', or 'quit'.")

    return prices


def main():
    # Step 1: Get previous week's pattern
    previous_pattern = get_previous_pattern()

    # Step 2: Get partial turnip prices from the user
    partial_prices = get_prices_from_user()

    if not partial_prices:
        print("No prices entered. Exiting.")
        return

    # Remove trailing None values (e.g. if user just skipped at the end)
    while partial_prices and partial_prices[-1] is None:
        partial_prices.pop()

    if not partial_prices:
        print("No valid prices entered. Exiting.")
        return

    print("\nRunning prediction...\n")

    # Step 3: Predict the pattern and the rest of the week's prices
    predicted_pattern, predicted_prices = predict_from_partial(
        partial_prices=partial_prices,
        model_folder=str(model_training_path),
        previous_pattern_raw=previous_pattern)

    # Step 4: Print complete timeline of the week
    if predicted_prices:
        full_prices = partial_prices + predicted_prices

        periods_labels = [
            "Mon AM", "Mon PM", "Tue AM", "Tue PM",
            "Wed AM", "Wed PM", "Thu AM", "Thu PM",
            "Fri AM", "Fri PM", "Sat AM", "Sat PM"
        ]

        print("\n📅 Full week price timeline:")
        for i, label in enumerate(periods_labels):
            if i < len(partial_prices):
                price = partial_prices[i]
                print(f"{label}: {'❓' if price is None else price}")
            else:
                pred_index = i - len(partial_prices)
                print(f"{label}: 🔮 {predicted_prices[pred_index]:.2f}")


if __name__ == "__main__":
    main()