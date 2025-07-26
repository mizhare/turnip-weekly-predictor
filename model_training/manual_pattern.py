import random

def detect_pattern(prices):
    prices_valid = [p for p in prices if p is not None]
    if len(prices_valid) < 4:
        return "unknown"

    # Check for decreasing pattern across all available values
    if all(earlier > later for earlier, later in zip(prices_valid, prices_valid[1:])):
        return "decreasing"

    # Spike detection logic as before
    initial_avg = sum(prices_valid[:3]) / 3
    max_val = max(prices_valid)
    max_idx = prices_valid.index(max_val)

    ratio = max_val / initial_avg if initial_avg != 0 else 0
    fuzz = random.uniform(-0.1, 0.1)
    adjusted_ratio = ratio + fuzz

    if adjusted_ratio >= 3 and 6 <= max_idx <= 10:
        return "large_spike"
    if 1.5 <= adjusted_ratio < 3 and 4 <= max_idx <= 10:
        return "small_spike"

    return "fluctuating"


"""if __name__ == "__main__":
    print("=== Pattern Detection Test ===")

    # Labels for days/times to guide user input
    time_labels = ["Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM", "Wed PM", "Thu AM", "Thu PM", "Fri AM", "Fri PM",
                   "Sat AM", "Sat PM"]

    # Collect input prices
    prices = []
    print("Enter your turnip prices for each time period. Press Enter to skip a time slot:")
    for label in time_labels:
        raw = input(f"{label}: ").strip()
        if raw == "":
            prices.append(None)
        else:
            try:
                price = float(raw)
                prices.append(price)
            except ValueError:
                print("Invalid input. Using None.")
                prices.append(None)

    # Detect pattern
    result = detect_pattern(prices)

    # Output
    print("\n🔍 Detected Pattern:", result)"""