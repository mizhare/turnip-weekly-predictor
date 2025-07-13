import random

def detect_pattern(prices):
    prices_valid = [p for p in prices if p is not None]
    if len(prices_valid) < 4:
        return "unknown"

    # Use only the first 4 values to detect a decreasing pattern
    initial_segment = prices_valid[:4]

    # Check for decreasing pairs in the initial segment
    pairs = list(zip(initial_segment, initial_segment[1:]))
    decreasing_pairs = sum(1 for earlier, later in pairs if earlier > later)

    # If all 3 pairs are decreasing, classify as decreasing
    if decreasing_pairs == 3:
        return "decreasing"

    # If not decreasing, evaluate for spikes using the full week
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