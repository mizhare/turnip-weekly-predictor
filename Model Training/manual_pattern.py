import random

def detect_pattern(prices):
    prices_valid = [p for p in prices if p is not None]

    if len(prices_valid) < 4:
        return "unknown"

    # Use the average of the first 3 prices to avoid early noise
    initial_values = prices_valid[:3]
    initial_avg = sum(initial_values) / len(initial_values)

    max_val = max(prices_valid)
    try:
        max_idx = prices.index(max_val)
    except ValueError:
        max_idx = -1

    # Compute the base ratio
    ratio = max_val / initial_avg if initial_avg != 0 else 0

    # Add a small fuzz to introduce variability in pattern assignment
    fuzz = random.uniform(-0.1, 0.1)
    adjusted_ratio = ratio + fuzz

    # Check if prices are strictly decreasing
    decreasing = True
    for earlier, later in zip(prices_valid, prices_valid[1:]):
        if earlier <= later:
            decreasing = False
            break
    if decreasing:
        return "decreasing"

    # Large spike: very high peak between Wed AM and Fri PM
    if adjusted_ratio >= 3 and 6 <= max_idx <= 10:
        return "large_spike"

    # Small spike: moderate peak between Tue AM and Fri PM
    if 1.5 <= adjusted_ratio < 3 and 4 <= max_idx <= 10:
        return "small_spike"

    # Default case: not clearly a spike or decreasing
    return "fluctuating"


