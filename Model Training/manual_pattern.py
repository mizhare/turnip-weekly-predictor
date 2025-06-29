def detect_pattern(prices):
    prices_valid = [p for p in prices if p is not None]

    if len(prices_valid) < 4:
        return "unknown"

    initial_values = prices_valid[:3]   # Prevent the ratio inflexibility, prices more constant
    initial_avg = sum(initial_values) / len(initial_values)

    max_val = max(prices_valid)
    try:
        max_idx = prices.index(max_val)
    except ValueError:
        max_idx = -1

    if initial_avg != 0:   # Our average ratio that will guide our patterns
        ratio = max_val / initial_avg
    else:
        ratio = 0

    decreasing = True   # Asking if the prices are continuosly decreasing
    for earlier, later in zip(prices_valid, prices_valid[1:]):
        if earlier <= later:
            decreasing = False
            break
    if decreasing:
        return "decreasing"

    if ratio >= 3 and 6 <= max_idx <= 10:   # Check for ratio >= 3 and peak between Wed AM and Fri PM
        return "large_spike"

    if 1.5 <= ratio < 3 and 4 <= max_idx <= 10:   # Check for ratio between 1.5 and 3 and peak between Tue AM and Fri PM
        return "small_spike"

    return "fluctuating"   #  In case there´s no clear pattern in decrease and increase


