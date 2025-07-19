import joblib
import numpy as np
from pathlib import Path

def count_rises(prices):
    """
    Count the number of price increases in a list of prices, ignoring None values.
    """
    count = 0
    for i in range(1, len(prices)):
        if prices[i] is not None and prices[i-1] is not None:
            if prices[i] > prices[i-1]:
                count += 1
    return count

def predict_from_partial(partial_prices, model_folder, previous_pattern_raw):
    """
    Predict the weekly turnip price pattern and future prices based on partial input prices and previous pattern.

    Args:
        partial_prices (list): List of partial prices (None for missing).
        model_folder (str): Path to folder with saved models.
        previous_pattern_raw (str): Previous week's pattern as string.

    Returns:
        tuple: predicted pattern and list of predicted prices.
    """
    model_folder = Path(model_folder)

    # Load models and encoder
    clf = joblib.load(model_folder / "rf_classifier.joblib")
    imputer = joblib.load(model_folder / "imputer.pkl")
    regressors = joblib.load(model_folder / "regressors.joblib")
    encoder = joblib.load(model_folder / "previous_pattern_encoder.joblib")

    # Ensure input length max 12 and pad with None if needed
    full_week = partial_prices[:12] + [None] * (12 - len(partial_prices))
    prices_array = np.array(full_week).reshape(1, -1)

    # Impute missing price values (12 features)
    prices_imputed = imputer.transform(prices_array)

    # Encode previous pattern or use zero vector if unknown
    if previous_pattern_raw == 'unknown':
        prev_pattern_encoded = np.zeros((1, len(encoder.categories_[0])))
    else:
        prev_pattern_encoded = encoder.transform([[previous_pattern_raw]])

    # Prepare input for classifier: concatenate imputed prices + encoded pattern
    classifier_input = prices_imputed.copy()
    n_available = len(partial_prices)
    classifier_input[0, n_available:] = 0
    clf_input = np.hstack([classifier_input, prev_pattern_encoded])

    # Predict pattern
    predicted_pattern = clf.predict(clf_input)[0]
    print(f"\n🔮 Predicted pattern: {predicted_pattern}")

    # Print prediction probabilities
    probs = clf.predict_proba(clf_input)[0]
    for pattern, prob in zip(clf.classes_, probs):
        print(f" - {pattern}: {prob*100:.2f}%")

    # Predict future prices using Gaussian Process model for the predicted pattern
    gp = regressors.get(predicted_pattern)
    if gp is None:
        print(f"⚠️ No regressor found for pattern '{predicted_pattern}'. Cannot predict prices.")
        return predicted_pattern, []

    # Use first 4 prices (Mon AM to Tue PM) as input to regressor
    gp_input = prices_imputed[:, :4]
    mean_pred, std_pred = gp.predict(gp_input, return_std=True)
    mean_pred = mean_pred.ravel()
    std_pred = std_pred.ravel()

    # Adjust prices for decreasing pattern to ensure smooth downward trend
    if predicted_pattern == "decreasing":
        last_known_price = gp_input[0, -1]
        for i in range(len(mean_pred)):
            if i == 0:
                mean_pred[i] = min(mean_pred[i], last_known_price)
            else:
                mean_pred[i] = min(mean_pred[i], mean_pred[i - 1] * 0.98)

    # Adjust prices for spike patterns to ensure at least 3 rises total
    if predicted_pattern in ["small_spike", "large_spike"]:
        num_rises_input = count_rises(partial_prices)
        min_rises_needed = 3
        remaining_rises_needed = max(0, min_rises_needed - num_rises_input)
        last_known_price = gp_input[0, -1]
        mean_pred_adjusted = []
        rises_so_far = 0

        for i in range(len(mean_pred)):
            pred = mean_pred[i]

            if i == 0:
                # Force first predicted value to be a spike (≥10% rise)
                pred = max(pred, last_known_price * 1.1)
                mean_pred_adjusted.append(pred)
                if pred > last_known_price:
                    rises_so_far += 1
            else:
                prev = mean_pred_adjusted[-1]

                if rises_so_far < remaining_rises_needed:
                    # Force rise between 10%–20%
                    forced_min = prev * 1.10
                    forced_max = prev * 1.20
                    pred = np.clip(pred, forced_min, forced_max)
                    rises_so_far += 1
                else:
                    # After required rises, allow soft decline (max -5%)
                    pred = min(pred, prev * 0.95)

                mean_pred_adjusted.append(pred)

        mean_pred = np.array(mean_pred_adjusted)

    # Print predicted prices (Wed AM to Sat PM)
    prediction_days_periods = [
        ("Wednesday", "AM"), ("Wednesday", "PM"),
        ("Thursday", "AM"), ("Thursday", "PM"),
        ("Friday", "AM"), ("Friday", "PM"),
        ("Saturday", "AM"), ("Saturday", "PM")
    ]

    print("\n📈 Predicted prices for rest of week:")
    for i, (day, period) in enumerate(prediction_days_periods):
        mean = mean_pred[i]
        std = std_pred[i]
        print(f"  {day} {period}: {mean:.2f} ± {std:.2f}")

    return predicted_pattern, mean_pred.tolist()
