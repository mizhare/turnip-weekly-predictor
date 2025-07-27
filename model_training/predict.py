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
        tuple: predicted pattern (str), list of predicted prices (list of floats),
               dict of pattern probabilities (pattern -> float),
               list of tuples (mean, std) for confidence intervals.
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

    # Prediction probabilities
    probs = clf.predict_proba(clf_input)[0]

    # Predict future prices using Gaussian Process model for the predicted pattern
    gp = regressors.get(predicted_pattern)
    if gp is None:
        return predicted_pattern, [], dict(zip(clf.classes_, probs)), []

    # Use first 4 prices (Mon AM to Tue PM) as input to regressor
    gp_input = prices_imputed[:, :4]
    mean_pred, std_pred = gp.predict(gp_input, return_std=True)
    mean_pred = mean_pred.ravel()
    std_pred = std_pred.ravel()

    # Adjust prices for decreasing pattern
    if predicted_pattern == "decreasing":
        last_known_price = gp_input[0, -1]
        for i in range(len(mean_pred)):
            if i == 0:
                mean_pred[i] = min(mean_pred[i], last_known_price)
            else:
                mean_pred[i] = min(mean_pred[i], mean_pred[i - 1] * 0.98)

    # Adjust prices for spike patterns
    if predicted_pattern in ["small_spike", "large_spike"]:
        # Force a spike mid-week
        base_price = gp_input[0, -1]
        peak_multiplier = 2.0 if predicted_pattern == "large_spike" else 1.5

        peak_pos = 4  # Wednesday PM
        mean_pred_adjusted = []

        for i in range(len(mean_pred)):
            if i == peak_pos:
                pred = base_price * peak_multiplier
            elif i < peak_pos:
                pred = base_price * (1 + 0.05 * i)
            else:
                pred = base_price * max(0.4, 1 - 0.2 * (i - peak_pos))

            mean_pred_adjusted.append(pred)

        mean_pred = np.array(mean_pred_adjusted)
        
    prob_dict = dict(zip(clf.classes_, probs))
    ci_list = list(zip(mean_pred, std_pred))

    return predicted_pattern, mean_pred.tolist(), prob_dict, ci_list
