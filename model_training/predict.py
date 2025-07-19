import joblib
import numpy as np

def predict_from_partial(partial_prices, model_folder="model_training", previous_pattern=None):
    """
    Predict the weekly turnip price pattern and future prices based on partial input prices and previous pattern.

    Args:
        partial_prices (list): List of partial prices (None for missing).
        model_folder (str): Path to folder with saved models.
        previous_pattern (str or None): Previous week's pattern as string (optional).

    Returns:
        None. Prints prediction results.
    """

    # Load models and encoder
    clf = joblib.load(f"{model_folder}/classifier_model.pkl")
    imputer = joblib.load(f"{model_folder}/imputer.pkl")
    regressors_by_pattern = joblib.load(f"{model_folder}/regressors_by_pattern.pkl")
    encoder = joblib.load(f"{model_folder}/previous_pattern_encoder.pkl")

    # Ensure input length max 12
    partial_prices = partial_prices[:12]
    # Pad to length 12 with None if needed
    if len(partial_prices) < 12:
        partial_prices += [None] * (12 - len(partial_prices))

    # Convert to numpy array (shape (12,))
    prices_array = np.array(partial_prices, dtype=np.float64).reshape(1, -1)

    # Impute only prices (12 features)
    prices_imputed = imputer.transform(prices_array)

    # Encode previous pattern if provided, else zero vector
    if previous_pattern is None or previous_pattern == "unknown":
        prev_pattern_encoded = np.zeros((1, len(encoder.categories_[0])))
    else:
        prev_pattern_encoded = encoder.transform([[previous_pattern]])

    # Concatenate imputed prices with previous pattern encoding (shape (1, 12 + n))
    X_input = np.hstack([prices_imputed, prev_pattern_encoded])

    # Predict pattern class probabilities and class
    pattern_probs = clf.predict_proba(X_input)[0]
    predicted_pattern = clf.classes_[np.argmax(pattern_probs)]

    print(f"🔮 Predicted pattern: {predicted_pattern}")
    for cls, prob in zip(clf.classes_, pattern_probs):
        print(f" - {cls}: {prob*100:.2f}%")

    # Predict future prices with Gaussian Process regressor for predicted pattern
    gp = regressors_by_pattern.get(predicted_pattern)
    if gp is None:
        print(f"⚠️ No regressor found for pattern '{predicted_pattern}'. Cannot predict future prices.")
        return

    # For GP input, use first 4 prices (Mon AM to Tue PM)
    gp_input = prices_imputed[:, :4]

    mean_pred, std_pred = gp.predict(gp_input, return_std=True)

    # Print predicted future prices (Wed AM to Sat PM)
    days = ["Wed AM", "Wed PM", "Thu AM", "Thu PM", "Fri AM", "Fri PM", "Sat AM", "Sat PM"]
    print("\n📈 Predicted prices for rest of week:")
    for day, mean, std in zip(days, mean_pred.flatten(), std_pred.flatten()):
        print(f"  {day}: {mean:.2f} ± {std:.2f}")