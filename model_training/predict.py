import numpy as np
import joblib
import pandas as pd
from utils import prepare_input_for_prediction

def enforce_decreasing(predictions):
    """
    Adjust the predicted prices to ensure non-increasing sequence.
    """
    adjusted = predictions.copy()
    for i in range(1, len(adjusted)):
        if adjusted[i] > adjusted[i - 1]:
            adjusted[i] = adjusted[i - 1]
    return adjusted

def predict_from_partial(partial_week, model_folder):
    """
    Given partial prices (list), load models and predict pattern and future prices.
    """
    # Load models
    clf = joblib.load(f"{model_folder}/classifier_model.pkl")
    regressors_by_pattern = joblib.load(f"{model_folder}/regressors_by_pattern.pkl")
    imputer = joblib.load(f"{model_folder}/imputer.pkl")

    # Load historical data (optional)
    try:
        df_history = pd.read_pickle(f"{model_folder}/historical_patterns.pkl")
    except FileNotFoundError:
        df_history = None

    # Prepare input for classifier and regressor
    full_week_imputed, reg_input_imputed, missing_count, original_input = prepare_input_for_prediction(partial_week, imputer)

    # Predict pattern
    predicted_pattern = clf.predict(full_week_imputed)[0]
    probabilities = clf.predict_proba(full_week_imputed)[0]

    print(f"🔮 Predicted pattern: {predicted_pattern}")
    for pattern, prob in zip(clf.classes_, probabilities):
        print(f" - {pattern}: {prob * 100:.2f}%")

    # Predict future prices using Gaussian Process regressor for predicted pattern
    gp = regressors_by_pattern.get(predicted_pattern)
    if gp is None:
        print(f"⚠️ No regressor model found for pattern '{predicted_pattern}'. Cannot predict future prices.")
        return

    mean_pred, std_pred = gp.predict(reg_input_imputed, return_std=True)
    mean_pred = mean_pred[0]
    std_pred = std_pred[0]

    # If decreasing pattern, enforce non-increasing prediction
    if predicted_pattern == "decreasing":
        mean_pred = enforce_decreasing(mean_pred)

    # Define all time slots and select future ones based on length of partial_week
    full_slots = [
        "Mon AM", "Mon PM", "Tue AM", "Tue PM",
        "Wed AM", "Wed PM", "Thu AM", "Thu PM",
        "Fri AM", "Fri PM", "Sat AM", "Sat PM"
    ]
    future_slots = full_slots[len(partial_week):]

    print("\n📈 Estimated future prices with uncertainty (mean ± std):")
    for slot, mean, std in zip(future_slots, mean_pred[-len(future_slots):], std_pred[-len(future_slots):]):
        print(f"{slot}: {mean:.2f} ± {std:.2f}")

    # Show typical average future prices from historical data if available
    if df_history is not None:
        matching = df_history[df_history["label"] == predicted_pattern]
        if not matching.empty:
            avg_future = np.mean(matching["future"].tolist(), axis=0)
            typical_future = avg_future[-len(future_slots):]
            print("\n📊 Typical average future prices for this pattern:")
            for slot, price in zip(future_slots, typical_future):
                print(f"{slot}: {price:.2f}")
        else:
            print("\n⚠️ No historical data available for this pattern.")
    else:
        print("\n⚠️ Historical data file not found.")