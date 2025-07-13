import numpy as np
import joblib
import pandas as pd
from utils import prepare_input_for_prediction

def enforce_decreasing(predictions):
    """
    Adjusts the prediction array to ensure it is non-increasing (decreasing or flat).
    If a future value is greater than the previous one, it is set equal to the previous value.
    """
    adjusted = predictions.copy()
    for i in range(1, len(adjusted)):
        if adjusted[i] > adjusted[i - 1]:
            adjusted[i] = adjusted[i - 1]
    return adjusted

def main():
    # Load models
    clf = joblib.load("classifier_model.pkl")
    regressors_by_pattern = joblib.load("regressors_by_pattern.pkl")
    imputer = joblib.load("imputer.pkl")

    # Load historical data with patterns (for typical averages)
    df_history = pd.read_pickle("historical_patterns.pkl")

    # Initial input - example decreasing pattern
    partial_week = [93, 89, 70, 60]  # Mon AM to Tue PM

    # Prepare input for models: impute missing values and shape arrays
    full_week_imputed, reg_input_imputed, missing_count, original_input = prepare_input_for_prediction(
        partial_week, imputer
    )

    # Predict pattern with classifier
    predicted_pattern = clf.predict(full_week_imputed)[0]
    probabilities = clf.predict_proba(full_week_imputed)[0]

    print(f"🔮 Predicted pattern: {predicted_pattern}")
    for pattern, prob in zip(clf.classes_, probabilities):
        print(f" - {pattern}: {prob * 100:.2f}%")

    # Predict future prices with Gaussian Process regressor
    gp = regressors_by_pattern[predicted_pattern]

    mean_pred, std_pred = gp.predict(reg_input_imputed, return_std=True)
    mean_pred = mean_pred[0]
    std_pred = std_pred[0]

    # If pattern is decreasing, enforce non-increasing future predictions
    if predicted_pattern == "decreasing":
        mean_pred = enforce_decreasing(mean_pred)

    # Define full time slots and select only future slots based on input length
    full_slots = [
        "Mon AM", "Mon PM", "Tue AM", "Tue PM",
        "Wed AM", "Wed PM", "Thu AM", "Thu PM",
        "Fri AM", "Fri PM", "Sat AM", "Sat PM"
    ]
    future_slots = full_slots[len(partial_week):]

    print("\n📈 Estimated future prices with uncertainty (mean ± std):")
    for slot, mean, std in zip(future_slots, mean_pred[-len(future_slots):], std_pred[-len(future_slots):]):
        print(f"{slot}: {mean:.2f} ± {std:.2f}")

    # Show typical average historical prices for this pattern, excluding known slots
    matching = df_history[df_history["label"] == predicted_pattern]
    if not matching.empty:
        avg_future = np.mean(matching["future"].tolist(), axis=0)
        typical_future = avg_future[-len(future_slots):]  # only future slots
        print("\n📊 Typical average future prices for this pattern:")
        for slot, price in zip(future_slots, typical_future):
            print(f"{slot}: {price:.2f}")
    else:
        print("\n⚠️ No historical data available for this pattern.")

if __name__ == "__main__":
    main()