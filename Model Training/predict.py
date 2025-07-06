import numpy as np
import joblib
from utils import prepare_input_for_prediction

def main():
    # Loading models
    clf = joblib.load('classifier_model.pkl')
    imputer = joblib.load('imputer.pkl')
    regressors_by_pattern = joblib.load('regressors_by_pattern.pkl')

    # Initial (input)
    partial_week = [114, 85, 92, 87, 65, 60]

    # Preparing data for the models
    full_week_imputed, reg_input_imputed, missing_count, original_input = prepare_input_for_prediction(
        partial_week, imputer
    )

    predicted_pattern = clf.predict(full_week_imputed)[0]
    probabilities = clf.predict_proba(full_week_imputed)[0]

    print("🔮 Predicted pattern:", predicted_pattern)
    for pattern, prob in zip(clf.classes_, probabilities):
        print(f" - {pattern}: {prob * 100:.2f}%")

    # Selecting the best regressor for the pattern
    reg = regressors_by_pattern[predicted_pattern]

    # Using all trees
    all_preds = [tree.predict(reg_input_imputed)[0] for tree in reg.estimators_]
    all_preds = np.array(all_preds)

    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    # Predict the future slots within the base slots
    full_slots = ["Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM", "Wed PM",
                  "Thu AM", "Thu PM", "Fri AM", "Fri PM", "Sat AM", "Sat PM"]

    future_slots = full_slots[len(partial_week):]

    print("\n📈 Estimated future prices (mean ± std):")
    for slot, mean, std in zip(future_slots, mean_pred[-len(future_slots):], std_pred[-len(future_slots):]):
        print(f"{slot}: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    main()

