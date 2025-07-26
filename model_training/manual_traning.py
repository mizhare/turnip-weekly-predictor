import random
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import joblib
import numpy as np
from pathlib import Path

from utils import parse_dataset, prepare_dataset, augment_week
from manual_pattern import detect_pattern


def flatten_week(week_dict, days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], periods_per_day=2):
    """Flatten a dictionary week format into a flat list of prices ordered by day and AM/PM."""
    flat = []
    for day in days:
        flat.extend(week_dict[day])
    return flat


def generate_decreasing_week(base_week, noise_level=0.05):
    """
    Create a synthetic decreasing version from a base week.
    Applies slight noise while maintaining decreasing order.
    """
    flat_week = flatten_week(base_week)

    # Ensure prices decrease smoothly
    for i in range(1, len(flat_week)):
        # Make sure price[i] < price[i-1] with a small random gap
        flat_week[i] = min(flat_week[i], flat_week[i-1] - random.uniform(0.5, 3.0))

    # Apply light noise to avoid perfectly linear sequence
    noisy_week = augment_week(flat_week, noise_level=noise_level)

    # Fix to guarantee decreasing order after noise
    for i in range(1, len(noisy_week)):
        noisy_week[i] = min(noisy_week[i], noisy_week[i-1] - 0.1)

    return noisy_week


def main():
    filepath = "turnips_prices.xlsx"
    dataset = parse_dataset(filepath)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    periods_per_day = 2  # AM and PM

    print("📊 Original dataset loaded. Augmenting data...")

    # Start with original dataset
    augmented_dataset = dataset.copy()

    # Add 3 noisy versions per week for data augmentation (general)
    for week in dataset:
        flat_week = flatten_week(week)

        for _ in range(3):
            noisy_flat = augment_week(flat_week, noise_level=0.05)

            # Convert flat noisy week back to dictionary format
            noisy_week_dict = {}
            for i, day in enumerate(days):
                noisy_week_dict[day] = noisy_flat[i * periods_per_day : (i + 1) * periods_per_day]

            augmented_dataset.append(noisy_week_dict)

    # Generate more synthetic decreasing weeks to balance the dataset
    decreasing_weeks = [week for week in dataset if detect_pattern(flatten_week(week)) == "decreasing"]

    for base_week in decreasing_weeks:
        for _ in range(10):  # generate 10 artificial versions per original decreasing week
            dec_flat = generate_decreasing_week(base_week)
            dec_week_dict = {}
            for i, day in enumerate(days):
                dec_week_dict[day] = dec_flat[i * periods_per_day : (i + 1) * periods_per_day]
            augmented_dataset.append(dec_week_dict)

    print(f"✅ Data augmentation complete. Total weeks: {len(augmented_dataset)}")

    # Convert dataset to feature matrix and get imputer
    X, imputer = prepare_dataset(augmented_dataset)

    # Label each week with its turnip price pattern
    labels = [detect_pattern(list(prices)) for prices in X]

    # Assign previous week's pattern
    previous_patterns = []
    for i in range(len(augmented_dataset)):
        if i == 0:
            previous_patterns.append("unknown")
        else:
            previous_patterns.append(labels[i - 1])

    # Filter out unknown patterns for training
    X_filtered = []
    y_filtered = []
    prev_patterns_filtered = []

    for features, label, prev_pattern in zip(X, labels, previous_patterns):
        if label != "unknown":
            X_filtered.append(features)
            y_filtered.append(label)
            prev_patterns_filtered.append(prev_pattern)

    X_filtered = np.array(X_filtered)

    # One-hot encode previous pattern
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    prev_pattern_encoded = encoder.fit_transform(np.array(prev_patterns_filtered).reshape(-1, 1))

    # Combine prices and encoded previous pattern into one input array
    X_final = np.hstack([X_filtered, prev_pattern_encoded])

    print("🧠 Training pattern classifier (Random Forest)...")

    # Train classifier
    clf = RandomForestClassifier(random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X_final, y_filtered)

    # Show classification report on training set
    print("\n📋 Classification Report (training set):")
    y_pred_train = clf.predict(X_final)
    print(classification_report(y_filtered, y_pred_train, target_names=clf.classes_))

    # Plot the first decision tree for visualization
    feature_names = [f"{day}_{period}" for day in days for period in ["AM", "PM"]]
    feature_names += list(encoder.categories_[0])  # previous pattern categories

    plt.figure(figsize=(50, 25))
    plot_tree(
        clf.estimators_[0],
        feature_names=feature_names,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        max_depth=5,
        fontsize=5,
    )
    plt.show()

    print("🌱 Training regressors for each pattern (Gaussian Process)...")

    # Prepare GP regressors for each pattern
    kernel = C(1.0, (1e-4, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    regressors_by_pattern = {}

    max_input_len = 4   # Mon AM to Tue PM
    output_len = 8      # Wed AM to Sat PM

    for pattern in set(y_filtered):
        X_pat = []
        y_pat = []

        for features, label in zip(X_filtered, y_filtered):
            if label == pattern:
                x_in = features[:max_input_len]
                y_out = features[max_input_len : max_input_len + output_len]

                # Ensure proper shape by padding if needed
                if len(x_in) < max_input_len:
                    x_in = list(x_in) + [np.nan] * (max_input_len - len(x_in))
                if len(y_out) < output_len:
                    y_out = list(y_out) + [np.nan] * (output_len - len(y_out))

                X_pat.append(x_in)
                y_pat.append(y_out)

        X_pat = np.array(X_pat)
        y_pat = np.array(y_pat)

        if len(X_pat) == 0 or len(y_pat) == 0:
            print(f"⚠️ No training data for pattern '{pattern}'. Skipping.")
            continue

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-1,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        )
        gp.fit(X_pat, y_pat)
        regressors_by_pattern[pattern] = gp
        print(f"✅ Regressor trained for pattern '{pattern}'")

    # Save all models and encoders
    joblib.dump(clf, "rf_classifier.joblib")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(regressors_by_pattern, "regressors.joblib")
    joblib.dump(encoder, "previous_pattern_encoder.joblib")

    print("\n✅ All models saved successfully!")


if __name__ == "__main__":
    main()