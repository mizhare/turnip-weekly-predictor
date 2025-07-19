import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

from utils import parse_dataset, prepare_dataset, augment_week
from manual_pattern import detect_pattern

def flatten_week(week_dict, days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], periods_per_day=2):
    """Flatten the weekly dict of prices into a list ordered by days and periods (AM, PM)."""
    flat = []
    for day in days:
        flat.extend(week_dict[day])
    return flat

def main():
    filepath = "turnips_prices.xlsx"
    dataset = parse_dataset(filepath)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    periods_per_day = 2  # AM and PM

    # Start with original dataset
    augmented_dataset = dataset.copy()

    # Add 3 noisy augmented versions of each week (same dict format)
    for week in dataset:
        # Flatten week into list of prices (length 12)
        flat_week = flatten_week(week)

        for _ in range(3):
            noisy_flat = augment_week(flat_week, noise_level=0.05)

            # Rebuild dictionary for noisy week
            noisy_week_dict = {}
            for i, day in enumerate(days):
                noisy_week_dict[day] = noisy_flat[i*periods_per_day : (i+1)*periods_per_day]

            augmented_dataset.append(noisy_week_dict)

    # Prepare data for training (X matrix and imputer)
    X, imputer = prepare_dataset(augmented_dataset)

    # Label each week pattern (list of 12 prices per sample)
    labels = [detect_pattern(list(prices)) for prices in X]

    # Add previous pattern for each week (except first)
    previous_patterns = []
    for i in range(len(augmented_dataset)):
        if i == 0:
            previous_patterns.append("unknown")
        else:
            previous_patterns.append(labels[i-1])

    # Filter out unknown patterns
    X_filtered = []
    y_filtered = []
    prev_patterns_filtered = []

    for features, label, prev_pattern in zip(X, labels, previous_patterns):
        if label != "unknown":
            X_filtered.append(features)
            y_filtered.append(label)
            prev_patterns_filtered.append(prev_pattern)

    X_filtered = np.array(X_filtered)

    # One-hot encode previous week's pattern
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    prev_pattern_encoded = encoder.fit_transform(np.array(prev_patterns_filtered).reshape(-1, 1))

    # Concatenate price features with previous pattern encoding
    X_final = np.hstack([X_filtered, prev_pattern_encoded])

    # Train Random Forest classifier to predict pattern
    clf = RandomForestClassifier(random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X_final, y_filtered)

    # Plot first tree for visualization
    feature_names = [f"{day}_{period}" for day in days for period in ["AM", "PM"]]
    feature_names += list(encoder.categories_[0])  # previous pattern categories

    plot_tree(
        clf.estimators_[0],
        feature_names=feature_names,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        max_depth=5,
    )
    plt.show()

    # Train Gaussian Process regressors per pattern
    regressors_by_pattern = {}

    kernel = C(1.0, (1e-4, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    for pattern in set(y_filtered):
        X_pat = []
        y_pat = []

        for features, label in zip(X_final, y_filtered):
            if label == pattern:
                # Use first 4 prices (Mon AM to Tue PM) as input — exclude previous pattern features here
                X_pat.append(features[:4])
                # Use rest as output (Wed AM to Sat PM)
                y_pat.append(features[4:12])  # only price features, ignore previous pattern encoding here

        if len(X_pat) == 0 or len(y_pat) == 0:
            print(f"⚠️ No training data for pattern {pattern}. Skipping regressor.")
            continue

        X_pat = np.array(X_pat)
        y_pat = np.array(y_pat)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-1,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        )
        gp.fit(X_pat, y_pat)
        regressors_by_pattern[pattern] = gp

    # Save models, imputer and encoder
    joblib.dump(clf, "classifier_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(regressors_by_pattern, "regressors_by_pattern.pkl")
    joblib.dump(encoder, "previous_pattern_encoder.pkl")

    print("✅ Models trained and saved successfully!")

if __name__ == "__main__":
    main()