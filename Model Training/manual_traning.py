import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
import pandas as pd

from utils import parse_dataset, prepare_dataset, augment_week
from manual_pattern import detect_pattern

def main():
    # 1. Load and prepare dataset
    filepath = "turnips_prices.xlsx"
    dataset = parse_dataset(filepath)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    periods_per_day = 2  # AM and PM

    # Initialize augmented dataset with original weeks
    augmented_dataset = dataset.copy()

    # Add 3 noisy versions of each week, preserving dictionary structure
    for week in dataset:
        # Flatten original week into list of prices (length 12)
        flat_week = sum([week[day] for day in days], [])

        for _ in range(3):  # Number of augmentations
            noisy_flat = augment_week(flat_week, noise_level=0.05)

            # Reconstruct dictionary for noisy week
            noisy_week_dict = {}
            for i, day in enumerate(days):
                noisy_week_dict[day] = noisy_flat[i*periods_per_day : (i+1)*periods_per_day]

            augmented_dataset.append(noisy_week_dict)

    # Prepare dataset (feature matrix X and imputer)
    X, imputer = prepare_dataset(augmented_dataset)

    # 3. Label each week pattern using detect_pattern
    labels = [detect_pattern(list(prices)) for prices in X]

    # Filter samples with valid labels (exclude 'unknown')
    X_filtered = []
    y_filtered = []
    for features, label in zip(X, labels):
        if label != "unknown":
            X_filtered.append(features)
            y_filtered.append(label)

    # Train Random Forest classifier to identify patterns
    clf = RandomForestClassifier(random_state=42, max_depth=5, class_weight='balanced')
    clf.fit(X_filtered, y_filtered)

    # Visualize decision tree
    plot_tree(
        clf.estimators_[0],
        feature_names=[f"{day}_{period}" for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] for period in ["AM", "PM"]],
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        max_depth=5,
    )
    plt.show()

    # Train separate Gaussian Process regressors per pattern
    regressors_by_pattern = {}
    kernel = C(1.0, (1e-4, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    for pattern in set(y_filtered):
        X_pat = []
        y_pat = []
        for features, label in zip(X_filtered, y_filtered):
            if label == pattern:
                # Use Mon AM to Tue PM as input
                X_pat.append(features[:4])
                # Use Wed AM to Sat PM as output (target)
                y_pat.append(features[4:])

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-1,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        )
        gp.fit(X_pat, y_pat)
        regressors_by_pattern[pattern] = gp

    # Save models and imputer for later prediction
    joblib.dump(clf, "classifier_model.pkl")
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(regressors_by_pattern, "regressors_by_pattern.pkl")

    print("✅ Models trained and saved successfully!")

if __name__ == "__main__":
    main()