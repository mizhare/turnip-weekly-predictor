import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def parse_dataset(filepath):
    """
    Reads the Excel file containing turnip prices (AM and PM for each day),
    and converts each row into a dictionary with weekdays as keys
    and values as [AM, PM] price lists.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    # Load only weekday columns from Excel
    df = pd.read_excel(filepath, usecols=weekdays)
    dataset = []

    for _, row in df.iterrows():
        week = {}

        for day in weekdays:
            raw_input = str(row.get(day, "")).strip()  # e.g., "60, 90" or empty
            prices = raw_input.replace(',', ' ').split()  # Accept "," or space as separator

            if len(prices) == 2:
                am, pm = int(prices[0]), int(prices[1])
            elif len(prices) == 1:
                am, pm = int(prices[0]), None
            else:
                am, pm = None, None

            week[day] = [am, pm]

        dataset.append(week)

    return dataset

def prepare_dataset(dataset):
    """
    Converts the structured weekly dataset into a 2D list with 12 price slots (Mon AM to Sat PM),
    and imputes missing values using the column mean.
    Returns the imputed dataset and the fitted imputer.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    X = []

    for week in dataset:
        week_flat = []
        for day in days:
            prices = week.get(day, [None, None])
            week_flat.extend(prices)
        X.append(week_flat)

    X_array = np.array(X, dtype=np.float64)  # None becomes np.nan

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X_array)

    return X_imputed, imputer

def pad_week(week):
    """
    Pads the input list to 12 elements (full week) with None if needed.
    """
    padded = list(week)
    while len(padded) < 12:
        padded.append(None)
    return padded

def prepare_input_for_prediction(partial_week, imputer, total_len=12):
    """
    Prepares:
    - full_week_imputed: 12-element input for classification
    - reg_input_imputed: 4-element input for regression
    - missing_count: how many values are missing to complete 12
    - partial_week: original user input (for reference)
    """
    full_week = partial_week + [np.nan] * (total_len - len(partial_week))
    full_week = full_week[:12]
    full_week_imputed = imputer.transform([full_week])

    reg_input = partial_week[:4] + [np.nan] * (4 - len(partial_week[:4]))
    reg_input_imputed = imputer.transform([reg_input + [np.nan] * 8])[:, :4]

    missing_count = total_len - len(partial_week)

    return full_week_imputed, reg_input_imputed, missing_count, partial_week

def augment_week(prices, noise_level=0.15):
    """
    Adds random noise to each price in a given week to simulate variation.
    """
    return [price * (1 + np.random.uniform(-noise_level, noise_level)) for price in prices]

def extract_features_with_peak_info(weeks):
    """
    Extracts extra features from each week:
    - ratio of the peak price to the average of the first 3 prices
    - index of the peak price
    - normalized peak index (between 0 and 1)
    """
    features_list = []
    for week in weeks:
        flat_week = week

        valid_prices = [p for p in flat_week if p is not None]

        if len(valid_prices) == 0:
            features_list.append([0] * len(flat_week) + [0, -1, 0])
            continue

        max_val = max(valid_prices)
        max_idx = flat_week.index(max_val) if max_val in flat_week else -1

        initial_avg = np.mean(flat_week[:3]) if len(flat_week) >= 3 else 1
        ratio = max_val / initial_avg if initial_avg != 0 else 0

        extended_features = flat_week + [ratio, max_idx, max_idx / 11]

        features_list.append(extended_features)
    return features_list