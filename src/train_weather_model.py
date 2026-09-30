import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/processed_weather.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("WEATHER TEMPERATURE PREDICTION - MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"\nTotal records: {len(df)}")


# ============================================================
# Features
# ============================================================

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

TARGET = "target_temperature"


# ============================================================
# Remove rows with missing target
# ============================================================

df = df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# Chronological train/test split
# ============================================================

# Sort globally by time
df = df.sort_values("time").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"\nTraining records: {len(train_df)}")
print(f"Testing records:  {len(test_df)}")

print(
    f"\nTraining period:"
    f"\n{train_df['time'].min()}"
    f" → {train_df['time'].max()}"
)

print(
    f"\nTesting period:"
    f"\n{test_df['time'].min()}"
    f" → {test_df['time'].max()}"
)


# ============================================================
# Models
# ============================================================

models = {

    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            DecisionTreeRegressor(
                max_depth=12,
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        )
    ])
}


# ============================================================
# Train and evaluate
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE:  {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"R²:   {r2:.4f}")


# ============================================================
# Results table
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Select best model
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print(
    f"\nBest model based on lowest RMSE: "
    f"{best_model_name}"
)


# ============================================================
# Save best model
# ============================================================

model_path = MODEL_DIR / "weather_prediction_model.pkl"

joblib.dump(
    best_model,
    model_path
)

# Save results
results_path = MODEL_DIR / "weather_model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


# Save feature list
features_path = MODEL_DIR / "weather_model_features.txt"

with open(
    features_path,
    "w"
) as file:

    for feature in FEATURES:
        file.write(
            feature + "\n"
        )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"\nBest model:")
print(best_model_name)

print(f"\nModel saved to:")
print(model_path)

print(f"\nResults saved to:")
print(results_path)

print(f"\nFeature list saved to:")
print(features_path)