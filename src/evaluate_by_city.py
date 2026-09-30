import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load data
df = pd.read_csv("data/processed_weather.csv")
df["time"] = pd.to_datetime(df["time"])

# Same features used during training
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

# Sort exactly as during training
df = df.sort_values("time").reset_index(drop=True)

# Same 80/20 chronological split
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

# Load trained model
model = joblib.load(
    "models/weather_prediction_model.pkl"
)

print("=" * 70)
print("RANDOM FOREST - CITY-BY-CITY EVALUATION")
print("=" * 70)

results = []

for city in sorted(test_df["city"].unique()):

    city_df = test_df[
        test_df["city"] == city
    ]

    X = city_df[FEATURES]
    y = city_df[TARGET]

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    results.append({
        "City": city,
        "Records": len(city_df),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format
        }
    )
)

print("\n" + "=" * 70)
print("AVERAGE CITY RESULTS")
print("=" * 70)

print(
    f"Average MAE:  {results_df['MAE'].mean():.4f} °C"
)

print(
    f"Average RMSE: {results_df['RMSE'].mean():.4f} °C"
)

print(
    f"Average R²:   {results_df['R2'].mean():.4f}"
)

# Save results
results_df.to_csv(
    "models/city_evaluation_results.csv",
    index=False
)

print("\nSaved to:")
print("models/city_evaluation_results.csv")