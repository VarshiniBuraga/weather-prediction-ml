import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("WEATHER PREDICTION - MODEL VALIDATION")
print("=" * 60)

df = pd.read_csv("data/weather.csv")

print("\nDataset Shape:", df.shape)

# app_temp is intentionally removed for validation
features = [
    "clouds",
    "clouds_hi",
    "clouds_low",
    "clouds_mid",
    "dewpt",
    "dhi",
    "dni",
    "ghi",
    "ozone",
    "pop",
    "precip",
    "pres",
    "rh",
    "slp",
    "snow",
    "snow_depth",
    "solar_rad",
    "uv",
    "vis",
    "wind_dir",
    "wind_gust_spd",
    "wind_spd"
]

target = "temp"

X = df[features]
y = df[target]

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget:", target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            random_state=42,
            max_depth=5
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8
        ))
    ])
}

results = {}

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

results_df = pd.DataFrame(results).T

print("\n" + "=" * 60)
print("MODEL COMPARISON WITHOUT app_temp")
print("=" * 60)

print(results_df)

best_model_name = results_df["RMSE"].idxmin()

best_model = models[best_model_name]

print("\nBest model based on RMSE:")
print(best_model_name)

joblib.dump(
    best_model,
    "models/weather_prediction_model_no_app_temp.pkl"
)

print("\nValidation model saved:")
print("models/weather_prediction_model_no_app_temp.pkl")

results_df.to_csv(
    "models/model_results_no_app_temp.csv"
)

print("Validation results saved:")
print("models/model_results_no_app_temp.csv")

print("\nValidation completed successfully!")