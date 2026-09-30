import requests
import pandas as pd
import joblib


# ============================================================
# SETTINGS
# ============================================================

CITY = "Hyderabad"

MODEL_PATH = "models/location_weather_model.pkl"


# ============================================================
# 1. FIND LOCATION
# ============================================================

print("=" * 60)
print("LOCATION WEATHER PREDICTION")
print("=" * 60)

print(f"\nSearching for: {CITY}")

geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

geocoding_params = {
    "name": CITY,
    "count": 1,
    "language": "en",
    "format": "json"
}

geo_response = requests.get(
    geocoding_url,
    params=geocoding_params,
    timeout=10
)

geo_response.raise_for_status()

geo_data = geo_response.json()

if "results" not in geo_data or not geo_data["results"]:
    print("Location not found.")
    exit()

location = geo_data["results"][0]

latitude = location["latitude"]
longitude = location["longitude"]
city_name = location["name"]
country = location.get("country", "")

print("\nLocation found:")
print("City      :", city_name)
print("Country   :", country)
print("Latitude  :", latitude)
print("Longitude :", longitude)


# ============================================================
# 2. GET WEATHER DATA
# ============================================================

print("\nGetting live weather data...")

weather_url = "https://api.open-meteo.com/v1/forecast"

hourly_variables = (
    "relative_humidity_2m,"
    "dew_point_2m,"
    "precipitation,"
    "cloud_cover,"
    "cloud_cover_low,"
    "cloud_cover_mid,"
    "cloud_cover_high,"
    "pressure_msl,"
    "visibility,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "wind_gusts_10m,"
    "shortwave_radiation,"
    "direct_radiation,"
    "diffuse_radiation,"
    "uv_index"
)

weather_params = {
    "latitude": latitude,
    "longitude": longitude,

    "hourly": hourly_variables,

    # Dataset temperature is treated as Fahrenheit
    "temperature_unit": "fahrenheit",

    # Use km/h to match Open-Meteo's default wind unit.
    "wind_speed_unit": "kmh",

    "forecast_days": 1,

    "timezone": "auto"
}

weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=10
)

weather_response.raise_for_status()

weather_data = weather_response.json()


# ============================================================
# 3. EXTRACT FIRST AVAILABLE HOUR
# ============================================================

hourly = weather_data["hourly"]

weather_df = pd.DataFrame(hourly)

if weather_df.empty:
    print("No weather data received.")
    exit()

row = weather_df.iloc[0]

print("\nWeather data received.")
print("Time:", row["time"])


# ============================================================
# 4. PREPARE ML FEATURES
# ============================================================

# These MUST be in exactly the same order used during training.

features = [
    "clouds",
    "clouds_hi",
    "clouds_low",
    "clouds_mid",
    "dewpt",
    "dhi",
    "dni",
    "ghi",
    "precip",
    "pres",
    "rh",
    "slp",
    "solar_rad",
    "uv",
    "vis",
    "wind_dir",
    "wind_gust_spd",
    "wind_spd"
]


# ------------------------------------------------------------
# Map Open-Meteo variables to our dataset variables
# ------------------------------------------------------------

model_input = {
    "clouds": row["cloud_cover"],
    "clouds_hi": row["cloud_cover_high"],
    "clouds_low": row["cloud_cover_low"],
    "clouds_mid": row["cloud_cover_mid"],

    "dewpt": row["dew_point_2m"],

    "dhi": row["diffuse_radiation"],
    "dni": row["direct_radiation"],

    "ghi": row["shortwave_radiation"],

    "precip": row["precipitation"],

    "pres": row["pressure_msl"],
    "rh": row["relative_humidity_2m"],

    # Sea-level pressure is used for both fields
    "slp": row["pressure_msl"],

    "solar_rad": row["shortwave_radiation"],

    "uv": row["uv_index"],

    "vis": row["visibility"],

    "wind_dir": row["wind_direction_10m"],

    "wind_gust_spd": row["wind_gusts_10m"],

    "wind_spd": row["wind_speed_10m"]
}


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

X_live = pd.DataFrame(
    [[model_input[feature] for feature in features]],
    columns=features
)


print("\nML input features:")
print(X_live.T)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 7. PREDICT TEMPERATURE
# ============================================================

prediction = model.predict(X_live)

predicted_temperature = prediction[0]


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("ML WEATHER PREDICTION RESULT")
print("=" * 60)

print("\nLocation:")
print(f"{city_name}, {country}")

print("\nPredicted Temperature:")
print(f"{predicted_temperature:.2f} °F")

print("\n" + "=" * 60)
print("Prediction completed successfully!")
print("=" * 60)