import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

MODEL_PATH = "models/weather_prediction_model.pkl"


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


# --------------------------------------------------
# GET LOCATION
# --------------------------------------------------

def get_location(city):

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        GEOCODING_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        raise ValueError(
            f"Location not found: {city}"
        )

    result = data["results"][0]

    return {
        "name": result["name"],
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "country": result.get("country", "")
    }


# --------------------------------------------------
# GET CURRENT WEATHER
# --------------------------------------------------

def get_weather(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ]),

        "hourly": ",".join([
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
        ]),

        "forecast_days": 2,

        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# PREPARE MODEL INPUT
# --------------------------------------------------

def prepare_features(weather_data):

    hourly = weather_data["hourly"]

    weather_df = pd.DataFrame(hourly)

    weather_df["time"] = pd.to_datetime(
        weather_df["time"]
    )

    # Find current/latest available hour
    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    ).tz_localize(None)

    weather_df["time_difference"] = abs(
        weather_df["time"] - now
    )

    current_index = (
        weather_df["time_difference"]
        .idxmin()
    )

    row = weather_df.loc[current_index].copy()

    current_time = row["time"]

    # Time features
    hour = current_time.hour
    day_of_week = current_time.dayofweek
    month = current_time.month
    day_of_year = current_time.dayofyear

    # Cyclic time features
    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * day_of_year / 365
    )

    day_cos = np.cos(
        2 * np.pi * day_of_year / 365
    )

    features = {
        "temperature_2m": row["temperature_2m"],
        "relative_humidity_2m": row["relative_humidity_2m"],
        "dew_point_2m": row["dew_point_2m"],
        "precipitation": row["precipitation"],
        "cloud_cover": row["cloud_cover"],
        "cloud_cover_low": row["cloud_cover_low"],
        "cloud_cover_mid": row["cloud_cover_mid"],
        "cloud_cover_high": row["cloud_cover_high"],
        "pressure_msl": row["pressure_msl"],
        "wind_speed_10m": row["wind_speed_10m"],
        "wind_direction_10m": row["wind_direction_10m"],
        "wind_gusts_10m": row["wind_gusts_10m"],
        "shortwave_radiation": row["shortwave_radiation"],
        "direct_radiation": row["direct_radiation"],
        "diffuse_radiation": row["diffuse_radiation"],
        "latitude": weather_data["latitude"],
        "longitude": weather_data["longitude"],
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "day_of_year": day_of_year,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
    }

    return pd.DataFrame(
        [features],
        columns=FEATURES
    ), row, current_time


# --------------------------------------------------
# MAIN PREDICTION
# --------------------------------------------------

def predict_weather(city):

    print("\nSearching location...")

    location = get_location(city)

    print(
        f"Location: {location['name']}, "
        f"{location['country']}"
    )

    print(
        f"Coordinates: "
        f"{location['latitude']}, "
        f"{location['longitude']}"
    )

    print("\nFetching live weather...")

    weather_data = get_weather(
        location["latitude"],
        location["longitude"]
    )

    current_temperature = (
        weather_data["current"]["temperature_2m"]
    )

    print(
        f"Current temperature: "
        f"{current_temperature:.1f} °C"
    )

    X, current_row, current_time = (
        prepare_features(weather_data)
    )

    print(
        f"Weather time used: "
        f"{current_time}"
    )

    print("\nLoading ML model...")

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(X)[0]

    print(
        f"\nPredicted next-hour temperature: "
        f"{prediction:.1f} °C"
    )

    print("\nPrediction completed successfully!")

    return prediction


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    city = input(
        "\nEnter city/location: "
    ).strip()

    if not city:
        print("Please enter a location.")
    else:
        try:
            predict_weather(city)

        except Exception as e:
            print(
                f"\nError: {e}"
            )