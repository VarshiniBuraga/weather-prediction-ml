import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ML-Based Weather Prediction System",
    page_icon="🌤️",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🌤️ ML-Based Weather Prediction System")

st.write(
    "Predict the next-hour temperature using "
    "Machine Learning and live weather data."
)


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/weather_prediction_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load ML model: {e}")
    st.stop()


# ==================================================
# LOCATION FUNCTION
# ==================================================

def get_location(city):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if "results" not in data:
        return None

    if len(data["results"]) == 0:
        return None

    return data["results"][0]


# ==================================================
# WEATHER FUNCTION
# ==================================================

def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    current_variables = (
        "temperature_2m,"
        "relative_humidity_2m,"
        "dew_point_2m,"
        "precipitation,"
        "cloud_cover,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m"
    )

    hourly_variables = (
        "temperature_2m,"
        "relative_humidity_2m,"
        "dew_point_2m,"
        "precipitation,"
        "precipitation_probability,"
        "cloud_cover,"
        "cloud_cover_low,"
        "cloud_cover_mid,"
        "cloud_cover_high,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m,"
        "shortwave_radiation,"
        "direct_radiation,"
        "diffuse_radiation"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": current_variables,
        "hourly": hourly_variables,
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ==================================================
# PREPARE FEATURES
# ==================================================

def prepare_features(weather_data):

    hourly = weather_data["hourly"]

    hourly_df = pd.DataFrame(hourly)

    hourly_df["time"] = pd.to_datetime(
        hourly_df["time"]
    )

    now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    now_naive = pd.Timestamp(
        now.replace(tzinfo=None)
    )

    hourly_df["time_difference"] = abs(
        hourly_df["time"] - now_naive
    )

    current_index = hourly_df[
        "time_difference"
    ].idxmin()

    row = hourly_df.loc[current_index]

    timestamp = row["time"]

    # ----------------------------------------------
    # TIME FEATURES
    # ----------------------------------------------

    hour = timestamp.hour

    day_of_week = timestamp.dayofweek

    month = timestamp.month

    day_of_year = timestamp.dayofyear

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

    # ----------------------------------------------
    # MODEL FEATURES
    # ----------------------------------------------

    features = pd.DataFrame([{

        "temperature_2m":
            row["temperature_2m"],

        "relative_humidity_2m":
            row["relative_humidity_2m"],

        "dew_point_2m":
            row["dew_point_2m"],

        "precipitation":
            row["precipitation"],

        "cloud_cover":
            row["cloud_cover"],

        "cloud_cover_low":
            row["cloud_cover_low"],

        "cloud_cover_mid":
            row["cloud_cover_mid"],

        "cloud_cover_high":
            row["cloud_cover_high"],

        "pressure_msl":
            row["pressure_msl"],

        "wind_speed_10m":
            row["wind_speed_10m"],

        "wind_direction_10m":
            row["wind_direction_10m"],

        "wind_gusts_10m":
            row["wind_gusts_10m"],

        "shortwave_radiation":
            row["shortwave_radiation"],

        "direct_radiation":
            row["direct_radiation"],

        "diffuse_radiation":
            row["diffuse_radiation"],

        "latitude":
            weather_data["latitude"],

        "longitude":
            weather_data["longitude"],

        "hour":
            hour,

        "day_of_week":
            day_of_week,

        "month":
            month,

        "day_of_year":
            day_of_year,

        "hour_sin":
            hour_sin,

        "hour_cos":
            hour_cos,

        "day_sin":
            day_sin,

        "day_cos":
            day_cos

    }])

    return features, timestamp, hourly_df


# ==================================================
# USER INPUT
# ==================================================

city = st.text_input(
    "📍 Enter city/location",
    placeholder="Example: Ongole"
)

predict_button = st.button(
    "🔮 Predict Weather",
    type="primary"
)


# ==================================================
# MAIN APPLICATION
# ==================================================

if predict_button:

    if city.strip() == "":

        st.warning(
            "Please enter a city or location."
        )

    else:

        try:

            # ==========================================
            # LOCATION
            # ==========================================

            with st.spinner(
                "Searching location..."
            ):

                location = get_location(city)

            if location is None:

                st.error(
                    "Location not found. "
                    "Please enter a valid city name."
                )

                st.stop()

            latitude = location["latitude"]

            longitude = location["longitude"]

            location_name = location["name"]

            country = location.get(
                "country",
                ""
            )


            # ==========================================
            # WEATHER
            # ==========================================

            with st.spinner(
                "Fetching live weather..."
            ):

                weather_data = get_weather(
                    latitude,
                    longitude
                )


            # ==========================================
            # PREPARE FEATURES
            # ==========================================

            features, weather_time, hourly_df = (
                prepare_features(weather_data)
            )


            # ==========================================
            # ML PREDICTION
            # ==========================================

            prediction = model.predict(
                features
            )[0]

            current = weather_data["current"]

            current_temperature = (
                current["temperature_2m"]
            )


            # ==========================================
            # RAIN INFORMATION
            # ==========================================

            current_index = hourly_df[
                "time_difference"
            ].idxmin()

            current_hour_position = (
                hourly_df.index.get_loc(
                    current_index
                )
            )

            next_hour_position = (
                current_hour_position + 1
            )

            if next_hour_position < len(
                hourly_df
            ):

                next_hour = hourly_df.iloc[
                    next_hour_position
                ]

                next_hour_rain = (
                    next_hour["precipitation"]
                )

                next_hour_rain_probability = (
                    next_hour[
                        "precipitation_probability"
                    ]
                )

            else:

                next_hour = None

                next_hour_rain = 0.0

                next_hour_rain_probability = 0.0


            # ==========================================
            # LOCATION INFORMATION
            # ==========================================

            st.success(
                f"Weather data loaded for "
                f"{location_name}, {country}"
            )

            st.write(
                f"**Coordinates:** "
                f"{latitude:.4f}, "
                f"{longitude:.4f}"
            )

            st.write(
                f"**Weather time used:** "
                f"{weather_time}"
            )


            st.divider()


            # ==========================================
            # CURRENT WEATHER
            # ==========================================

            st.subheader(
                "🌤️ Current Weather"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🌡️ Temperature",
                    f"{current['temperature_2m']:.1f} °C"
                )

            with col2:

                st.metric(
                    "💧 Humidity",
                    f"{current['relative_humidity_2m']:.0f}%"
                )

            with col3:

                st.metric(
                    "💨 Wind Speed",
                    f"{current['wind_speed_10m']:.1f} km/h"
                )


            col4, col5, col6 = st.columns(3)

            with col4:

                st.metric(
                    "🧭 Wind Direction",
                    f"{current['wind_direction_10m']:.0f}°"
                )

            with col5:

                st.metric(
                    "☁️ Cloud Cover",
                    f"{current['cloud_cover']:.0f}%"
                )

            with col6:

                st.metric(
                    "🎚️ Pressure",
                    f"{current['pressure_msl']:.1f} hPa"
                )


            st.divider()


            # ==========================================
            # ML PREDICTION
            # ==========================================

            st.subheader(
                "🤖 ML Temperature Prediction"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Current Temperature",
                    f"{current_temperature:.1f} °C"
                )

            with col2:

                st.metric(
                    "Next-Hour Prediction",
                    f"{prediction:.1f} °C",
                    f"{prediction - current_temperature:+.1f} °C"
                )


            st.divider()


            # ==========================================
            # RAIN CONDITIONS
            # ==========================================

            st.subheader(
                "🌧️ Rain & Additional Conditions"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🌧️ Current Precipitation",
                    f"{current['precipitation']:.1f} mm"
                )

            with col2:

                st.metric(
                    "☔ Rain Probability",
                    f"{next_hour_rain_probability:.0f}%"
                )

            with col3:

                st.metric(
                    "🌧️ Next-Hour Rainfall",
                    f"{next_hour_rain:.1f} mm"
                )


            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**💨 Wind Gusts:** "
                    f"{current['wind_gusts_10m']:.1f} km/h"
                )

            with col2:

                if next_hour is not None:

                    st.write(
                        f"**🕐 Next Hour:** "
                        f"{next_hour['time']}"
                    )


            st.divider()


            # ==========================================
            # 24-HOUR TEMPERATURE FORECAST
            # ==========================================

            st.subheader(
                "📈 Next 24 Hours Temperature Forecast"
            )

            forecast_df = hourly_df.copy()

            forecast_df = forecast_df[
                forecast_df["time"] >= weather_time
            ].head(24)

            chart_df = forecast_df[
                ["time", "temperature_2m"]
            ].copy()

            chart_df = chart_df.set_index(
                "time"
            )

            chart_df = chart_df.rename(
                columns={
                    "temperature_2m":
                        "Temperature (°C)"
                }
            )

            st.line_chart(chart_df)


            st.divider()


            # ==========================================
            # INFORMATION
            # ==========================================

            st.subheader(
                "ℹ️ About the Prediction"
            )

            st.info(
                "Current weather values are obtained "
                "from live weather data. The Machine "
                "Learning model uses the current "
                "weather features to predict the "
                "temperature for the next hour."
            )


        # ==========================================
        # ERROR HANDLING
        # ==========================================

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )