import requests
import pandas as pd
import time
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_FILE = Path("data/historical_weather.csv")

# One year of historical data
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# Indian cities
CITIES = {
    "Ongole": (15.50357, 80.04454),
    "Hyderabad": (17.3850, 78.4867),
    "Vijayawada": (16.5062, 80.6480),
    "Visakhapatnam": (17.6868, 83.2185),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
}

# Weather variables
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "pressure_msl",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "uv_index",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# --------------------------------------------------
# Download data for one city
# --------------------------------------------------

def download_city_weather(city, latitude, longitude):

    print(f"\nDownloading weather data for {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data returned for {city}")

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    # Add city information
    df["city"] = city
    df["latitude"] = latitude
    df["longitude"] = longitude

    print(f"{city}: {len(df)} hourly records downloaded.")

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    all_data = []

    print("=" * 60)
    print("HISTORICAL WEATHER DATA DOWNLOADER")
    print("=" * 60)

    for city, (latitude, longitude) in CITIES.items():

        try:
            city_df = download_city_weather(
                city,
                latitude,
                longitude
            )

            all_data.append(city_df)

            # Small delay between requests
            time.sleep(1)

        except Exception as e:
            print(f"ERROR downloading {city}: {e}")

    if not all_data:
        print("\nNo data was downloaded.")
        return

    # Combine all cities
    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Convert time column
    final_df["time"] = pd.to_datetime(final_df["time"])

    # Sort data
    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # Create data directory if needed
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save CSV
    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETED")
    print("=" * 60)

    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")

    print("\nRecords per city:")
    print(final_df["city"].value_counts())

    print("\nFirst 5 rows:")
    print(final_df.head())

    print("\nMissing values:")
    print(final_df.isnull().sum())


if __name__ == "__main__":
    main()