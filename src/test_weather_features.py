import requests

latitude = 17.3850
longitude = 78.4867

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitude,
    "longitude": longitude,

    "hourly": (
        "temperature_2m,"
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
    ),

    "timezone": "auto",

    "forecast_days": 1
}

response = requests.get(
    url,
    params=params,
    timeout=10
)

response.raise_for_status()

data = response.json()

print("\nWeather API features received:")
print("=" * 60)

for key, values in data["hourly"].items():

    print(f"{key}: {values[0]}")