import requests

city = "Hyderabad"

# 1. Search for the city
geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

geocoding_params = {
    "name": city,
    "count": 5,
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

print("\nLocation found:")
print("Name:", location["name"])
print("Country:", location.get("country"))
print("Latitude:", latitude)
print("Longitude:", longitude)


# 2. Get weather data
weather_url = "https://api.open-meteo.com/v1/forecast"

weather_params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "apparent_temperature,"
        "precipitation,"
        "cloud_cover,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "wind_gusts_10m"
    ),
    "timezone": "auto"
}

weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=10
)

weather_response.raise_for_status()

weather_data = weather_response.json()

print("\nCurrent Weather:")

current = weather_data["current"]

for key, value in current.items():
    print(f"{key}: {value}")