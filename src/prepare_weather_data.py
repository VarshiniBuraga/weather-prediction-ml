import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/historical_weather.csv")
OUTPUT_FILE = Path("data/processed_weather.csv")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 60)
print("WEATHER DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal rows: {len(df)}")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# --------------------------------------------------
# Sort correctly by city and time
# --------------------------------------------------

df = df.sort_values(
    ["city", "time"]
).reset_index(drop=True)

# --------------------------------------------------
# Create NEXT-HOUR temperature target
# --------------------------------------------------

df["target_temperature"] = (
    df.groupby("city")["temperature_2m"].shift(-1)
)

# --------------------------------------------------
# Create time-based features
# --------------------------------------------------

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek
df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear

# Cyclic time features
df["hour_sin"] = (
    __import__("numpy").sin(
        2 * __import__("numpy").pi * df["hour"] / 24
    )
)

df["hour_cos"] = (
    __import__("numpy").cos(
        2 * __import__("numpy").pi * df["hour"] / 24
    )
)

df["day_sin"] = (
    __import__("numpy").sin(
        2 * __import__("numpy").pi * df["day_of_year"] / 365
    )
)

df["day_cos"] = (
    __import__("numpy").cos(
        2 * __import__("numpy").pi * df["day_of_year"] / 365
    )
)

# --------------------------------------------------
# Remove rows without a target
# --------------------------------------------------

df = df.dropna(
    subset=["target_temperature"]
).reset_index(drop=True)

# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# Display information
# --------------------------------------------------

print(f"\nProcessed rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nRecords per city:")
print(df["city"].value_counts())

print("\nSample:")
print(
    df[
        [
            "city",
            "time",
            "temperature_2m",
            "target_temperature",
            "hour",
            "month"
        ]
    ].head(10)
)

print("\nMissing values:")
print(
    df.isnull().sum()
)

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)

print(f"Saved to: {OUTPUT_FILE}")