import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/weather.csv")

print("=" * 50)
print("WEATHER DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

print("\nTemperature Statistics:")
print(df["temp"].describe())

print("\nWeather Conditions:")
print(df["weather"].value_counts())

# Correlation analysis
numeric_columns = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_columns.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("Weather Features Correlation Heatmap")
plt.tight_layout()

plt.savefig("screenshots/correlation_heatmap.png")
plt.show()

# Temperature distribution
plt.figure(figsize=(8, 5))

plt.hist(df["temp"], bins=20)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("screenshots/temperature_distribution.png")
plt.show()

# Temperature over time
plt.figure(figsize=(12, 5))

plt.plot(df["temp"])

plt.title("Temperature Variation")
plt.xlabel("Observation")
plt.ylabel("Temperature")

plt.tight_layout()
plt.savefig("screenshots/temperature_variation.png")
plt.show()

print("\nEDA completed successfully!")
print("Graphs saved inside the screenshots folder.")