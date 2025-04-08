import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv("traffic_data.csv")
if len(df) < 10:  # Need enough data
    print("Not enough data for prediction.")
    exit()

# Prepare data
X = df["timestamp"].values.reshape(-1, 1)
y = df["density"].values  # Density is now in vehicles per million pixels

# Train simple model
model = LinearRegression()
model.fit(X, y)

# Predict next 2 seconds (assuming 30 FPS, ~60 frames, but using timestamp seconds)
last_time = df["timestamp"].iloc[-1]
future_time = np.array([[last_time + 2]])  # 2 seconds ahead
predicted_density = model.predict(future_time)[0]

# Output results
print(f"Current Density (per million pixels): {df['density'].iloc[-1]:.2f}")
print(f"Predicted Density in 2s (per million pixels): {predicted_density:.2f}")

# Define congestion threshold (adjusted for new scale)
if predicted_density > 75:  # Congestion threshold for ~16-17 vehicles in 640x360
    print("Warning: Congestion likely!")
elif predicted_density > 50:  # Moderate traffic threshold for ~11-12 vehicles
    print("Moderate traffic detected!")
else:
    print("Traffic looks clear.")