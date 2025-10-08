import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulate 30 days of energy consumption data (in kWh)
days = np.arange(0, 30).reshape(-1, 1)
energy = 50 + 2 * days.squeeze() + np.random.randn(30) * 5

# Train linear regression model
model = LinearRegression()
model.fit(days, energy)

# Predict next 7 days
future_days = np.arange(30, 37).reshape(-1, 1)
predicted_energy = model.predict(future_days)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(days, energy, label="Observed", marker="o")
plt.plot(future_days, predicted_energy, label="Forecast", marker="x", linestyle="--")
plt.xlabel("Day")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Lightweight Energy Consumption Forecasting")
plt.legend()
plt.tight_layout()
plt.show()
