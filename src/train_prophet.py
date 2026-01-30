import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from itertools import product

# -------------------------------
# STEP 1: Generate Sample Data
# -------------------------------
dates = pd.date_range(start="2020-01-01", periods=500)
values = np.sin(np.arange(500)/20) + np.random.normal(0, 0.3, 500)

df = pd.DataFrame({
    "ds": dates,
    "y": values
})

# Train-test split
train = df.iloc[:-50]
test = df.iloc[-50:]

# -------------------------------
# STEP 2: Hyperparameter Grid
# -------------------------------
param_grid = {
    "changepoint_prior_scale": [0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 5.0]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
best_rmse = float("inf")
best_params = None

# -------------------------------
# STEP 3: Hyperparameter Search
# -------------------------------
for params in all_params:
    model = Prophet(**params)
    model.fit(train)

    future = model.make_future_dataframe(periods=50)
    forecast = model.predict(future)

    preds = forecast["yhat"].iloc[-50:].values
    rmse = np.sqrt(mean_squared_error(test["y"], preds))

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("Best Params:", best_params)

# -------------------------------
# STEP 4: Train Final Model
# -------------------------------
model = Prophet(**best_params)
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# -------------------------------
# STEP 5: Save Outputs
# -------------------------------
forecast[["ds", "yhat"]].to_csv("outputs/forecast.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(df["ds"], df["y"], label="Actual")
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
plt.legend()
plt.savefig("outputs/forecast_plot.png")

print("Training Complete.")
