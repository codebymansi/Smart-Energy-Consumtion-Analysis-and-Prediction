
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# 1. Project paths
# ---------------------------------

DATA_PATH = "dataset/household_power_consumption.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading electricity dataset...")

# ---------------------------------
# 2. Load dataset
# ---------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# ---------------------------------
# 3. Data cleaning
# ---------------------------------

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

df.drop(columns=["Date", "Time"], inplace=True)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df.dropna(subset=["datetime", "Global_active_power"], inplace=True)

df = df.sort_values("datetime").set_index("datetime")

print("Cleaning completed.")
print("Records:", len(df))

# ---------------------------------
# 4. Hourly consumption
# ---------------------------------

hourly = df["Global_active_power"].resample("h").mean()

# Convert average kW to hourly kWh
hourly = hourly.dropna()

data = hourly.to_frame(name="energy_kwh")

# ---------------------------------
# 5. Feature engineering
# ---------------------------------

data["hour"] = data.index.hour
data["day"] = data.index.day
data["month"] = data.index.month
data["weekday"] = data.index.dayofweek

data["lag_1"] = data["energy_kwh"].shift(1)
data["lag_2"] = data["energy_kwh"].shift(2)
data["lag_3"] = data["energy_kwh"].shift(3)

data["rolling_mean_3"] = (
    data["energy_kwh"].shift(1).rolling(3).mean()
)

data.dropna(inplace=True)

features = [
    "hour",
    "day",
    "month",
    "weekday",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3"
]

X = data[features]
y = data["energy_kwh"]

# ---------------------------------
# 6. Chronological train/test split
# ---------------------------------

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ---------------------------------
# 7. Train ML models
# ---------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    joblib.dump(
        model,
        f"models/{name.lower().replace(' ', '_')}.pkl"
    )

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# ---------------------------------
# 8. Save evaluation results
# ---------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Save actual and predicted values
prediction_df = pd.DataFrame({
    "Actual": y_test,
    "Linear Regression": predictions["Linear Regression"],
    "Random Forest": predictions["Random Forest"]
})

prediction_df.to_csv(
    "results/predictions.csv"
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nTraining completed successfully!")
print("Models saved inside models folder.")
