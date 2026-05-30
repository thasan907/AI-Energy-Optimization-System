import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv(
    "data/household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values=["?"]
)

# Clean Data
df.dropna(inplace=True)

# Datetime
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

# Convert Power Column
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"]
)

# Daily Average Consumption
daily = df.resample(
    "D",
    on="Datetime"
)["Global_active_power"].mean()

daily = daily.reset_index()

# Create Features
daily["Previous_Day"] = daily[
    "Global_active_power"
].shift(1)

daily["Day"] = daily["Datetime"].dt.day
daily["Month"] = daily["Datetime"].dt.month
daily["Year"] = daily["Datetime"].dt.year

daily.dropna(inplace=True)

# Features and Target
X = daily[
    ["Previous_Day", "Day", "Month", "Year"]
]

y = daily["Global_active_power"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(
    y_test,
    predictions
)

print("Mean Absolute Error:", mae)

# Sample Predictions
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print(result.head(10))
import joblib

joblib.dump(
    model,
    "models/energy_model.pkl"
)

print("Model Saved Successfully!")