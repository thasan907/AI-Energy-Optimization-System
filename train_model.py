import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values=["?"]
)

df.dropna(inplace=True)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"]
)

daily_power = df.resample(
    "D",
    on="Datetime"
)["Global_active_power"].mean()

plt.figure(figsize=(12,5))
plt.plot(daily_power)

plt.title("Daily Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Average Power Consumption")

plt.show()