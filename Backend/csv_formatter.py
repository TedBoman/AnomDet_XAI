from datetime import datetime
import pandas as pd
import numpy as np

DATASET_PATH = "../Datasets/daily_minimum_temperatures_in_me.csv"

def to_timestamp(x: str):
    return datetime.timestamp(x)

def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False, parse_dates=["Date"], index_col=False)
    print(df)
    df.rename(columns={"Date": "timestamp"}, inplace=True)
    #df["timestamp"] = df["timestamp"].astype(str)
    df["timestamp"] = df["timestamp"].apply(to_timestamp)
    df["timestamp"] = df["timestamp"].astype("int32")
    print(df)
    df.to_csv("../Datasets/temperatures.csv", index=False)

if __name__ == "__main__":
    main()