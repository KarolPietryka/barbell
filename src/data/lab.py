import pandas as pd
from glob import glob
import os

data_path = "data/raw/MetaMotion/"
files = glob(data_path + "*.csv")
def read_data_from_files(files):
    df_acc = pd.DataFrame()
    df_gyr = pd.DataFrame()

    df_acc_set = 1
    df_gyr_set = 1
    for f in files:
        df = pd.read_csv(f)
        df["participant"] = f.split("-")[0].lstrip(data_path + "\\")
        df["label"] = f.split("-")[1]
        df["category"] = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        if("Accelerometer" in f):
            df["set"] = df_acc_set
            df_acc_set += 1
            df_acc = pd.concat([df_acc, df])
        elif("Gyroscope" in f):
            df["set"] = df_gyr_set
            df_gyr_set += 1
            df_gyr = pd.concat([df_gyr, df])

    df_acc.index = pd.to_datetime(df_acc["epoch (ms)"], unit="ms")
    df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")

    del df_acc["epoch (ms)"]
    del df_acc["time (01:00)"]
    del df_acc["elapsed (s)"]
    del df_gyr["epoch (ms)"]
    del df_gyr["time (01:00)"]
    del df_gyr["elapsed (s)"]

    return df_acc, df_gyr
# Main call
df_acc, df_gyr = read_data_from_files(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([df_acc.iloc[:,:3], df_gyr], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gye_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"  
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gye_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last" 
    }
numeric_cols = ["acc_x", "acc_y", "acc_z", "gye_x", "gyr_y", "gyr_z", "set"]
non_numeric_cols = ["participant", "label", "category"]

numeric_sampling = {col: "mean" for col in numeric_cols}
non_numeric_sampling = {col: "last" for col in non_numeric_cols}

resampled_numeric = data_merged[numeric_cols].resample(rule="200ms").agg(numeric_sampling)
resampled_non_numeric = data_merged[non_numeric_cols].resample(rule="200ms").agg(non_numeric_sampling)

data_merged = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)