import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"
files = glob(data_path + "*.csv")

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
f = files[0]
participant = f.split("-")[0].lstrip(data_path + "\\")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
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
        


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
# df_acc.info()

df_acc.index = pd.to_datetime(df_acc["epoch (ms)"], unit="ms")
df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")

del df_acc["epoch (ms)"]
del df_acc["time (01:00)"]
del df_acc["elapsed (s)"]
del df_gyr["epoch (ms)"]
del df_gyr["time (01:00)"]
del df_gyr["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"
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
    "gyr_x",
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
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last" 
    }
numeric_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
non_numeric_cols = ["participant", "label", "category", "set"]

numeric_sampling = {col: "mean" for col in numeric_cols}
non_numeric_sampling = {col: "last" for col in non_numeric_cols}

resampled_numeric = data_merged[numeric_cols].resample(rule="200ms").agg(numeric_sampling)
resampled_non_numeric = data_merged[non_numeric_cols].resample(rule="200ms").agg(non_numeric_sampling)

data_merged = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
nan_rows = data_merged[data_merged['acc_x'].isna() & data_merged['gyr_x'].isna()]

data_merged = data_merged.dropna(subset=['acc_x', 'gyr_x'])

data_merged.info()

data_merged["set"] = data_merged["set"].astype("int")

# -------------------------------------------------------------- 
# Export dataset
# --------------------------------------------------------------
data_merged.to_pickle("../../data/interim/01_data_processed.pkl")