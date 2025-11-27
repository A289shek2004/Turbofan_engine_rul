import pandas as pd
import os
import numpy as np

DATA_DIR = "."
dataset = "FD001"

cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

print(f"Loading {dataset}...")
try:
    train_path = os.path.join(DATA_DIR, f"train_{dataset}.txt")
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    print("Train head:")
    print(train.head())
    print("Train dtypes:")
    print(train.dtypes)
    
    # Check for non-numeric
    for c in cols:
        if train[c].dtype == 'object':
            print(f"Column {c} is object type. Unique values (first 10): {train[c].unique()[:10]}")

    test_path = os.path.join(DATA_DIR, f"test_{dataset}.txt")
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
    print("Test head:")
    print(test.head())
    print("Test dtypes:")
    print(test.dtypes)
    
    rul_path = os.path.join(DATA_DIR, f"RUL_{dataset}.txt")
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
    print("RUL head:")
    print(rul.head())

except Exception as e:
    print(f"Error: {e}")
