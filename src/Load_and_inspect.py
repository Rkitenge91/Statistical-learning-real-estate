import pandas as pd

# Load dataset
df = pd.read_csv("data_raw/Res.csv")

# Basic inspection
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nInfo:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:", df.duplicated().sum())

# Preview data
print("\nFirst 5 rows:")
print(df.head())
