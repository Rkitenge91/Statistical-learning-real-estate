# ----------------------------
# Step 5: Feature Engineering
# ----------------------------


import pandas as pd

# Load cleaned data
df = pd.read_csv("data_processed/cleaned_real_estate.csv")

# Convert date
df["SaleDate"] = pd.to_datetime(df["SaleDate"])

# Create features
df["SaleYear"] = df["SaleDate"].dt.year
df["SaleMonth"] = df["SaleDate"].dt.month

# Drop original date
df = df.drop(columns=["SaleDate"])

# Selecting predictors
df_model = df[
    [
        "TotalFinishedArea",
        "LivingUnits",
        "TotalAppraisedValue",
        "AssrLandUse",
        "SaleYear",
        "SaleMonth",
        "PriceRatio"
    ]
]

# Encode categorical
df_model = pd.get_dummies(df_model, columns=["AssrLandUse"], drop_first=True)

print("\nColumns in feature-engineered dataset:")
print(df_model.columns.tolist())

print("\nShape of feature-engineered dataset:")
print(df_model.shape)

print("\nFirst 5 rows of feature-engineered dataset:")
print(df_model.head())

print("\nData types:")
print(df_model.dtypes)

df_model.to_csv("data_processed/feature_engineered_real_estate.csv", index=False)
print("\nFeature engineered dataset saved!")