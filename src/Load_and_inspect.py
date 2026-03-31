import pandas as pd

# -----------------------------
# Step 0: Raw Dataset: Loading and Inspection
# -----------------------------


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

# -----------------------------
# Step 1: Drop irrelevant columns
# -----------------------------

# List of columns to drop
columns_to_drop = [
    "OBJECTID",
    "PropertyID",
    "ParcelID",
    "GlobalID",
    "xrDeedID",
    "LegalReference",
    "OwnerLastName",
    "OwnerFirstName",
    "PrimaryGrantor",
    "ApartmentUnitNumber",
    "StreetNameAndWay",
    "LocationStartNumber",
    "xrPrimaryNeighborhoodID",
    "xrCompositeLandUseID",
    "xrBuildingTypeID",
    "xrSalesValidityID"
]

# Drop columns
df = df.drop(columns=columns_to_drop)

# Check remaining columns
print("\nRemaining columns after dropping:")
print(df.columns.tolist())

print("\nNew shape:", df.shape)


# -----------------------------
# Step 2: Handle missing values
# -----------------------------

# 1. Drop column with too many missing values
df = df.drop(columns=["LandSF"])

# 2. Remove rows with missing critical variables
df = df.dropna(subset=["SalePrice", "TotalAppraisedValue"])

# 3. Remove rows with missing smaller variables
df = df.dropna(subset=["TotalFinishedArea"])

# Check remaining missing values
print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nNew shape after removing missing values:", df.shape)


# --------------------------------------------------------------
# Step 3: Remove invalid observations and create target variable
# --------------------------------------------------------------

# 1. Remove invalid values
df = df[(df["SalePrice"] > 0) & (df["TotalAppraisedValue"] > 0)]
df = df[df["SalePrice"] >= 1000]

# Check new shape after removing invalid rows
print("\nShape after removing invalid observations:", df.shape)

# 2. Create target variable
df["PriceRatio"] = df["SalePrice"] / df["TotalAppraisedValue"]

# Check summary of new variable
print("\nSummary of PriceRatio:")
print(df["PriceRatio"].describe())

# Preview dataset
print("\nFirst 5 rows with PriceRatio:")
print(df.head())


# -----------------------------
# Step 4: Remove extreme outliers
# -----------------------------

# Compute percentiles
lower_bound = df["PriceRatio"].quantile(0.01)
upper_bound = df["PriceRatio"].quantile(0.99)

# Filter dataset
df = df[(df["PriceRatio"] >= lower_bound) & (df["PriceRatio"] <= upper_bound)]

print("\nShape after removing outliers:", df.shape)

print("\nUpdated PriceRatio summary:")
print(df["PriceRatio"].describe())

