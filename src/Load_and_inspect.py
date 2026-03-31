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