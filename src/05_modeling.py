# ----------------------------------
# Step 7: Train/ Test Split
# ----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

# Define X and y
X = df.drop(columns=["PriceRatio"])
y = df["PriceRatio"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check shapes
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTarget variable (train):")
print(y_train.describe())