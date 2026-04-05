# ----------------------------------
# Step 6: Exploratory Data Analysis
# ----------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Load feature engineered dataset
df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

# Histogram: Price Ratio Distribution
plt.figure()
plt.hist(df["PriceRatio"], bins=50)
plt.title("Distribution of PriceRatio")
plt.xlabel("PriceRatio")
plt.ylabel("Frequency")

plt.savefig("figures/price_ratio_distribution.png")
plt.show()

#-------------------------------------------------------------------------------------------------------------

# Get columns related to property type
landuse_cols = [col for col in df.columns if "AssrLandUse" in col]

# Compute average PriceRatio per category
avg_ratios = df[landuse_cols].multiply(df["PriceRatio"], axis=0).sum() / df[landuse_cols].sum()

plt.figure()
avg_ratios.plot(kind="bar")
plt.title("Average PriceRatio by Property Type")
plt.ylabel("Average PriceRatio")

plt.savefig("figures/avg_price_ratio_by_property.png")
plt.show()

#------------------------------------------------------------------------------------------------------------

avg_by_year = df.groupby("SaleYear")["PriceRatio"].mean()

plt.figure()
avg_by_year.plot()
plt.title("Average PriceRatio Over Time")
plt.xlabel("Year")
plt.ylabel("Average PriceRatio")

plt.savefig("figures/price_ratio_over_time.png")
plt.show()

