## Step 1: Data loading and initial inspection

Date: 2026-03-20

Summary:
The dataset contains 7,410 observations and 23 variables, representing individual real estate transactions. Key variables include SalePrice and TotalAppraisedValue, which will be used to construct the target variable. Initial inspection revealed no duplicate records but identified missing values in several variables, including SalePrice and property-related features. This highlights the need for careful data cleaning before modeling.

Key findings:
- 7,410 observations and 23 variables
- No duplicate records were found
- Missing values are present in SalePrice and several property-related variables
- TotalAppraisedValue is complete and suitable for target construction
- Several identifier and owner-related variables are likely not useful for modeling

Next step:
Proceed with data cleaning by removing irrelevant variables and handling missing values in key fields.

--------------------------------------------------------------------------------------------------------------

## Step 2: Removal of irrelevant variables

Summary:
We removed variables that do not contribute to predicting the outcome, including identifiers, owner-related fields, and text-heavy attributes. These variables do not provide meaningful information for modeling and may introduce noise or overfitting.

Key actions:
- Dropped identifier columns (e.g., OBJECTID, PropertyID, GlobalID)
- Removed owner-related variables (e.g., OwnerFirstName, OwnerLastName)
- Eliminated text-based and legal reference fields

Result:
The dataset was reduced to a smaller set of relevant variables that are more suitable for statistical modeling.

Next step:
Handle missing values and remove invalid observations.


## Step 3: Removal of invalid observations and target construction

Summary:
We removed observations with non-positive or unrealistic values in SalePrice and TotalAppraisedValue to ensure valid computations. Additionally, very small sale prices were excluded as they do not represent meaningful market transactions. We then constructed the target variable, PriceRatio, defined as:

PriceRatio = SalePrice / TotalAppraisedValue

This variable captures how sale prices compare to assessed values, allowing for meaningful analysis of pricing behavior.

Key actions:
- Removed observations with SalePrice ≤ 0 or TotalAppraisedValue ≤ 0
- Removed observations with unrealistically low SalePrice values
- Created PriceRatio = SalePrice / TotalAppraisedValue

Result:
The dataset now includes a well-defined and meaningful target variable suitable for modeling.


## Step 4: Remove unrealistic sale-to-appraisal ratios

Summary:
To improve data quality, we filtered out observations with unrealistic PriceRatio values. Extremely low or high ratios likely reflect data errors, non-arm’s length transactions, or unusual sales that do not represent typical market behavior.

We retained only observations where PriceRatio is between 0.5 and 1.5, ensuring that the analysis focuses on realistic property transactions.

Key actions:
- Removed observations with PriceRatio < 0.5 (extreme undervaluation)
- Removed observations with PriceRatio > 1.5 (extreme overvaluation)

Distribution check:
A histogram of the filtered PriceRatio shows a well-centered distribution ranging from approximately 0.5 to 1.5. The majority of observations are concentrated between about 0.8 and 1.3, with a peak around 1.0–1.1.

This indicates that most properties sell close to their assessed value, with a slight right skew reflecting some higher sale-to-appraisal ratios.

Result:
The distribution of PriceRatio is now more stable and centered around 1, making the dataset more suitable for modeling and interpretation.


## Step 5: Feature Engineering

Summary:
We transformed the cleaned dataset into a modeling-ready format by generating time-based features and encoding categorical variables.

Key actions:
- Converted SaleDate to datetime format
- Extracted SaleYear and SaleMonth
- Selected key predictors for modeling
- Encoded categorical variable AssrLandUse using one-hot encoding
- Created final modeling dataset

Result:
The dataset is now fully prepared for modeling, with structured numerical and categorical features.

