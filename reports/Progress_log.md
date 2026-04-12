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


## Step 6: Exploratory Data Analysis

Summary:

We explored the distribution of the target variable (PriceRatio) and examined its relationship with key predictors such as property type and time.

Key findings:
- The distribution of PriceRatio is centered around 1, indicating that most properties sell close to their assessed value.
- There is moderate variation, with both undervalued (<1) and overvalued (>1) properties present.
- The average PriceRatio differs slightly across property types, suggesting variation in pricing behavior by category.
- Over time, PriceRatio shows an increasing trend, indicating that sale prices are becoming higher relative to assessed values.

Result:

The exploratory analysis confirms that the dataset contains meaningful structure and variation, making it suitable for predictive modeling.


## Step 7: Train/Test Split

Summary:

The dataset was split into training and test sets to prepare for predictive modeling. The target variable is PriceRatio, and all remaining variables are used as predictors.

Key actions:
- Defined predictors (X) and target variable (y)
- Split data into 80% training and 20% test sets using train_test_split
- Set random_state = 42 for reproducibility

Result:
The data is now properly partitioned, ensuring that model evaluation will be performed on unseen data.


## Step 8: Baseline Regression Models

Summary:

We implemented two baseline regression models, Ridge and Lasso, to predict PriceRatio.

Key actions:
- Fit Ridge and Lasso models on the training data
- Generated predictions on the test set
- Evaluated model performance using RMSE and MAE

Result:
- Ridge RMSE: 0.2107  
- Ridge MAE: 0.1621  
- Lasso RMSE: 0.2133  
- Lasso MAE: 0.1644  
Both models produced similar performance metrics.


## Step 9: Cross-Validation for Baseline Models

- Applied 5-fold cross-validation using KFold with shuffling.
- Evaluated Ridge and Lasso using cross-validated MSE and RMSE.

Results:
- Ridge 5-fold CV RMSE: 0.2079  
- Lasso 5-fold CV RMSE: 0.2064  
Cross-validation results are consistent with test set performance, indicating stable model behavior.



## Step 10: Random Forest Modeling

 - fit a random forest to the engineered dataset
 - observed feature importance through built-in functions

 Results:
 - TotalFinishedArea is the most important feature by permutation
 - SaleMonth and SaleYear are the next two most important

## Step 11: Ridge Regression (Gradient Descient from Scratch)

 - fit a ridge regression model using both the closed form solution and gradient descent
 - fit a model using skl for sanity check
 - computed predicted values and saved to csv

 Results:
 - Gradient Descent MSE: 1.1725
 - Closed Form MSE 0.04399
 - Builtin Model MSE 0.0434

## Step 12: Evaluating Model Errors

 - read in predicted values of each model
 - computed MSE
 - created a pandas DataFrame to directly compare

 Results:
 - RandomForestRMSE          0.225446
 - RidgeClosedFormRMSE       0.209744
 - RidgeBuiltinRMSE          0.208367
 - RidgeGradientDescentRMSE  1.082838

## Step 13: Bayesian Regression Model

Summary:

We implemented a Bayesian linear regression model using PyMC to estimate the relationship between predictors and PriceRatio, while explicitly capturing uncertainty in parameter estimates.

Key actions:
- Defined response variable (PriceRatio) and predictors (TotalFinishedArea, LivingUnits, TotalAppraisedValue, SaleYear, SaleMonth, and property type indicators)
- Standardized continuous variables using StandardScaler to improve model convergence
- Specified priors
- Constructed linear predictor combining all features
- Performed posterior sampling 

Results:
- Posterior distributions were stable and approximately symmetric, indicating reliable estimation
- Trace plots showed good mixing across chains with no major convergence issues
- R-hat values were all approximately 1.00, confirming convergence
- SaleYear showed the strongest positive association with PriceRatio (mean ≈ 0.087, 95% credible interval fully above 0)
- LivingUnits also had a positive effect (mean ≈ 0.065), though smaller than SaleYear
- TotalAppraisedValue had a negligible effect (mean ≈ 0.004) with a credible interval crossing zero, indicating weak evidence of influence
- Some property type indicators showed moderate variability, but with wider credible intervals, suggesting higher uncertainty

Conclusion:
The Bayesian model provides a deeper understanding of the drivers of PriceRatio by quantifying uncertainty in each coefficient. It highlights that time-related effects (SaleYear) play a more significant role than appraisal value, offering more nuanced insight into market behavior.


