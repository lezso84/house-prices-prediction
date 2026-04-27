# House Prices Prediction Project

## Project Overview

This project focuses on predicting residential house prices using the Ames Housing dataset from the Kaggle House Prices competition.

The goal was not only to build a predictive model, but also to follow a structured machine learning workflow:

* data understanding,
* missing value handling,
* exploratory data analysis,
* feature engineering,
* baseline modeling,
* target transformation,
* decision tree modeling,
* ensemble methods,
* model tuning,
* and final model selection based on multiple evaluation metrics.

The final model selected for this project was a tuned Gradient Boosting Regressor.

---

## Dataset

The dataset contains detailed information about residential homes, including:

* overall material and finish quality,
* above-ground living area,
* garage capacity and garage area,
* basement size,
* neighborhood,
* year built,
* year remodeled,
* bathroom count,
* and the final sale price.

The target variable was:

```python
SalePrice
```

## Data Cleaning

The initial dataset contained several missing values. Missing value treatment was based on the meaning of each variable.

Columns with too many missing values were removed:

* `PoolQC`
* `MiscFeature`
* `Alley`
* `Fence`

Some categorical missing values were filled with "None" where the missing value represented the absence of a feature, for example:

* `GarageType`
* `GarageFinish`
* `GarageQual`
* `GarageCond`
* `BsmtQual`
* `BsmtCond`
* `BsmtExposure`
* `BsmtFinType1`
* `BsmtFinType2`
* `FireplaceQu`
* `MasVnrType`

Some numerical missing values were filled with 0, for example:

* `GarageYrBlt`
* `MasVnrArea`

`LotFrontage` was filled with the median value, while `Electrical` was filled with the mode.

Two outliers were removed based on the relationship between `GrLivArea` and `SalePrice`.

## Exploratory Data Analysis

The target variable, SalePrice, was right-skewed. This means that most houses were in the lower-to-mid price range, while a smaller number of expensive houses created a long right tail.

This was important because linear models often perform better when the target variable is closer to a normal distribution.

The strongest numerical predictors included:

* `OverallQual`
* `GrLivArea`
* `GarageCars`
* `GarageArea`
* `TotalBsmtSF`

The Neighborhood variable was also important and was encoded using one-hot encoding.

## Feature Engineering

Several new features were created to improve model performance.

```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
```

These engineered features were designed to capture:

* total usable house size,
* age of the property,
* time since remodeling,
* and total bathroom capacity.

## Target Transformation

Because SalePrice was right-skewed, a log transformation was applied to the target variable.

The model was trained on the transformed target, then predictions were converted back to the original price scale for evaluation.

This significantly improved model performance, especially for the linear regression baseline.

## Model Evaluation Metrics

The models were evaluated using the following metrics:

|Metric|Meaning|
|-|-|
|R²|Measures how much of the variance in house prices is explained by the model|
|MAE|Mean Absolute Error; average absolute prediction error|
|RMSE|Root Mean Squared Error; penalizes large prediction errors more strongly|

For final model selection, RMSE was especially important because large pricing errors are more costly in a real estate context.

## Model Comparison

Several models were tested and compared.

|Model|R² Original Scale|MAE|RMSE|Notes|
|-|-:|-:|-:|-|
|Untuned Decision Tree|0.7287|26,696|38,712|Severe overfitting|
|Tuned Decision Tree|0.8116|23,017|32,257|Improved, but still weak|
|Random Forest|0.8682|18,560|26,983|Stronger ensemble model|
|Log-Target Linear Regression|0.8824|18,411|25,490|Very strong baseline|
|Base Gradient Boosting|0.8956|17,336|24,012|New best model|
|Tuned Gradient Boosting|0.9009|16,804|23,402|Final selected model|
|Regularized XGBoost|0.9098|16,976|26,307|Higher R², but worse RMSE|

## Final Model

The final selected model was:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
```

Final performance:

|Metric|Value|
|-|-:|
|Train R² on log target|0.9652|
|Test R² on log target|0.8696|
|Test R² on original price scale|0.9009|
|MAE|16,804|
|RMSE|23,402|

## Why Gradient Boosting Was Selected

The tuned Gradient Boosting model was selected as the final model because it provided the best overall balance between predictive power and error control.

Although the regularized XGBoost model achieved a slightly higher R² score on the original price scale, its RMSE was significantly worse.

This means that XGBoost explained slightly more overall variance, but made larger errors on some observations.

For a house price prediction problem, large errors are especially important. A model that occasionally makes very large mistakes is less attractive from a practical business perspective.

The tuned Gradient Boosting model had:

* the best RMSE,
* the best overall error profile,
* strong R² performance,
* better control of large prediction errors,
* and more stable business usefulness.

Therefore, the final decision was not based only on the highest R² score, but on the full model performance profile.

## Key Learnings

This project showed that a more complex model is not automatically better.

The log-transformed linear regression model was already a very strong baseline. The untuned decision tree severely overfitted, while the tuned decision tree improved but remained weaker.

Random Forest improved substantially over a single decision tree, but it still did not outperform the log-target linear regression model.

Gradient Boosting provided the best result because it learned sequential corrections to previous errors and captured nonlinear relationships more effectively.

XGBoost was also tested, but in this version of the project it did not become the final model because its RMSE was worse than the tuned Gradient Boosting model.

The most important lesson was that model selection should not rely on one metric only. R², MAE, RMSE, train-test gap, and business interpretation should all be considered together.

## Final Conclusion

The best model in this project was a tuned Gradient Boosting Regressor.

It achieved:

```text
R² original scale: 0.9009
MAE:               16,804
RMSE:              23,402
```

This model provided the strongest balance between accuracy, stability, and practical usefulness for predicting house prices.

