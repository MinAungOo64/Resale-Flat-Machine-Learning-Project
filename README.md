# Resale-Flat-Machine-Learning-Project
Machine learning project to predict future flat resale prices using SciKit Learn library  

This project aims to predict future flat resale prices.  
The original datasets can be found here https://data.gov.sg/collections/189/view

---
## EDA and Data Processing
I will highlight some important steps below regarding data processing and explain the rationale behind each decision.

### Selecting the data
Data between 2012 to 2025 are selected for training. Resale prices before 2012 are based on when the transactions are aprroved while resale prices after 2012 are based on when transactions are registered. Furthermore, we do not want the models to learn outdated price trends.

### Resale Price over the years
Resale prices seem to increase drstically after covid-19. This maybe due to BTO projects being halted, leading to limited supply and increased demands for housing.  
With this information in mind, 2 dataframes will be created, one covering the period from 2012 to 2025 and one covering the period post-covid from 2020 to 2025.

![image](https://github.com/user-attachments/assets/fbdb89bd-11e6-4ab4-b7c4-f8f996505071)


### New Features for Date-Time Features
Regression models do not handle date-time features well, instead we engineer new features that represent time-based differences in a numerical format.  
#### remaining_lease
We will calculate the remaining_lease at the time of resale since we want to predict the resale price at the time of resale.  
```python
df['remaining_lease'] = df.apply(
    lambda x: 99 - years_difference(x['lease_commence_date'], x['month']),
    axis=1
)
```  
#### months_since_base
The month column records the date of each resale transaction. We set the earliest transaction date, 2012-03-01, as the baseline. Using this baseline, we calculate the number of months elapsed between the earliest transaction and each subsequent transaction.  
This time-based feature helps the model capture market trends and inflation effects over time. For example, the model can learn that resale prices generally increase as the time since the baseline grows.  
```python
# Define baseline date
baseline_date = pd.to_datetime('2012-03-01')

df['months_since_base'] = df.apply(
    lambda x: months_difference(baseline_date, x['month']),
    axis=1
)
```

### Assessing resale_price and floor_area_sqm
The distribution of both features are right-skewed. Logging the values will reduce the skewness.

![image](https://github.com/user-attachments/assets/564df2fb-35b6-4ded-a153-df435a260d1c)

The correlation coefficient between resale_price and floor_area_sqm is relatively low (0.59) implying that they do not have a perfect linear relationship.
After the log transformation, the correlation coefficient increased to 0.65.
The log transformation reduced the impact of extreme values (outliers) and helped normalize the distribution, making the correlation more representative of the general trend.

```python
# Apply log transformation
df['log_resale_price'] = np.log(df['resale_price'])
df['log_floor_area_sqm'] = np.log(df['floor_area_sqm'])

# Recompute correlation
log_correlation = df[['log_resale_price', 'log_floor_area_sqm']].corr()
print(log_correlation)
```

### Encoding for categorial variables
Categorial features such as block and street_name have too many unique values will be encoded. Furthermore, the town feaure is a geo-location feature. Hence block and street_name can be dropped for better generalisation.

![image](https://github.com/user-attachments/assets/051ea7b9-e632-456a-a858-e9cd19e1cb88)

#### Ordinal Encoding
flat_type and storey_range have generally clear ordering to be ordinally encoded.

![image](https://github.com/user-attachments/assets/66e5e98d-f85e-4674-9423-397764d5b658)
![image](https://github.com/user-attachments/assets/54f8287d-ce17-4b04-833a-57a3cf8dfa26)

#### One-hot Encoding
town and flat model do not have clear ordering hence will be one-hot encoded.

![image](https://github.com/user-attachments/assets/c2bf5621-38d0-4f3e-b618-cd83757813c0)
![image](https://github.com/user-attachments/assets/23a022b8-3a57-4f6d-87fc-9ba016d6d00d)


### Two dataframes  
As mentioned above, 2 dataframes will be created, df_base covering the period from 2012 to 2025 and df_post_covid covering the period post-covid from 2020 to 2025.  
We save them as `.pkl' files

---
## Splitting Train and Test
We split the Train Test to be 80/20. We do not want a random split as the objective is for the model to predict future unseen resale prices. As such the train set will be data from earlier periods than those of the test set.

```python
# 1. Sort by time column
df_base_sorted = df_base.sort_values(by='months_since_base')
df_post_sorted = df_post_covid.sort_values(by='months_since_base')

# 2. Compute split indices
split_idx_base = int(len(df_base_sorted) * 0.8)
split_idx_post = int(len(df_post_sorted) * 0.8)

# 3. Split df_base into train/test
df_base_train = df_base_sorted.iloc[:split_idx_base]
df_base_test = df_base_sorted.iloc[split_idx_base:]
```

After splitting train and test, we separate the input features X and the output y for both dataframes. We will have a total of 8 files.

```python
# 5. Save all 8 files
X_base_train.to_pickle("X_base_train.pkl")
X_base_test.to_pickle("X_base_test.pkl")
y_base_train.to_pickle("y_base_train.pkl")
y_base_test.to_pickle("y_base_test.pkl")

X_post_train.to_pickle("X_post_train.pkl")
X_post_test.to_pickle("X_post_test.pkl")
y_post_train.to_pickle("y_post_train.pkl")
y_post_test.to_pickle("y_post_test.pkl")
````
---
## Oridnary Least Squares, F-test, t-test
### df_base  
The F-test returns a p-value close to 0, indicating that the regression model as a whole is statistically significant — i.e., at least one predictor has a non-zero effect on resale price.

The t-tests for individual coefficients identify six predictors that are not statistically significant at the 5% level (p > 0.05): town_SERANGOON, flat_model_Apartment, flat_model_Premium Apartment Loft, flat_model_Premium Maisonette, flat_model_Type S1, and flat_model_Type S2.

Since the six predictors are one-hot encoded variables or dummy variables part of a category, we have decided to keep them instead of dropping.
### df_post_covid  
The F-test returns a p-value close to 0, indicating that the regression model as a whole is statistically significant — i.e., at least one predictor has a non-zero effect on resale price.

The t-tests for individual coefficients identify three predictors that are not statistically significant at the 5% level (p > 0.05): flat_model_Model A2, flat_model_New Generation, flat_model_Premium Apartment

Since the three predictors are one-hot encoded variables or dummy variables part of a category, we have decided to keep them instead of dropping.

---
## Linear Regression, Regularisation and Polynomial Regression

| Data | Model                    | Hyperparam       | MAE       | MSE         | RMSE      | R²    | Adjusted R² |
|------|--------------------------|------------------|-----------|-------------|-----------|-------|--------------|
| base | Linear Regression        | -                | 1.28e+05  | 2.19e+10    | 1.48e+05  | 0.383 | 0.383        |
| base | L1 Regularisation Lasso  | alpha=2.16e-04   | 1.28e+05  | 2.21e+10    | 1.49e+05  | 0.380 | 0.379        |
| base | L2 Regularisation Ridge  | alpha=1000.0     | 1.28e+05  | 2.21e+10    | 1.49e+05  | 0.379 | 0.379        |
| base | Polynomial (degree=2)    | degree=2         | 4.55e+04  | 4.33e+09    | 6.58e+04  | 0.878 | 0.878        |
| post | Linear Regression        | -                | 4.72e+04  | 3.90e+09    | 6.24e+04  | 0.899 | 0.899        |
| post | L1 Regularisation Lasso  | alpha=2.19e-04   | 4.70e+04  | 3.89e+09    | 6.23e+04  | 0.900 | 0.900        |
| post | L2 Regularisation Ridge  | alpha=100.0      | 4.71e+04  | 3.89e+09    | 6.24e+04  | 0.900 | 0.899        |
| post | Polynomial (degree=2)    | degree=2         | 4.39e+04  | 3.91e+09    | 6.26e+04  | 0.899 | 0.899        |

---
## KNN

| Data | Model                  | Hyperparam                                 | MAE       | MSE         | RMSE      | R²    | Adjusted R² |
|------|------------------------|---------------------------------------------|-----------|-------------|-----------|-------|--------------|
| base | KNN + Linear Regression | n=5, weights=distance                      | 5.85e+04  | 6.18e+09    | 7.86e+04  | 0.826 | 0.826        |
| post | KNN + Linear Regression | n=7, weights=distance                      | 4.55e+04  | 3.97e+09    | 6.30e+04  | 0.897 | 0.897        |

---
## Decision Tree

| Data | Model             | Hyperparam                                                                 | Train MSE     | Test MSE      | Test/Train MSE Ratio |
|------|-------------------|----------------------------------------------------------------------------|---------------|---------------|-----------------------|
| base | Decision Tree     | max_depth=29, min_split=13, min_leaf=3                                     | 9.57e+08      | 7.26e+09      | 7.58                  |
| base | Gradient Boosting | lr=0.162, max_depth=9, min_leaf=3, min_split=5, n_estimators=236           | 8.59e+08      | 5.64e+09      | 6.57                  |
| post | Decision Tree     | max_depth=25, min_split=17, min_leaf=3                                     | 1.44e+09      | 5.35e+09      | 3.73                  |
| post | Gradient Boosting | lr=0.0737, max_depth=9, min_leaf=5, min_split=2, n_estimators=253          | 1.06e+09      | 3.40e+09      | 3.21                  |

---
## Random Forest
The values have to be logged first.

| Data | Model         | Hyperparam                                                                      | Train MSE | Test MSE | Test/Train MSE Ratio |
|------|---------------|-------------------------------------------------------------------------------- |-----------|----------|-----------------------|
| base | Random Forest | max_depth=39, max_features=log2, min_leaf=1, min_split=5, n_est=149             | 0.002004  | 0.015404 | 7.69                  |
| post | Random Forest | max_depth=39, max_features=log2, min_leaf=1, min_split=5, n_est=149             | 0.001759  | 0.008635 | 4.91                  |

---
