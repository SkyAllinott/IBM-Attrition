# Predicting IBM Attrition
## Overview
I predict whether an IBM employee quits (attrition) using Lasso, randomForest, and XGBoost models. I utilize cross validation to tune model parameters.

## Results
I plot the ROC curve for each model below. A model with no predictive power is a 45 degree line. A model with perfect predictive accuracy traces the left and top axes.

![IBM](https://user-images.githubusercontent.com/52394699/177026295-ad483883-f1e0-4d2f-ab0e-9a68daacc370.png)

The area under the ROC curve (AUC), is a measure of model power, with higher values indicating better models.

The boosting model outperforms the others, and randomForest performs the worst. 

## Important Variables
Lasso suggests that overtime hours, job role, and satisfaction are the most important indicator of if an employee will quit.

RandomForest say that monthly income, overtime hours, and distance from home are the most important.

XGBoost suggests that job role, overtime, and monthly income is the most important.
