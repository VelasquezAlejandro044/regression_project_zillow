
   
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard Libraries: 
import pandas as pd
import numpy as np 

# Visuals:
import matplotlib.pyplot as plt
import seaborn as sns

# Stats:
from scipy import stats

# Splitting
from sklearn.model_selection import train_test_split

# Modeling
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import TweedieRegressor
import sklearn.preprocessing

# My Files
from env import host, user, password
import wrangle
import explore

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# ------------------------------Basae Line ------------------------

def baseline(y_train, y_validate):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1 Predict mean
    pred_mean = y_train.tax_assessed_value_target.mean()
    y_train['baseline_pred_mean'] = pred_mean
    y_validate['baseline_pred_mean'] = pred_mean
    
    # 2. Predict median
    pred_median = y_train.tax_assessed_value_target.median()
    y_train['baseline_pred_median'] = pred_median
    y_validate['baseline_pred_median'] = pred_median
    
    # 3. RMSE of mean
    rmse_train = mean_squared_error(y_train.tax_assessed_value_target, y_train.baseline_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_assessed_value_target, y_validate.baseline_pred_mean)**(1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # 4. RMSE of median
    rmse_train = mean_squared_error(y_train.tax_assessed_value_target, y_train.baseline_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_assessed_value_target, y_validate.baseline_pred_median)**(1/2)
    
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # building a df of our metrics for model selsection 
    metric_df = pd.DataFrame(data=[{'model': 'mean_baseline', 'RMSE_train': rmse_train, 'RMSE_validate': rmse_validate}])
    return metric_df


# ------------------------------Compute Resifual ------------------------
def residual(df):
    df['residual'] = df['y_hat'] - df['tax_assessed_value_target']
    df['residual_baseline'] = df['yhat_baseline'] - df['tax_assessed_value_target']
    return df