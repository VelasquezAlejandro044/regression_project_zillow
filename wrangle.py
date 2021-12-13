import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
from sklearn.impute import SimpleImputer
import scipy
# import acquire
# import prepare
from scipy import stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")

#sklearn imports
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# -------------------------- Aquiere -------------------------------------------------
# -------------------------- Aquiere -------------------------------------------------
def get_connection(db_name):
    from env import user, host, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


# function contains the sql needed and to return a pands data frame of the zillow data
# SQL Query bringing all info needed for 2017 transactions 
def get_zillow_data():
    '''
    This function reads the Telco data from the Codeup db into a df.
    '''
    sql = """
    SELECT bedroomcnt as bedrooms, 
           bathroomcnt as bathrooms,
           calculatedfinishedsquarefeet as square_feet,
           yearbuilt as year,
           taxamount as tax_of_property,
           taxvaluedollarcnt as tax_assessed_value_target,
           fips as fips,
           regionidzip as zip_code,
           transactiondate as transaction_date
    FROM predictions_2017
    JOIN properties_2017 USING(parcelid)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE transactiondate < '2018-01-01' AND
    transactiondate > '2016-12-31' AND
    propertylandusetypeid LIKE '261'
    ORDER BY fips;
    """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql, get_connection('zillow'))
    
    return df

# ------------------------------------ Prepare -----------------------------------------

def prepare_zillow(df):
    # convert treansaction_date to date object type
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    # Create value counts to see upper outliers
    cols = df.columns.values
    for col in cols:
        print(col.upper())
        print(df[col].value_counts(dropna=False,ascending=True).head(10))



def remove_outliers(df, k, col_list):
    ''' 
    
    Remove outliers from a list of columns in a pd.df and return it
    
    '''

    # #  bring all recors expet the 15649
    # dfcopy[dfcopy.index != 15643]
    # # rewrites the dataframe withot record
    # dfcopy = dfcopy[dfcopy.index != 15643]
    # cleane a few manualy 
    df  = df[~df.index.isin([ 10862, 15643, 36707, 45004, 50245])]

    # dropping nulls
    df = df.dropna()

    # drop transaction date
    df = df.drop(labels=["transaction_date"], axis=1)

    # Eliminate properties with 0 bathrooms 
    # This eleminates 42 properties
    df = df[(df.bathrooms != 0)]

    # Drop houses smaller than 200 
    df = df[df['square_feet'] > 300]

    # Use z score to detect outliers
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    df = df[filtered_entries]
    df.shape

    #correct data type
    df.year = df.year.astype(int)
    df.fips = df.fips.astype(int)
    df.zip_code = df.zip_code.astype(int)

    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def clean_and_prep(df):

    """
    function prepares data and clean outliers
    """
    # clean outliers identified by Alejandro Velasquez
    df  = df[~df.index.isin([ 10862, 15643, 36707, 45004, 50245])]
  

    # drop fips and transaction_date
    df = df.drop(labels=["transaction_date"], axis=1)
    
    # dropping nulls
    df = df.dropna()

    # Drop properties that don't have bathrooms
    # 41 records
    df = df[(df.bathrooms != 0)]

    # Drop houses smaller than 200 
    df = df[df['square_feet'] > 300]

    # Change rows that have complete units from float to interger
    df.year = df.year.astype(int)
    df.fips = df.fips.astype(int)
    df.zip_code = df.zip_code.astype(int)
    
    # Use z_score to remove outliers
    # Elinminates 1636 
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis = 1)
    df = df[filtered_entries]

    # remove outliers using a function define in this module as remove_outliers
    # 6139 outliers removed
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'year', 'tax_of_property', 'tax_assessed_value_target', 'zip_code'])

# Split the data
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state= 42)
    train, validate = train_test_split(train_validate, test_size=.3, random_state= 42)

    return train, validate, test
    
# ------------------------------------ Scale -----------------------------------------

def robust_scaler_viz(X_train, X_train_scaled):
    #plot the scaled and unscaled distributions
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, bins=5, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, bins=5, ec='black')
    plt.title('Scaled')



def data_to_samples(features, train, validate, test):
    # split data into samples
    # Note that we only call .fit with the training data,
    X_train = train[features]
    y_train = train.tax_assessed_value_target
    
    x_validate = validate[features]
    y_validate = validate.tax_assessed_value_target
    
    x_test = test[features]
    y_test = test.tax_assessed_value_target

    return X_train, y_train, x_validate, y_validate, x_test, y_test