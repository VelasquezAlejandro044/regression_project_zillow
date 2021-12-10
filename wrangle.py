import pandas as pd
import os
from env import host, user, password


#acquires zillow dataset
def get_zillow_data():
    return pd.read_sql(sql,get_connection('zillow'))

import pandas as pd
import os
from env import host, user, password

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

def remove_outliers(df, k, col_list):
    ''' 
    
    Here, we remove outliers from a list of columns in a dataframe and return that dataframe
    
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def clean_and_prep(df):
    # drop fips and transaction_date
    # dropping nulls
    df = df.drop
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'year', 'tax_of_property', 'tax_assessed_value_target', 'zip_code'])

