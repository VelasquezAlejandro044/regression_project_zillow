import pandas as pd
import os
from env import host, user, password
import pandas as pd
import os
from env import host, user, password

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

def remove_outliers(df, k, col_list):
    ''' 
    
    Remove outliers from a list of columns in a pd.df and return it
    
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




    

