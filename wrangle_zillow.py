# Disable warnings

import warnings
warnings.filterwarnings("ignore")

# Import libraries 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Import to obtain data from Codeup SQL databse server

from env import host, user, password

# Library dealing with NA values

from sklearn.impute import SimpleImputer

# connection url

# This function uses my info from the env file to create a connection url to access the Codeup db.
# It takes in a string name of a database as an argument.

def get_connection(db, user=user, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

###########################################################################
############################# get zillow data #############################
def get_zillow_data():
    '''
    This function reads in Zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    sql_query = """
SELECT prop.*,
       pred.logerror,
       pred.transactiondate,
       air.airconditioningdesc,
       arch.architecturalstyledesc,
       build.buildingclassdesc,
       heat.heatingorsystemdesc,
       landuse.propertylandusedesc,
       story.storydesc,
       construct.typeconstructiondesc
FROM   properties_2017 prop
       INNER JOIN (SELECT parcelid,
                   Max(transactiondate) transactiondate
                   FROM   predictions_2017
                   GROUP  BY parcelid) pred
               USING (parcelid)
               			JOIN predictions_2017 as pred USING (parcelid, transactiondate)
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
       LEFT JOIN storytype story USING (storytypeid)
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE  prop.latitude IS NOT NULL
       AND prop.longitude IS NOT NULL
                """
    if os.path.isfile('zillow_data.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = pd.read_sql(sql_query, get_connection('zillow'))
        
        # Cache data
        df.to_csv('zillow_data.csv')

    return df

##########################################################

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

##########################################################

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

############################################################

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3).to_markdown())
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe().to_markdown())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

#######################################################################

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df



def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df


# combining everything in a cleaning function:

def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

###############################################################################
# Function to remove outliers in dataframe

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df  

# Function that combines the above functions

def wrangle_zillow():
    '''
    We will call the other functions from the file in our wrangle_zillow function.
    '''
    #Acquire data
    df = get_zillow()

    #Clean data
    df = impute_null_values(df)

    #Remove outliers
    df = remove_outliers(df, 2.5, df.columns)

    #Return DataFrame
    return df

#####################################################################################
# Replace the null values with the mean value of each column

def impute_null_values():
    '''
    Using SimpleImputer to impute the mean value into the null values into each column.
    '''
    #Using the mean imputer function

    imputer = SimpleImputer(strategy='mean')

    # Create a for loop that will impute all the null values in each one of our columns.

    for col in df.columns:
        df[[col]] = imputer.fit_transform(df[[col]])
    return df

