# Disable warnings

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from pydataset import data

# acquire
from env import host, user, password

# visualize
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(11, 9))
plt.rc('font', size=13)

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing


###############acquire#################


def get_connection(database, user=user, host=host, password=password):
    '''get URL with user, host, and password from env '''
    
    return f"mysql+pymysql://{user}:{password}@{host}/{database}"
    
    
def cache_sql_data(df, database):
    '''write dataframe to csv with title database_query.csv'''
    
    df.to_csv(f'{database}_query.csv',index = False)
        

def get_sql_data(database,query):
    ''' check if csv exists for the queried database
        if it does read from the csv
        if it does not create the csv then read from the csv  
    '''
    
    if os.path.isfile(f'{database}_query.csv') == False:   # check for the file
        
        df = pd.read_sql(query, get_connection(database))  # create file 
        
        cache_sql_data(df, database) # cache file
        
    return pd.read_csv(f'{database}_query.csv') # return contents of file


def get_zillow_data():
    ''' acquire zillow data'''
    
    query = '''

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

    '''

    database = "zillow"
    
    # create/read csv for query
    
    df = get_sql_data(database,query) 
    
    # drop duplicate parcelids keeping the latest
    
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last') 
    
    return df 

#################################prepare###############################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        return df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]  
    

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
        take in a dataframe and a proportion for columns and rows
        return dataframe with columns and rows not meeting proportions dropped
    '''
    col_thresh = int(round(prop_required_column*df.shape[0],0)) # calc column threshold
    
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    row_thresh = int(round(prop_required_row*df.shape[1],0))  # calc row threshhold
    
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df    
    
    
def impute(df, my_strategy, column_list):
    ''' take in a df strategy and cloumn list
        return df with listed columns imputed using input stratagy
    '''
        
    imputer = SimpleImputer(strategy=my_strategy)  # build imputer

    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns

    return df

def prepare_zillow(df):
    ''' Prepare Zillow Data'''
    
    # Restrict propertylandusedesc to those of single unit
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
          (df.propertylandusedesc == 'Mobile Home') |
          (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
          (df.propertylandusedesc == 'Townhouse')]
    
    # remove outliers in bed count, bath count, and area to better target single unit properties
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])
    
    # dropping cols/rows where more than half of the values are null
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)
    
    # dropping the columns with 17K missing values too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt', 'heatingorsystemdesc'])
    
    # imputing descreet columns with most frequent value
    df = impute(df, 'most_frequent', ['calculatedbathnbr', 'fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock'])
    
    # imputing continuous columns with median value
    df = impute(df, 'median', ['finishedsquarefeet12', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount'])
    
    return df
