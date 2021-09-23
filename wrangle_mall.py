# Disable warnings

import warnings
warnings.filterwarnings("ignore")

# Import libraries 

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import seaborn as sns

# Library dealing with NA values

from sklearn.impute import SimpleImputer

# acquire
from env import host, user, password
import wrangle_mall

# Imports

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

######################################################################################

# connection url

# This function uses my info from the env file to create a connection url to access the Codeup db.
# It takes in a string name of a database as an argument.

def get_connection(db, user=user, host=host, password=password):
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

####################################################################################### 
### Function takes in a database and query check if csv exists, writes data to
### a csv file if a local file does not exist, and returns a df

def get_sql_data(database,query):

    if os.path.isfile(f'{database}.csv') == False:   # check for the file
        
        df = pd.read_sql(query, get_connection(database))  # create file 
        
        df.to_csv(f'{database}.csv',index = False)   # cache file
        
    return pd.read_csv(f'{database}.csv') # return contents of file

#######  This function reads in mall_customers data from Codeup database

def get_mall_data():
    ''' acquire data from mall_customers database'''
    
    database = "mall_customers"

    query = "select * from customers"

    df = get_sql_data(database,query)
    
    return df.set_index('customer_id')

############## Show Nulls by Column

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

############## Show Nulls by Row

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

#############

def summarize(df):
    '''
    This function will take in a single argument (pandas DF)
    and output to console various statistics on said DF, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observe null values
    '''
    print('----------------------------------------------------')
    print('DataFrame Head')
    print(df.head(3))
    print('----------------------------------------------------')
    print('DataFrame Info')
    print(df.info())
    print('----------------------------------------------------')
    print('DataFrame Description')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------------------------------------')
    print('DataFrame Value Counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
            print('--------------------------------------------')
            print('')
        else:
            print(df[col].value_counts(bins=10, sort=False))
            print('--------------------------------------------')
            print('')
    print('----------------------------------------------------')
    print('Nulls in DataFrame by Column: ')
    print(nulls_by_col(df))
    print('----------------------------------------------------')
    print('Nulls in DataFrame by Rows: ')
    print(nulls_by_row(df))
    print('----------------------------------------------------')
    df.hist()
    plt.tight_layout()
    plt.show()
    
    df['gender'].value_counts().plot(kind='bar', title = f"{col} distribution")


#################### Detect outliers using IQR

def detect_outliers(df, k, col_list):
    ''' get upper and lower bound for list of columns in a dataframe 
        if desired return that dataframe with the outliers removed
    '''
    
    outliers = pd.DataFrame()
    
    for col in col_list:

        q1, q2, q3 = df[f'{col}'].quantile([.25, .5, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        
        # print each col and upper and lower bound for each column
        print(f"{col}: Median = {q2} lower_bound = {lower_bound} upper_bound = {upper_bound}")

        # return dataframe of outliers
        outliers = outliers.append(df[(df[f'{col}'] < lower_bound) | (df[f'{col}'] > upper_bound)])
            
    return outliers

# will show outliers for mall data
# outliers = detect_outliers(df, 1.5,['age', 'annual_income', 'spending_score'])

#################### Upper Outliers

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

################## Upper Outliers Columns

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

################# Lower Outliers

def get_lower_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x:max([x - lower_bound, 0]))

################ Lower Outliers Columns

def add_lower_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    return df

################ Describe Outliers

def outlier_describe(df):
    outlier_cols = [col for col in df.columns if col.endswith('_outliers')]
    for col in outlier_cols:
        print(col, ': ')
        subset = df[col][df[col] > 0]
        print(subset.describe())

################ Function to remove outliers in dataframe

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

################### Encode categorical columns

def encoding(df, cols, drop_first=True):
    '''
    Take in df and list of columns
    add encoded columns derived from columns in list to the df
    '''
    for col in cols:

        dummies = pd.get_dummies(df[f'{col}'], drop_first=drop_first) # get dummy columns

        df = pd.concat([df, dummies], axis=1) # add dummy columns to df
        
    return df

####################  Split data (train, validate, and test split)

def split_data(df):
    '''split df into train, validate, test'''
    
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test

############################ Scale Data

def min_max_scaling(train, validate, test, num_cols):
    '''
    Add scaled versions of a list of columns to train, validate, and test
    '''
    
    # reset index for merge 
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    scaler = sklearn.preprocessing.MinMaxScaler() # create scaler object

    scaler.fit(train[num_cols]) # fit the object 

    # transform to get scaled columns
    train_scaled = pd.DataFrame(scaler.transform(train[num_cols]), columns = train[num_cols].columns + "_scaled")
    validate_scaled = pd.DataFrame(scaler.transform(validate[num_cols]), columns = validate[num_cols].columns + "_scaled")
    test_scaled = pd.DataFrame(scaler.transform(test[num_cols]), columns = test[num_cols].columns + "_scaled")
    
    # add scaled columns to dataframes
    train = train.merge(train_scaled, left_index=True, right_index=True)
    validate = validate.merge(validate_scaled, left_index=True, right_index=True)
    test = test.merge(train_scaled, left_index=True, right_index=True)
    
    return train_scaled, validate_scaled, test_scaled

# to run for mall data
# train_scaled, validate_scaled, test_scaled = min_max_scaling(train, validate, test, ['age', 'annual_income', 'spending_score'])

#############################

def wrangle_mall_data():
    col_list = ['age', 'annual_income', 'spending_score']
    # let's acquire our data...
    df = get_mall_data()
    # summarize the data
    print(summarize(df))
    # add upper outlier columns
    df = add_upper_outlier_columns(df)
    # add lower outlier columns
    df = add_lower_outlier_columns(df)
    # describe the outliers
    print(outlier_describe(df))
    # drop outliers
    df = remove_outliers(df, 1.5, col_list)
    # split the data
    train, validate, test = split_data(df)
    # drop missing values from train
    train = train.dropna()
    # scale the data
    train_scaled, \
    validate_scaled, \
    test_scaled = min_max_scaling(train, validate, test, col_list)
    print(f'          train shape: {train.shape}')
    print(f'       validate shape: {validate.shape}')
    print(f'           test shape: {test.shape}')
    print(f'   train_scaled shape: {train_scaled.shape}')
    print(f'validate_scaled shape: {validate_scaled.shape}')
    print(f'    test_scaled shape: {test_scaled.shape}')
    
    #######################################################

def viz_iris(iris, kmeans):
    
    centroids = np.array(iris.groupby('cluster')['petal_length', 'sepal_length'].mean())
    cen_x = [i[0] for i in centroids]
    cen_y = [i[1] for i in centroids]
    # cen_x = [i[0] for i in kmeans.cluster_centers_]
    # cen_y = [i[1] for i in kmeans.cluster_centers_]
    iris['cen_x'] = iris.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    iris['cen_y'] = iris.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

    colors = ['#DF2020','#2095DF', '#81DF20' ]
    iris['c'] = iris.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    #plot scatter chart for Actual species and those predicted by K - Means

    #specify custom palette for sns scatterplot
    colors1 = ['#2095DF','#81DF20' ,'#DF2020']
    customPalette = sns.set_palette(sns.color_palette(colors1))

    #plot the scatterplots

    #Define figure (num of rows, columns and size)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # plot ax1 
    ax1 = plt.subplot(2,1,1) 
    sns.scatterplot(data = iris, x = 'petal_length', y = 'sepal_length', ax = ax1, hue = 'species', palette=customPalette)
    plt.title('Actual Species')

    #plot ax2
    ax2 = plt.subplot(2,1,2) 
    ax2.scatter(iris.petal_length, iris.sepal_length, c=iris.c, alpha = 0.6, s=10)
    ax2.set(xlabel = 'petal_length', ylabel = 'sepal_length', title = 'K - Means')

    # plot centroids on  ax2
    ax2.scatter(cen_x, cen_y, marker='X', c=colors, s=200)
    
    
    iris.drop(columns = ['cen_x', 'cen_y', 'c'], inplace = True)
    plt.tight_layout()
    plt.show()