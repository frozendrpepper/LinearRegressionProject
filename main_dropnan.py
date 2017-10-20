# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 23:37:20 2017

@author: cck3
"""

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

from patsy import dmatrix

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

 
    
def label_encoding(X):
    '''Label Encoding and mapping'''
    
    #sub_area encoding
    sub_area_le = LabelEncoder()
    X['sub_area'] = sub_area_le.fit_transform(X['sub_area'])
    
    #product_type encoding
    product_type_le = LabelEncoder()
    X['product_type'] = product_type_le.fit_transform(X['product_type'])
    
    #All the yes_no encoding
    yes_no_le = LabelEncoder()
    yes_no_list = ['thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
       'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 
       'water_1line', 'big_road1_1line', 'railroad_1line', 'culture_objects_top_25']
    for column in yes_no_list:
        X[column] = yes_no_le.fit_transform(X[column])
    
    #ecology mapping
    X['ecology'].unique()
    ecology_mapping = {'excellent' : 4, 'good' : 3, 'satisfactory' : 2, 'poor' : 1, 'no data' : np.nan}
    X['ecology'] = X['ecology'].map(ecology_mapping)
    
    encoder_dict = {'sub_area' : sub_area_le, 'product_type' : product_type_le, 'yes_no' : yes_no_le,
                    'ecology' : ecology_mapping}  
    return X, encoder_dict

def simple_filter(X):
    X_drop.loc[ (X_drop.build_year > 2015) | (X_drop.build_year < 1600 ), 'build_year'] = np.nan
    X_drop.loc[ (X_drop.state > 4) | (X_drop.state < 1), 'state'] = np.nan
    return X
     
'''Import the train and test data set'''
data_train = pd.read_csv('train.csv')

X = data_train.iloc[:, 2:291]
y = data_train.iloc[:, 291]
    
'''This variable is used to check how much data is missing'''
missing_data_count = data_train.isnull().sum()

'''Perform encoding and mapping'''
X, encoder_dict = label_encoding(X)

'''Drop all the data with nan terms'''
X_drop = X.dropna()
y_drop= y.dropna()

'''Data description'''
data_describe = X_drop.describe()

'''build_year and state obviously need some modification'''
X = simple_filter(X)

'''Once again drop the nan terms that were filtered, and look at the data 
description to make sure the filtration has been carried out properly'''
'''Reference: https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame'''
X_drop = X_drop.dropna().reset_index(drop = True)
data_describe = X_drop.describe()

'''Chek and see if all values are numerical'''
'''Reference: https://stackoverflow.com/questions/21771133/finding-non-numeric-rows-in-dataframe-in-pandas'''
print(X_drop[~X_drop.applymap(np.isreal).all(1)])


'''Create columns corresponding to different data type'''
nom_categorical_list =['material', 'product_type', 'sub_area', 'ID_metro', 'ID_railroad_station_walk',
                   'ID_railroad_station_avto', 'ID_big_road1', 
                   'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal',  
                   'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
                   'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 
                   'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 
                   'culture_objects_top_25']
ord_categorical_list = ['state', 'ecology']

filter_list = nom_categorical_list + ord_categorical_list
numeric_list = list(X_drop.columns)
for item in filter_list:
    numeric_list.remove(item)
'''Re-partitioning the DataFrame'''
X_drop = pd.concat([X_drop[numeric_list], X_drop[ord_categorical_list], X_drop[nom_categorical_list]], axis = 1)

'''Apply scaling to numeric value'''
sc = StandardScaler()
X_drop[numeric_list] = sc.fit_transform(X_drop[numeric_list])

