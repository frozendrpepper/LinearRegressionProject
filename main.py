# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:52:59 2017

@author: cck3
"""
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def mode(a):
    u, c = np.unique(a, return_counts=True)
    return u[c.argmax()]

def build_year_impute(fnc):
     '''A lot of these codes are there to modify the data format so that methods
     function properly'''
     class_le = LabelEncoder()
     sub_area_le = nom_categorical_data['sub_area'].values
     sub_area_le = np.array([class_le.fit_transform(sub_area_le)]).T
     sub_area_le = pd.DataFrame(sub_area_le)
     build_year_impute_df = pd.concat([sub_area_le, data_train['build_year']], axis = 1)
     build_year_impute_df.columns = ['sub_area', 'build_year']
     '''Thish groups the dataframe via sub_area and applies mode function'''
     build_year_impute_df_group = build_year_impute_df.groupby('sub_area', as_index = False)['build_year'].apply(fnc)
    
     '''Some data is even missing even at this point. I'll just impute with median le sigh'''
     temp2 = build_year_impute_df_group[build_year_impute_df_group < 100]
     build_year_impute_df_group[list(temp2.index)] = build_year_impute_df_group.median()
     return build_year_impute_df_group, build_year_impute_df

'''Import the train and test data set'''
data_train = pd.read_csv('train.csv')

X = data_train.iloc[:, :291]
y = data_train.iloc[:, 291]

'''This variable is used to check how much data is missing'''
missing_data_count = data_train.isnull().sum()

'''Computing the correlation coefficient matrix'''
corrdata = data_train.iloc[:, 2:]
corrmatrix = corrdata.corr()

'''Categorical data preprocessing'''
categorical_list =['material', 'product_type', 'sub_area', 'ID_metro', 'ID_railroad_station_walk',
                   'ID_railroad_station_avto', 'ID_big_road1', 
                   'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal', 'ecology', 
                   'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
                   'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 
                   'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 
                   'culture_objects_top_25']
ord_categorical_data = data_train['state']
ord_categorical_data_missing = ord_categorical_data.isnull().sum()
nom_categorical_data = data_train.loc[:, categorical_list]
nom_categorical_data_missing = nom_categorical_data.isnull().sum()

'''Imputing categorical data'''
'''Let us first work on build year imputation. Fill it in with the most commonly
occuring build year in that sub_area'''
#build_year_filter, build_year_impute_df = build_year_impute(mode)
#build_year_mapping = pd.Series.to_dict(build_year_filter)
#build_year_impute_df['build_year_filter'] = np.where(str(build_year_impute_df['build_year'])=='nan', build_year_mapping[build_year_impute_df['sub_area']], build_year_impute_df['build_year'])


    


#ohe = OneHotEncoder()
#temp3  = ohe.fit_transform(temp2).toarray()


'''Compile numerical data'''
filter_list = categorical_list + ['id', 'timestamp']
numerical_data = X.copy().drop(filter_list, axis = 1)

'''For some numbers, the data cannot take decimal points. In this case, we might have to use
median instead of mean for imputation 138'''