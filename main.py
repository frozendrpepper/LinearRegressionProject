# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:52:59 2017

@author: cck3
"""
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def mode(a):
    u, c = np.unique(a, return_counts=True)
    return u[c.argmax()]

def build_year_sub_area_impute(fnc, data, clf = None, mapping = None):
     '''A lot of these codes are there to modify the data format so that methods
     function properly'''
     if clf == None:
         class_le = LabelEncoder()
         '''Input to label encoder has to a numpy array and the returned object from fit_transform
         is a simple array object of size (21570,). This is then converted to a DataFrame object
         so I can concat it with build year data column'''
         sub_area_le = pd.DataFrame(class_le.fit_transform(data_train['sub_area']))
     else:
         class_le = clf
         sub_area_le = pd.DataFrame(class_le.transform(data_train['sub_area']))
     
     '''Simple concatanation and relabeling the columns. First column contains the encoded 
     sub_area information while the second column contains the build_year information'''
     build_year_impute_df = pd.concat([sub_area_le, data['build_year']], axis = 1)
     build_year_impute_df.columns = ['sub_area', 'build_year']

     '''Thish groups the dataframe via sub_area and applies mode function'''
     '''Explanation for this code is here:
     https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas'''
     build_year_impute_df_group = build_year_impute_df.groupby('sub_area', as_index = False)['build_year'].apply(fnc)
    
     '''build_year_impute_df_group at this point contains the mode corresponding to encoded sub_area
     Now we filterd out the data that are empty and just impute them with the median value'''
     comparison_df = build_year_impute_df_group[build_year_impute_df_group < 100]
     build_year_impute_df_group[list(comparison_df.index)] = build_year_impute_df_group.median()

     '''The build_year_mapping is for mapping in the conditional statement. This part took me hours to figure out'''
     '''row_index returns a list containg True or False depending on wehther the column
     values of build_year is nan or not. The reference for this code are
     https://stackoverflow.com/questions/18196203/how-to-conditionally-update-dataframe-column-in-pandas
     and data preprocessing portion of Sebastian Raschka's book'''
     if mapping == None:
         build_year_mapping = pd.Series.to_dict(build_year_impute_df_group)
         row_index = build_year_impute_df.build_year.isnull()
         build_year_impute_df.loc[row_index, 'build_year'] = build_year_impute_df.loc[row_index, 'sub_area'].map(build_year_mapping)
         return build_year_mapping, build_year_impute_df, sub_area_le, class_le
     else:
         build_year_mapping = mapping
         row_index = build_year_impute_df.build_year.isnull()
         build_year_impute_df.loc[row_index, 'build_year'] = build_year_impute_df.loc[row_index, 'sub_area'].map(build_year_mapping)
         return build_year_impute_df, sub_area_le, class_le
     

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
ord_categorical_data = pd.DataFrame(data_train['state'])
ord_categorical_data_missing = ord_categorical_data.isnull().sum()
nom_categorical_data = data_train.loc[:, categorical_list]
nom_categorical_data_missing = nom_categorical_data.isnull().sum()

'''Apparently, build_year data has some data that needs to be handled
https://stackoverflow.com/questions/21608228/conditional-replace-pandas'''
data_train.loc[ (data_train.build_year > 2015) | (data_train.build_year < 1600 ), 'build_year'] = np.nan


'''Imputing categorical data'''
'''Let us first work on build year and sub_area imputation. Fill it in with the most commonly
occuring build year in that sub_area as per idea of:
https://www.r-bloggers.com/a-data-scientists-guide-to-predicting-housing-prices-in-russia/'''
build_year_mapping, build_year_sub_area_impute_df, sub_area_df, clf = build_year_sub_area_impute(mode, data_train)


'''Next up is the state imputation which uses the information on build_year and sub_area
data that was just imputed. I guess we could do a small regression. The reference confirming that
multiple regression imputation is okay is given by: https://measuringu.com/handle-missing-data/'''
ohe = OneHotEncoder(categorical_features = [0])
build_year_sub_area_impute_encoded_df = build_year_sub_area_impute_df.copy()
build_year_sub_area_impute_encoded_df = pd.DataFrame(ohe.fit_transform(build_year_sub_area_impute_encoded_df).toarray())


state_impute_df = pd.concat([build_year_sub_area_impute_encoded_df, ord_categorical_data], axis = 1)
state_missing = state_impute_df.isnull().sum()

state_impute_aug_df = sm.add_constant(state_impute_df)
state_impute_aug_drop_df = state_impute_aug_df.dropna()
state_impute_dfx = state_impute_aug_drop_df.iloc[:, :148]
state_impute_dfy = pd.DataFrame(state_impute_aug_drop_df.iloc[:, -1])

state_impute_model = sm.OLS(state_impute_dfy, state_impute_dfx)
state_impute_result = state_impute_model.fit()
print(state_impute_result.summary())




'''Compile numerical data'''
filter_list = categorical_list + ['id', 'timestamp']
numerical_data = X.copy().drop(filter_list, axis = 1)

'''For some numbers, the data cannot take decimal points. In this case, we might have to use
median instead of mean for imputation 138'''