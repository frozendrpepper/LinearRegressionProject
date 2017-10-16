# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:52:59 2017

@author: cck3
"""
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def apply_imputer(style):
    pass

'''Import the train and test data set'''
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

X = data_train.iloc[:, :291]
y = data_train.iloc[:, 291]
'''Cut out id/timestamp and separate the data into categorical and numerical
data'''
id_timestamp_data = data_train.loc[:, ['id', 'timestamp']]

'''Categorical data preprocessing'''
categorical_list =['product_type', 'sub_area', 'thermal_power_plant_raion', 'incineration_raion',
                   'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion',
                   'nuclear_reactor_raion', 'detention_facility_raion', 'ID_metro', 'ID_railroad_station_walk',
                   'ID_railroad_station_avto', 'water_1line', 'ID_big_road1', 'big_road1_1line',
                   'ID_big_road2', 'railroad_1line', 'ID_railroad_terminal', 'ID_bus_terminal', 'ecology']
categorical_data = data_train.loc[:,categorical_list]

'''Extract unique elements in each category column so we can use it to
apply imputer class'''
categorical_data['product_type'].unique()
#numerical_list = []
#numerical_data = data_train.loc[:, ]

'''From the original feature matrix, get rid of the categorical columns and 
columns that are unnecessary'''
X_copy = X.copy()
get_rid_of_list = categorical_list + ['id', 'timestamp', 'culture_objects_top_25']
for i in get_rid_of_list:
    del X_copy[i]

missing_data_copy = X_copy.isnull().sum()

'''This is the label, or y data'''
target_data = data_train.iloc[:, -1]
'''The first step is to get rid of data with NaN values and see if we still
have enough dataset to work with.

Update 10/03

Not sure if this is a good idea since it gets rid of too many data points
'''

#filtered_data_train = data_train.dropna(axis = 0)
#filtered_data_test = data_test.dropna(axis = 0)

'''Separate the train data into feature and X, and then split it into train/test set'''
#X = filtered_data_train.iloc[:, :291]
#y = filtered_data_train.iloc[:, -1]