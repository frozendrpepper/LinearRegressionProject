# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:52:59 2017

@author: cck3
"""
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import Imputer
import seaborn as sns


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def apply_imputer(style):
    pass

'''Import the train and test data set'''
data_train = pd.read_csv('train.csv')

X = data_train.iloc[:, :291]
y = data_train.iloc[:, 291]

missing_data_count = data_train.isnull().sum()

'''Let us at least attempt to draw the correlation coefficient heat map'''
corrdata = data_train.iloc[:, 2:]
corrmatrix = corrdata.corr()

'''Cut out id/timestamp and separate the data into categorical and numerical data'''
id_timestamp_data = data_train.loc[:, ['id', 'timestamp']]

'''Categorical data preprocessing'''
categorical_list =['material', 'state', 'product_type', 'sub_area', 'ID_metro', 'ID_railroad_station_walk',
                   'ID_railroad_station_avto', 'ID_big_road1', 
                   'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal', 'ecology', 
                   'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
                   'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 
                   'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']
categorical_data = data_train.loc[:,categorical_list]

delete_list = categorical_list + ['id', 'timestamp', 'culture_objects_top_25']
numerical_data = X.copy()
for item in delete_list:
    del numerical_data[item]

'''For some numbers, the data cannot take decimal points. In this case, we might have to use
median instead of mean for imputation 138'''