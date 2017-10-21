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
from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrix

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def convert_column(data_train):
     data_train_columns = list(data_train.columns.values)
     for i in range(len(data_train_columns)):
         if '-' in data_train_columns[i]:
             data_train_columns[i] = data_train_columns[i].replace('-', '_')
     return data_train_columns
    
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
    X.loc[ (X.build_year > 2015) | (X.build_year < 1600 ), 'build_year'] = np.nan
    X.loc[ (X.state > 4) | (X.state < 1), 'state'] = np.nan
    return X

def VIF_analysis(X):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif
     
'''Import the train and test data set'''
data_train = pd.read_csv('train.csv')

'''Some of the column names contain - signs which crashes the OLS algorithm'''
'''https://www.tutorialspoint.com/python/string_replace.htm'''
data_train_columns = convert_column(data_train)
data_train.columns = data_train_columns

X = data_train.iloc[:, 2:291]
y = data_train.iloc[:, 291]
    
'''This variable is used to check how much data is missing'''
missing_data_count = data_train.isnull().sum()

'''Perform encoding and mapping'''
X, encoder_dict = label_encoding(data_train.iloc[:, 2:])

'''Drop all the data with nan terms'''
X_drop = X.dropna()

'''Data description'''
data_describe = X_drop.describe()

'''build_year and state obviously need some modification'''
X_drop = simple_filter(X_drop)

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
target = 'price_doc'

filter_list = nom_categorical_list + ord_categorical_list + [target]
numeric_list = list(X_drop.columns)
for item in filter_list:
    numeric_list.remove(item)
    
'''Re-partitioning the DataFrame numeric -> ord_cat -> nom_cat'''
X_drop = pd.concat([X_drop[numeric_list], X_drop[ord_categorical_list], 
                    X_drop[nom_categorical_list], X_drop[target]], axis = 1)

'''Apply scaling to numeric value and declare a variable for the new standardized matrix'''
sc = StandardScaler()
X_drop_sc = X_drop[:]
X_drop_sc[numeric_list] = sc.fit_transform(X_drop_sc[numeric_list])

'''VIF analysis for features that are highly correlated'''
vif_metro = VIF_analysis(X_drop_sc[['metro_min_avto', 'metro_km_avto', 'metro_km_walk', 'metro_min_walk']])
vif_railroad_avto = VIF_analysis(X_drop_sc[['railroad_station_avto_min', 'railroad_station_avto_km']])
vif_railroad_walk = VIF_analysis(X_drop_sc[['railroad_station_walk_min', 'railroad_station_walk_km']])
vif_population = VIF_analysis(X_drop_sc[['raion_popul', 'children_school', 'children_preschool', 'full_all', 'male_f', 
                           'female_f', 'young_all', 'young_male', 'young_female', 'work_all', 'work_male', 
                           'work_female', 'ekder_all', 'ekder_male', 'ekder_female', '0_6_all', '0_6_male', 
                           '0_6_female', '7_14_all', '7_14_male', '7_14_female', '0_17_all', '0_17_male', 
                           '0_17_female', '16_29_all', '16_29_male', '16_29_female', '0_13_all', '0_13_male',
                           '0_13_female']])

'''Based on VIF analysis following features will be dropped'''
vif_delete_list = list(vif_metro.loc[vif_metro['VIF Factor'] > 30, 'features']) + \
                  list(vif_population.loc[vif_population['VIF Factor'] > 10**7 , 'features']) + \
                  ['railroad_station_avto_km', 'railroad_station_walk_km']

'''Drop above features from X_drop and update the numeric list'''
'''https://stackoverflow.com/questions/28538536/deleting-multiple-columns-in-pandas'''
X_drop.drop(vif_delete_list, axis = 1, inplace = True)
numeric_list_vif = numeric_list[:]
for item in vif_delete_list:
    numeric_list_vif.remove(item)

'''Compile a string that will be fed into sm.OLS.from_formula'''
'''Certain features have to be combined, so I'll take care of them first'''
spec_handle_cat = ['ID_metro', 'ID_railroad_station_walk', 'ID_railroad_station_avto', 'water_1line',
                   'ID_big_road1', 'ID_big_road2', 'ID_railroad_terminal', 'ID_bus_terminal']
spec_handle_real = ['metro_min_avto', 'railroad_station_walk_min', 'railroad_station_avto_min', 
                    'water_km', 'big_road1_km', 'big_road2_km', 'railroad_km', 'bus_terminal_avto_km']
    
'''This is for special combined continuous and category list'''
spec_handle_list = []
for cat, real in zip(spec_handle_cat, spec_handle_real):
    spec_handle_list.append("C(" + cat + "):" + real)
'''Don't forget to add this!'''
spec_handle_string = " + ".join(spec_handle_list)

'''Get rid of values that were used in the special combined variable'''
numeric_list_OLS = numeric_list_vif[:]
nom_list_OLS = nom_categorical_list[:]
for cat, real in zip(spec_handle_cat, spec_handle_real):
    numeric_list_OLS.remove(real)
    nom_list_OLS.remove(cat)
 
'''Handling numeric and ord categorical information. Add this too!'''
numeric_ord_cat_OLS_string = " + ".join(numeric_list_OLS + ord_categorical_list)

'''Handling the nominal categorical information with C()'''
category_OLS_list = []
for item in nom_list_OLS:
    category_OLS_list.append("C(" + item + ")")
category_OLS_string = " + ".join(category_OLS_list) 

'''This is the total string'''
OLS_string = spec_handle_string + " + " + numeric_ord_cat_OLS_string + " + " + category_OLS_string


'''Finally first OLS run'''
'''
OLS_string_final = "price_doc ~ " + OLS_string
model1 = sm.OLS.from_formula(OLS_string_final, data=X_drop)
result1 = model1.fit()
print(result1.summary())
'''