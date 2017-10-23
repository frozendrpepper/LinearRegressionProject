# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:19:59 2017

@author: cck3
"""

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

from patsy import dmatrix
from scipy import stats

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
    
def p_value_extract(result, confidence):
    '''https://stackoverflow.com/questions/37787698/how-to-sort-pandas-dataframe-from-one-column'''
    p_values = pd.DataFrame(result.pvalues)
    p_values.columns = ['p_value']
    p_values['p_value'] = p_values[p_values.p_value < confidence]
    p_values = p_values.dropna().sort_values('p_value')
    return p_values

    
def OLS_analysis(dfX, dfY, confidence, p_prev = 0, num_feature_compile = [], r2_compile = []):
    '''Function basically computes the result based on input dfX and dfY.
    Since this function will run multiple times for feature extraction, the
    adjusted R2 values are compiled for comparsion and the number of features
    that survive the p-value test is compared to the number from previous
    iteration. If the number of features cease to decrease, the analysis
    will terminate as well.'''
    end_loop = True
    model = sm.OLS(dfY, dfX)
    result = model.fit()
    r2_compile.append(result.rsquared_adj)
    
    p = p_value_extract(result, confidence)
    p_shape = p.shape[0]
    if int(p_shape) == int(p_prev):
        end_loop = False
  
    extracted_feature = list(p.index.values)
    num_feature_compile.append(extracted_feature)
    dfX_extract = dfX.loc[:, extracted_feature]
    return dfX_extract, result, r2_compile, num_feature_compile, p_shape, end_loop

def num_cat_divider(X, feature):
    X_numeric, X_cat = [], []
    for item in feature:
        if item == 'state':
            X_cat.append(item)
        elif len(X[item].unique()) > 2:
            X_numeric.append(item)
        else:
            X_cat.append(item)
    return X[X_numeric], X[X_cat]

def outlier_crusher(result):
    influence = result.get_influence()
    cooks_d2, pvals = influence.cooks_distance
    fox_cr = 4 / (len(dfY) - dfX2.shape[1] - 1)
    idx = np.where(cooks_d2 > fox_cr)[0]
    return idx
  
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

'''I've decided not to use the ID and sub_area category values'''
nom_categorical_list = ['material', 'product_type',  
                   'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
                   'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 
                   'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 
                   'culture_objects_top_25']
    
'''Re-partitioning the DataFrame numeric -> ord_cat -> nom_cat'''
X_drop = pd.concat([X_drop[numeric_list], X_drop[ord_categorical_list], 
                    X_drop[nom_categorical_list], X_drop[target]], axis = 1)

'''Apply scaling to numeric value and declare a variable for the new standardized matrix'''
sc = StandardScaler()
X_drop_sc = X_drop[:]
X_drop_sc[numeric_list + ord_categorical_list] = sc.fit_transform(X_drop_sc[numeric_list + ord_categorical_list])

'''VIF analysis for features that are highly correlated'''
'''Reference: http://support.minitab.com/en-us/minitab/17/topic-library/modeling-statistics/regression-and-correlation/model-assumptions/what-is-a-variance-inflation-factor-vif/'''
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
vif_delete_list = list(vif_metro.loc[vif_metro['VIF Factor'] > 10, 'features']) + \
                  list(vif_population.loc[vif_population['VIF Factor'] > 10**7 , 'features'])

'''Drop above features from X_drop and update the numeric list'''
'''https://stackoverflow.com/questions/28538536/deleting-multiple-columns-in-pandas'''
X_drop_sc.drop(vif_delete_list, axis = 1, inplace = True)
numeric_list_vif = numeric_list[:]
for item in vif_delete_list:
    numeric_list_vif.remove(item)
    
X_drop.drop(vif_delete_list, axis = 1, inplace = True)

'''Compile a string that will be fed into sm.OLS.from_formula'''
'''Handling numeric and ord categorical information. Add this too!'''
numeric_ord_cat_OLS_string = " + ".join(numeric_list_vif + ord_categorical_list)

'''Handling the nominal categorical information with C()'''
category_OLS_string = "C(material)"

'''yes no categorical values don't need to be separately encoded'''
yes_no_list = ['thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 
                   'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 
                   'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 
                   'culture_objects_top_25', 'product_type']
yes_no_OLS_string = " + ".join(yes_no_list)

'''This is the total string'''
OLS_string = numeric_ord_cat_OLS_string + " + " + category_OLS_string + " + " + yes_no_OLS_string + " + 0"

'''Create a dmatrix for encoding and extract the column variables so they can
be used to access features with low p values'''
'''https://stackoverflow.com/questions/23560104/fetching-names-from-designmatrix-in-patsy'''
dfX = dmatrix(OLS_string, data = X_drop)
dfX_columns = dfX.design_info.column_names
dfX = pd.DataFrame(dfX, columns = dfX_columns)

dfY = pd.DataFrame(X_drop_sc['price_doc'])

'''Feature extraction via multiple OLS runs''' 
dfX2, result, r2_list, num_feature_list, p_num, end_loop= OLS_analysis(dfX, dfY, 0.01)
while end_loop:
    dfX2, result, r2_list, num_feature_list, p_num, end_loop = OLS_analysis(dfX2, dfY, 0.01, p_prev = p_num, num_feature_compile = num_feature_list, r2_compile = r2_list)
OLS_feature_extracted = list(dfX2.columns.values)
print(result.summary())

'''Outlier extraction'''
idx = outlier_crusher(result)

'''https://stackoverflow.com/questions/39802076/pandas-drop-row-based-on-index-vs-ix'''

dfX2_idx = dfX2.drop(idx)
dfY_idx = dfY.drop(idx)

model_idx = sm.OLS(dfY_idx, dfX2_idx)
result_idx = model_idx.fit()
print(result_idx.summary())

'''After outlier removal big_market_km is having a high p value'''
del dfX2_idx['big_market_km']
OLS_feature_extracted.remove('big_market_km')

'''Now apply train_test analysis on OLS using exracted features after scaling
the necessary features. The split should be around 7:3 ratio, and make sure to
include stratified option'''

'''dfX2 is now our non-scaled feature matrix with dfY being the dependent variable
vector. First repartition dfX2 into numerical and categorical parts'''
dfX2_train, dfX2_test, dfY_train, dfY_test = train_test_split(dfX2_idx, dfY_idx, test_size = 0.3, random_state = 0)


'''
#Split into numerical anc categorical parts'''
dfX2_train_numeric, dfX2_train_cat = num_cat_divider(dfX2_train, OLS_feature_extracted)
dfX2_test_numeric, dfX2_test_cat = num_cat_divider(dfX2_test, OLS_feature_extracted)
#Index saved for later proprocessing
dfX2_train_index = list(dfX2_train_numeric.index.values)
dfX2_test_index = list(dfX2_test_numeric.index.values)


sc_dfx2 = StandardScaler()
dfX2_train_numeric_sc = pd.DataFrame(sc_dfx2.fit_transform(dfX2_train_numeric), 
                                     columns = dfX2_train_numeric.columns.values, index=dfX2_train_index)
dfX2_test_numeric_sc = pd.DataFrame(sc_dfx2.transform(dfX2_test_numeric), 
                                    columns = dfX2_test_numeric.columns.values, index=dfX2_test_index)

dfX2_train_sc = pd.concat([dfX2_train_numeric_sc, dfX2_train_cat], axis = 1)
dfX2_test_sc = pd.concat([dfX2_test_numeric_sc, dfX2_test_cat], axis = 1)


'''OLS run for train and test runs respectively'''
model_train = sm.OLS(dfY_train, dfX2_train_sc)
result_train = model_train.fit()
print(result_train.summary())

'''Reference: http://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html'''
'''Residual plot, Partial regression analysis and CCPR'''
stats.probplot(result_train.resid, plot=plt)
plt.show()

fig = plt.figure(figsize = (30, 18))
sm.graphics.plot_partregress_grid(result, fig=fig)
fig.suptitle("")
plt.show()


fig = plt.figure(figsize = (50, 20))
sm.graphics.plot_ccpr_grid(result, fig=fig)
fig.suptitle("")
plt.show()