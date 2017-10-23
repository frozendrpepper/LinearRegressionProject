# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:42:41 2017

@author: cck3
"""
'''This is a comprehensive analysis script where LinearRegression, Lasso, Ridge
and ElasticNet are all compared'''


from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''Load the data'''
boston = load_boston()

'''Sort out the data'''
X = pd.DataFrame(boston.data)
y = pd.DataFrame(boston.target)
column = boston.feature_names
X.columns = column
y.columns = ['Price']

'''Scale the features'''
sc = StandardScaler()
features_std = sc.fit_transform(X.values)
target = y.values.reshape(len(y), )

'''Make a list for different regularization strengths'''
alpha = np.array([0.01, 0.3, 1, 5, 10])

'''Perform LinearRegression separately alone since we don't need this to be repeated for
different sequences of alpha values'''
lr = LinearRegression()
kf = StratifiedKFold(y = target, n_folds = 10)
total_err= 0
for train, test in kf:
    lr.fit(features_std[train], target[train])
    p = lr.predict(features_std[test])
    err = p - target[test]
    total_err += np.dot(err, err)

rmse = np.sqrt( total_err / len(target) )
print('The KFold RMSE for linear regression is : {:.3f}'.format(rmse))
print()

for i in range(len(alpha)):
    classifier_list = [('lasso L1', Lasso(alpha = alpha[i])), ('ridge L2', Ridge(alpha = alpha[i])),
                       ('Elastic Net', ElasticNet(alpha[i]))]
    
    #kf = KFold(len(target), n_folds = 10)
    kf = StratifiedKFold(y = target, n_folds = 10)                 
    for name, met in classifier_list:
        total_err = 0
        for train, test in kf:
            met.fit(features_std[train], target[train])
            p = met.predict(features_std[test])
            err = p - target[test]
            total_err += np.dot(err, err)
            rmse = np.sqrt(total_err / len(target))
        print('The KFold RMSE for classifier {} with regularization strength {} is : {:.3f}'.format(name, alpha[i], rmse))
    print()
            