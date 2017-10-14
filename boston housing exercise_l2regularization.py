# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:52:21 2017

@author: cck3
"""

'''This version implements the Ridge (L2) regularized linear regression
L2 regression does not force sparcity, but according to Sebastian Raschka
it generally yields a better performance than L1'''
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

'''First load the boston housing data'''
boston = load_boston()

'''Convert the data into features and labels'''
X = pd.DataFrame(boston.data)
y = pd.DataFrame(boston.target)
X.columns = boston.feature_names
y.columns = ['Price']

'''Apply features scaling'''
sc = StandardScaler()
X_std = sc.fit_transform(X)
y = y.values.reshape((len(y), ))
X = X.values

'''Let us first try a simple case of Ridge Linear Regression'''
rd = Ridge(alpha = 0.5)
rd.fit(X_std, y)

'''Compute the RMSE for simple Ridge case'''
prediction = rd.predict(X_std)
error = prediction - y
total_error = np.dot(error, error)
rmse_train = np.sqrt( total_error / len(y) )


'''When the analysis is performed over non-scaled features, Ridge had a better
performance than vanilla regression. However, with scaled features, they virtually
performed the same'''

'''Apply regular kfold analysis'''
kf = KFold(len(X), n_folds = 10)
xval_error = 0

for train, test in kf:
    rd.fit(X_std[train], y[train])
    p = rd.predict(X_std[test])
    err = p - y[test]
    xval_error += np.dot(err, err)

rmse_kfold = np.sqrt( xval_error / len(y) )


'''Let us apply stratified kfold analysis'''
kf_stratified = StratifiedKFold(y = y, n_folds = 10)
xval_error_strat = 0

for train, test in kf_stratified:
    rd.fit(X_std[train], y[train])
    p = rd.predict(X_std[test])
    err = p - y[test]
    xval_error_strat += np.dot(err, err)
rmse_kfold_strat = np.sqrt( xval_error_strat / len(y) )

'''As it was observed with with regular linear regression, the Stratified K Fold
usually have a better RMSE result, which makes sense since Stratified KFold is the improved
version of KFold'''

'''Now let us try different values of alpha and how it impacts the result'''
alpha = np.array([0.01, 0.1, 1, 10, 100])
train_rmse_list = []
train_coef_list = []
kfold_rmse_list = []

for a in alpha:
    rd = Ridge(a)
    rd.fit(X_std, y)
    p = rd.predict(X_std)
    err = p - y
    total_err = np.dot(err, err)
    train_rmse_list.append(np.sqrt(total_err / len(y)))
    train_coef_list.append(rd.coef_)
    
    '''Kfold analysis with different alpha values'''
    kf_stratified = StratifiedKFold(y = y, n_folds = 10)
    xval_error_kfold = 0
    
    for train, test in kf_stratified:
        rd.fit(X_std[train], y[train])
        p = rd.predict(X_std[test])
        err = p - y[test]
        xval_error_kfold += np.dot(err, err)
    kfold_rmse_list.append(np.sqrt(xval_error_kfold / len(y)))

for i in range(len(alpha)):
    print('The RMSE train corresponding to alpha {} : {:5.3f}'.format(alpha[i], train_rmse_list[i]))
    print('The RMSE kfold corresponding to alpha {} : {:5.3f}'.format(alpha[i], kfold_rmse_list[i]))
    print()