# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# library
import gc
import resource
import rope
#import operator
import numpy as np
import pandas as pd
import matplotlib
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
#%% Load data
print('Start loading data ...')

# Load and combine training data
#train_1 = pd.read_csv('/users/mac/desktop/train_FD001.txt', delim_whitespace=True, header=None)
train_2 = pd.read_csv('./input/train_FD002.txt', delim_whitespace=True, header=None)
#train_3 = pd.read_csv('/users/mac/desktop/test_FD003.txt', delim_whitespace=True, header=None)
#train_4 = pd.read_csv('./input/train_FD004.txt', delim_whitespace=True, header=None)
train =train_3
# Check dimension
print('Dimension of train_FD003: ', train.shape)

# Assign column headers: id, te, (time evolutions) os1, os2, os3, (operational settings) s1, s2, ..., s21 (sensors)
sensor_name = ['s'+ str(i) for i in range(1,22)]
train.columns = ['id', 'te', 'os1', 'os2', 'os3'] + sensor_name
print('Number of cycles per machine: ', train['id'].unique().size)
s2_100_cycle = train.ix[:,'os2']
s2_1_cycle = train.ix[train['id']==1,'os2']
plt.figure(1)
plt.plot(s2_100_cycle)
plt.title('Signal from operational settings 2 in 100 cycles')
plt.figure(2)
plt.plot(s2_1_cycle)
plt.title('Signal from operational settings 2 in 1 cycle')
# Smooth technique: decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

res = seasonal_decompose(s2_1_cycle.values, freq=10)
print(res.trend)
plt.figure(3)
resplot = res.plot()
# access individual series by
trend = res.trend
seasonal = res.seasonal
residual = res.resid
plt.figure(4)
line_1, = plt.plot(s2_1_cycle, label='line 1')
line_2, = plt.plot(trend, label='line 2')
plt.title('Raw and smoothed signal')
plt.legend([line_1, line_2], ['o2', 'Smoothed o2'])
def f(col):
    smooth_col = seasonal_decompose(col.values, freq=10).trend
    return smooth_col
transformed = train.drop(['te', 'os1', 'os2', 'os3'], axis=1).groupby('id').transform(f)
transformed['id'] = train['id']  # add back because transform will throw away 'id'

# overwrite smooth signals back to train Dataframe
df_1 = train[['id', 'te', 'os1', 'os2', 'os3']]
df_2 = transformed.groupby('id').ffill().bfill().drop('id', axis=1)
train = pd.concat([df_1, df_2], axis=1)

# Plot out the time evolution
plt.plot(train['te'][1:600])
plt.title('Time evolution')
plt.ylabel('Time Evolution')
plt.xlabel('No of Cycles')
def f1(col):
    # Option 1: Reverse the time evolution, where remaining time of a machine is 1 at the failure.
    return col[::-1]  

train['rul'] = train[['id', 'te']].groupby('id').transform(f1)
plt.plot(train.rul[1:600])
plt.ylabel('Time Evolution')
plt.xlabel('No of Cycles')
# prepare the data
# then split them into train and test sets
y = train['rul']
features = train.columns.drop(['id', 'te', 'rul'])
X = pd.DataFrame(normalize(train[features], axis=0))
X.columns = features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
# try Lasso
ls = LassoCV(random_state=12)
ls = ls.fit(X_train, y_train)

#print('List of tried parameter values: ', ls.alphas_)
print('Optimal value: ', ls.alpha_)
print(ls.coef_)
print('Useful sensors to predict RUL: ', X_train.columns[abs(ls.coef_) > 1e-6])

# compare predict RUL and real RUL
plt.figure(1)
plt.plot(np.log(ls.predict(X_test)), np.log(y_test), 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.plot(range(6), range(6))

# compare predict RUL and real RUL
plt.figure(2)
plt.plot(ls.predict(X_test), y_test, 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.title('Comparison of Predicted RUL and Real RUL')
plt.axis([-100, 400, 0, 400])
plt.plot(range(300), range(300))  # plot the line y = x of perfect prediction
# try plotting signals to convince yourself that these important sensors really have correlation with our target
plt.plot(X_train['s4'][1:600], y_train[1:600], 'ro')
# Now let's try do log transformation to create exponential-degenerating RUL

def f1(col):
    # Option 2: transform time evolution into exponential-degenerating remaining health index 
    return np.log(col[::-1] + 1)  

plt.figure(1)
train['rul'] = train[['id', 'te']].groupby('id').transform(f1)
plt.plot(train.rul[1:600])

y = train['rul']
features = train.columns.drop(['id', 'te', 'rul'])
X = train[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# try Lasso
from sklearn.linear_model import LassoCV

ls = LassoCV(random_state=12)
ls = ls.fit(X_train, y_train)

#print('List of tried parameter values: ', ls.alphas_)
print('Optimal value: ', ls.alpha_)
print(ls.coef_)
print('Useful sensors to predict RUL: ', X_train.columns[abs(ls.coef_) > 1e-6])

# compare predict RUL and real RUL
plt.figure(2)
plt.plot(ls.predict(X_test), y_test, 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.plot(range(6), range(6))
plt.axis([-2, 6, 0, 6])

plt.figure(3)
plt.plot(np.exp(ls.predict(X_test)), np.exp(y_test), 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.title('Comparison of Predicted RUL and Real RUL')
plt.axis([-100, 400, 0, 400])
plt.plot(range(300), range(300))
def f1(col):
    # Option 1: Reverse the time evolution, where remaining time of a machine is 1 at the failure.
    return col[::-1] 

plt.figure(1)
train['rul'] = train[['id', 'te']].groupby('id').transform(f1)
plt.plot(train.rul[1:600])

y = train['rul']
features = train.columns.drop(['id', 'te', 'rul'])
X = train[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# try Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=12)
rf = rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(1)
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlim([-1, X_train.shape[1]])
plt.show()


# compare predict RUL and real RUL
plt.figure(2)
plt.plot(np.log(rf.predict(X_test)), np.log(y_test), 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.plot(range(6), range(6))

plt.figure(3)
plt.plot(rf.predict(X_test), y_test, 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.title('Comparison of Predicted RUL and Real RUL')
plt.plot(range(300), range(300))
# Now let's try do log transformation to create exponential-degenerating RUL

def f1(col):
    # Option 2: transform time evolution into exponential-degenerating remaining health index 
    return np.log(col[::-1] + 1)  

plt.figure(1)
train['rul'] = train[['id', 'te']].groupby('id').transform(f1)
plt.plot(train.rul[1:600])

y = train['rul']
features = train.columns.drop(['id', 'te', 'rul'])
X = train[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# try Lasso
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=12)
rf = rf.fit(X_train, y_train)

# compare predict RUL and real RUL
plt.figure(2)
plt.plot(rf.predict(X_test), y_test, 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.plot(range(6), range(6))
#plt.axis([-2, 6, 0, 6])

plt.figure(3)
plt.plot(np.exp(rf.predict(X_test)), np.exp(y_test), 'ro')
plt.xlabel('Predicted RUL')
plt.ylabel('Real RUL')
plt.title('Comparison of Predicted RUL and Real RUL')
plt.plot(range(300), range(300))
