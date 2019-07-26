#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clickisng run or pressing Shift+Enter) will list the files in the input directory
import numpy 
import pandas as pd
import json
############
#IEEE_Fraud_Detection


import numpy 
import pandas as pd
import json
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error

####
#Reduce the memory usage:
"""
:https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee
"""
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

sub0 = pd.read_csv('../input/prevsub/submission2.csv')

train = pd.merge(train_transaction, train_identity,
                on = 'TransactionID', how = 'left')
test = pd.merge(test_transaction, test_identity,
               on = 'TransactionID', how = 'left')

del train_identity, train_transaction, test_identity, test_transaction

#####
#Integrates more advanced feature engineering:
feature_mean_std = ['id_02', 'id_03', 'id_04', 'id_05', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', '']

#Starts feature engineering:
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].mean()
train['TransactionAmt_to_mean_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].mean()
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].std()
train['TransactionAmt_to_std_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].std()
train['TransactionAmt_to_mean_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].mean()
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].mean()
train['TransactionAmt_to_std_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].std()
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].std()


test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].mean()
test['TransactionAmt_to_mean_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].mean()
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].std()
test['TransactionAmt_to_std_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].std()
test['TransactionAmt_to_mean_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].mean()
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].mean()
test['TransactionAmt_to_std_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].std()
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].std()


train['id2_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id2_mean_card2'] = train['id_02'] / train.groupby(['card2'])['id_02'].transform('mean')
train['id2_mean_card3'] = train['id_02'] / train.groupby(['card3'])['id_02'].transform('mean')
train['id2_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')



train['id2_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id2_std_card2'] = train['id_02'] / train.groupby(['card2'])['id_02'].transform('std')
train['id2_std_card3'] = train['id_02'] / train.groupby(['card3'])['id_02'].transform('std')
train['id2_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')





test['id2_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id2_mean_card2'] = test['id_02'] / test.groupby(['card2'])['id_02'].transform('mean')
test['id2_mean_card3'] = test['id_02'] / test.groupby(['card3'])['id_02'].transform('mean')
test['id2_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')



test['id2_std_card1'] = train['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id2_std_card2'] = train['id_02'] / test.groupby(['card2'])['id_02'].transform('std')
test['id2_std_card3'] = train['id_02'] / test.groupby(['card3'])['id_02'].transform('std')
test['id2_std_card4'] = train['id_02'] / test.groupby(['card4'])['id_02'].transform('std')


train['D12_to_mean_card1'] = train['D12'] / train.groupby(['card1'])['D12'].transform('mean')
train['D12_to_mean_card4'] = train['D12'] / train.groupby(['card4'])['D12'].transform('mean')
train['D12_to_std_card1'] = train['D12'] / train.groupby(['card1'])['D12'].transform('std')
train['D12_to_std_card4'] = train['D12'] / train.groupby(['card4'])['D12'].transform('std')

test['D12_to_mean_card1'] = test['D12'] / test.groupby(['card1'])['D12'].transform('mean')
test['D12_to_mean_card4'] = test['D12'] / test.groupby(['card4'])['D12'].transform('mean')
test['D12_to_std_card1'] = test['D12'] / test.groupby(['card1'])['D12'].transform('std')
test['D12_to_std_card4'] = test['D12'] / test.groupby(['card4'])['D12'].transform('std')

train['D12_to_mean_card1'] = train['D12'] / train.groupby(['card1'])['D12'].transform('mean')
train['D12_to_mean_card4'] = train['D12'] / train.groupby(['card4'])['D12'].transform('mean')
train['D12_to_std_card1'] = train['D12'] / train.groupby(['card1'])['D12'].transform('std')
train['D12_to_std_card4'] = train['D12'] / train.groupby(['card4'])['D12'].transform('std')

test['D12_to_mean_card1'] = test['D12'] / test.groupby(['card1'])['D12'].transform('mean')
test['D12_to_mean_card4'] = test['D12'] / test.groupby(['card4'])['D12'].transform('mean')
test['D12_to_std_card1'] = test['D12'] / test.groupby(['card1'])['D12'].transform('std')
test['D12_to_std_card4'] = test['D12'] / test.groupby(['card4'])['D12'].transform('std')

train['D12_to_mean_card2'] = train['D12'] / train.groupby(['card2'])['D12'].transform('mean')
train['D12_to_mean_card3'] = train['D12'] / train.groupby(['card3'])['D12'].transform('mean')
train['D12_to_std_card2'] = train['D12'] / train.groupby(['card2'])['D12'].transform('std')
train['D12_to_std_card3'] = train['D12'] / train.groupby(['card3'])['D12'].transform('std')

test['D12_to_mean_card2'] = test['D12'] / test.groupby(['card2'])['D12'].transform('mean')
test['D12_to_mean_card3'] = test['D12'] / test.groupby(['card3'])['D12'].transform('mean')
test['D12_to_std_card2'] = test['D12'] / test.groupby(['card2'])['D12'].transform('std')
test['D12_to_std_card3'] = test['D12'] / test.groupby(['card3'])['D12'].transform('std')

train['D12_to_mean_card2'] = train['D12'] / train.groupby(['card2'])['D12'].transform('mean')
train['D12_to_mean_card3'] = train['D12'] / train.groupby(['card3'])['D12'].transform('mean')
train['D12_to_std_card2'] = train['D12'] / train.groupby(['card2'])['D12'].transform('std')
train['D12_to_std_card3'] = train['D12'] / train.groupby(['card3'])['D12'].transform('std')

test['D12_to_mean_card2'] = test['D12'] / test.groupby(['card2'])['D12'].transform('mean')
test['D12_to_mean_card3'] = test['D12'] / test.groupby(['card3'])['D12'].transform('mean')
test['D12_to_std_card2'] = test['D12'] / test.groupby(['card2'])['D12'].transform('std')
test['D12_to_std_card3'] = test['D12'] / test.groupby(['card3'])['D12'].transform('std')

train['D12_to_mean_addr1'] = train['D12'] / train.groupby(['addr1'])['D12'].transform('mean')
test['D12_to_mean_addr1'] = test['D12'] / test.groupby(['addr1'])['D12'].transform('mean')

train['D12_to_std_addr1'] = train['D12'] / train.groupby(['addr1'])['D12'].transform('std')
test['D12_to_std_addr1'] = test['D12'] / test.groupby(['addr1'])['D12'].transform('std')

train['D12_to_mean_addr2'] = train['D12'] / train.groupby(['addr2'])['D12'].transform('mean')
test['D12_to_mean_addr2'] = test['D12'] / test.groupby(['addr2'])['D12'].transform('mean')

train['D12_to_std_addr2'] = train['D12'] / train.groupby(['addr2'])['D12'].transform('std')
test['D12_to_std_addr2'] = test['D12'] / test.groupby(['addr2'])['D12'].transform('std')


train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_card2'] = train['D15'] / train.groupby(['card2'])['D15'].transform('mean')
train['D15_to_mean_card3'] = train['D15'] / train.groupby(['card3'])['D15'].transform('mean')
train['D15_to_std_card2'] = train['D15'] / train.groupby(['card2'])['D15'].transform('std')
train['D15_to_std_card3'] = train['D15'] / train.groupby(['card3'])['D15'].transform('std')

test['D15_to_mean_card2'] = test['D15'] / test.groupby(['card2'])['D15'].transform('mean')
test['D15_to_mean_card3'] = test['D15'] / test.groupby(['card3'])['D15'].transform('mean')
test['D15_to_std_card2'] = test['D15'] / test.groupby(['card2'])['D15'].transform('std')
test['D15_to_std_card3'] = test['D15'] / test.groupby(['card3'])['D15'].transform('std')

train['D15_to_mean_card2'] = train['D15'] / train.groupby(['card2'])['D15'].transform('mean')
train['D15_to_mean_card3'] = train['D15'] / train.groupby(['card3'])['D15'].transform('mean')
train['D15_to_std_card2'] = train['D15'] / train.groupby(['card2'])['D15'].transform('std')
train['D15_to_std_card3'] = train['D15'] / train.groupby(['card3'])['D15'].transform('std')

test['D15_to_mean_card2'] = test['D15'] / test.groupby(['card2'])['D15'].transform('mean')
test['D15_to_mean_card3'] = test['D15'] / test.groupby(['card3'])['D15'].transform('mean')
test['D15_to_std_card2'] = test['D15'] / test.groupby(['card2'])['D15'].transform('std')
test['D15_to_std_card3'] = test['D15'] / test.groupby(['card3'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')

train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')
test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')


train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)


##############
#Add more features:
import datetime
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train['Date'] = train['TransactionDT'].apply(lambda X: (startdate + datetime.timedelta(seconds = X)))
train['_Weekdays'] = train['Date'].dt.dayofweek
train['_Hours'] = train['Date'].dt.hour
train['_Days'] = train['Date'].dt.day



test['Date'] = test['TransactionDT'].apply(lambda X: (startdate + datetime.timedelta(seconds = X)))
test['_Weekdays'] = test['Date'].dt.dayofweek
test['_Hours'] = test['Date'].dt.hour
test['_Days'] = test['Date'].dt.day

train['_Weekdays'] = train['_Weekdays'].astype(np.int8)
test['_Weekdays'] = test['_Weekdays'].astype(np.int8)
train['_Hours'] = train['_Hours'].astype(np.int8)
test['_Hours'] = test['_Hours'].astype(np.int8)
train['_Days'] = train['_Days'].astype(np.int8)
test['_Days'] = test['_Days'].astype(np.int8)


##############
from scipy.stats import variation
#Integrate more features quantifying variation for each people within a week/hour/day:
gp1 = train.groupby(['_Hours', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_hour_prod_var'})
train = train.merge(gp1, on = [['_Hours', 'ProductCD']], how = 'left')
gp2 = train.groupby(['_Weekdays', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_week_prod_var'})
train = train.merge(gp2, on = [['_Weekdays', 'ProductCD']], how = 'left')
gp3 = train.groupby(['_Days', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_days_prod_var'})
train = train.merge(gp3, on = [['_Days', 'ProductCD']], how = 'left')

gp1 = test.groupby(['_Hours', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_hour_prod_var'})
test = test.merge(gp1, on = [['_Hours', 'ProductCD']], how = 'left')
gp2 = test.groupby(['_Weekdays', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_week_prod_var'})
test = test.merge(gp2, on = [['_Weekdays', 'ProductCD']], how = 'left')
gp3 = test.groupby(['_Days', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_days_prod_var'})
test = test.merge(gp3, on = [['_Days', 'ProductCD']], how = 'left')

#Add more feature related to the fraud:
gp1 = train.groupby(['_Hours', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_hour_prod_var'})
train = train.merge(gp1, on = [['_Hours', 'ProductCD']], how = 'left')
gp2 = train.groupby(['_Weekdays', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_week_prod_var'})
train = train.merge(gp2, on = [['_Weekdays', 'ProductCD']], how = 'left')
gp3 = train.groupby(['_Days', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_days_prod_var'})
train = train.merge(gp3, on = [['_Days', 'ProductCD']], how = 'left')

gp1 = test.groupby(['_Hours', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_hour_prod_var'})
test = test.merge(gp1, on = [['_Hours', 'ProductCD']], how = 'left')
gp2 = test.groupby(['_Weekdays', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_week_prod_var'})
test = test.merge(gp2, on = [['_Weekdays', 'ProductCD']], how = 'left')
gp3 = test.groupby(['_Days', 'ProductCD'])['TransactionAmt'].apply(lambda X: variation(X)).reset_index().rename(index = str, columns = {'TransactionAmt': 'Hetero_days_prod_var'})
test = test.merge(gp3, on = [['_Days', 'ProductCD']], how = 'left')

#########
#Remove the highly correlated features:


#########
#Start from multiple variables:
#Doesn't seem to help:
train = train.drop([['_Weekdays', '_Hours', 'Days']], axis = 1)
test = test.drop([['_Weekdasys', '_Hours', 'Days']], axis = 1)

#Drop all of the columns where has >90% missing values:
null_cols_train = [cols for cols in train.columns if train[cols].isnull().sum() / len(train) > 0.87]
null_cols_test = [cols for cols in test.columns if test[cols].isnull().sum() / len(train) > 0.87]
cols_dominating_value_train = [cols for cols in train.columns if train[cols].value_counts(dropna = False, normalize = True).values[0] > 0.87]
cols_dominating_value_test = [cols for cols in test.columns if test[cols].value_counts(dropna = False, normalize = True).values[0] > 0.87]


cols_dropping = list(set(null_cols_train + null_cols_test + cols_dominating_value_train + cols_dominating_value_test))
cols_dropping.remove('isFraud')


train = train.drop(cols_dropping, axis = 1)
test = test.drop(cols_dropping, axis = 1)


from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit


#Starting to train:
folds = TimeSeriesSplit(n_splits = 10)
#folds = KFold(n_splits = 5)


####
"""
Inspired by the following kernels: 
1. https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
2. https://www.kaggle.com/stocks/under-sample-with-multiple-runs
"""

def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual'):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                        'catboost_metric_name': 'AUC',
                        'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    
    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict_proba(X_test)
        
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        if averaging == 'usual':
            
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
                                  
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
                                  
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
        
    return prediction

def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    
#Transform categorical cols:
cate_variables = list(train.columns[train.dtypes == 'object'])
for col in cate_variables:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
        
        
####
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis = 1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis = 1)
del train
test = test[['TransactionDT', 'TransactionID']]



rf_model = RandomForestClassifier(n_estimators = 10000,
    max_depth = 20, min_samples_split = 50,
    oob_score = True, max_features = 0.1)
result_dict_rf = train_model_classification(X = X,
    X_test = X_test, y = y, folds = folds,model_type = 'sklearn', eval_metric = 'auc', early_stopping_rounds = 200, model = rf_model, averaging = 'rank' )
#xgb_params = {'eta': 0.03,
#               'max_depth': 6,
#               'subsample': 0.88,
#               'objective': 'binary:logistic',
#               'eval_metric': 'auc',
#               'silent': True,
#               'nthread': -1,
#               'tree_method': 'gpu_hist'}
#print('Starting_Training!!')
#result_dict_xgb = train_model_classification(X=X, X_test=X_test, y=y, params=xgb_params, folds=folds, model_type='xgb', eval_metric='auc', plot_feature_importance=False,
#                                                       verbose=500, early_stopping_rounds=200, n_estimators=5500, averaging='rank')




test = test.sort_values('TransactionDT')
test['prediction'] = result_dict_rf
sub0['isFraud'] = pd.merge(sub0, test, on='TransactionID')['prediction']
sub.to_csv('submission.csv', index=False)


