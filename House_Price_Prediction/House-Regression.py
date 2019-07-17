#!/usr/bin/env python
# coding: utf-8

# In[176]:


import math
import os
import pandas as pd
import numpy as np
import re
import pandas as pd
import sklearn
import joblib
from joblib import dump, load
from collections import Counter
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, LarsCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV, BayesianRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
import lightgbm as lgb

import imblearn
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, RandomOverSampler

import scipy
from scipy import stats 
from scipy.stats import norm
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.pyplot as plot


import keras
from keras import Sequential
from keras import backend as K
from keras import losses
from keras.layer import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.layers import average, LeakyReLU, Conv2D, UpSampling2D, MaxPooling2D
from keras.layers import Input, BatchNormalization
from keras.layers import Dense
from keras.layers import Lambda, Input, Dense, Dropout,GaussianNoise,Lambda,Flatten,Activation,concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.optimizers import Adam, Adadelta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet 

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[177]:


train = pd.read_csv('/Users/ruanhq/Desktop/Davis_PhD_Study/Jobs/Kaggle/House_regression/train.csv')
test = pd.read_csv('/Users/ruanhq/Desktop/Davis_PhD_Study/Jobs/Kaggle/House_regression/test.csv')


# In[178]:


test_ID = test['Id']


# In[179]:


train.head()


# In[180]:


train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)


# In[181]:


#First remove outliers:
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('Sale-Price', fontsize = 14)
plt.xlabel('Ground_livign_area', fontsize = 14)
plt.show()


# In[182]:


#Deleting the obvious outliers:
train = train.drop(train[(train['GrLivArea'] > 4000 ) &
                         (train['SalePrice'] < 400000)].index)
#Then redraw the plot:
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel('Gr_living_area', fontsize = 14)
plt.ylabel('Sale-Price', fontsize = 14)
plt.show()


# In[183]:


fig, ax = plt.subplots()
ax.scatter(x = train['LotArea'], y = train['SalePrice'])
plt.ylabel('Sale-Price', fontsize = 14)
plt.xlabel('Lot-Area', fontsize = 14)
plt.show()


# In[184]:


#train.drop(train[(train['LotArea'] > 150000) & (train['SalePrice'] < 400000)].index)
#fig, ax = plt.subplots()
#ax.scatter(x = train['LotArea'], y = train['SalePrice'])
#plt.xlabel('Lot-Area', fontsize = 14)
#plt.ylabel('Sale-Price', fontsize = 14)
#plt.show()


# In[185]:


#Target variable:
sns.distplot(train['SalePrice'], fit = norm)


# In[186]:


(mu, sigma) = norm.fit(train['SalePrice'])

mu, sigma


# In[187]:


train['SalePrice'].skew()


# In[188]:


#Draw the Q-Q plot:
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)


# In[189]:


#Log-transformation on the target variable:
train['SalePrice'] = np.log1p(train['SalePrice'])
norm.fit(train['SalePrice'])


# In[190]:


sns.distplot(train['SalePrice'])


# In[191]:


stats.probplot(train['SalePrice'], plot = plt)
#Much better


# In[192]:


######
#Then starts to do feature engineering:
n_train = train.shape[0]
n_test = test.shape[0]
Y_train = train['SalePrice'].values
whole_data = pd.concat((train, test)).reset_index(drop = True)
whole_data.drop(['SalePrice'], axis = 1, inplace = True)
whole_data.shape


# In[193]:


missi = whole_data.isnull().sum() / len(whole_data)
missi = missi[missi>0].sort_values(ascending = False)

# In[194]:


#Visualize the missing ratio by feature:
f, ax = plt.subplots(figsize = (10, 7))
plt.xticks(rotation = '90')
sns.barplot(x = missi.index, y = missi)


# In[195]:


#Then for the features with missing ratio larger than 0.4, we just use None to impute:
None1 = missi.index[missi>0.4].values
for t in None1:
    whole_data[t] = whole_data[t].fillna('None')


# In[196]:


#Then for LotFrontage: impute by the median:
whole_data['LotFrontage'] = whole_data.groupby('Neighborhood')['LotFrontage'].transform(lambda X: X.fillna(X.median()))


# In[197]:


#Then for Garage:
for t in ('GarageQual', 'GarageCond', 'GarageFinish', 'GarageType'):
    whole_data[t] = whole_data[t].fillna('None')


# In[198]:


for t in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    whole_data[t] = whole_data[t].fillna(0)


# In[199]:


for t in ('BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF'):
    whole_data[t] = whole_data.groupby('Neighborhood')[t].transform(lambda X: X.fillna(X.median()))
    


# In[200]:


for t in ('BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2'):
    whole_data[t] = whole_data[t].fillna('None')


# In[201]:


whole_data.drop(['Utilities'], axis = 1, inplace = True)


# In[202]:


for t in ('Electrical' ,'Exterior1st', 'Exterior2nd', 'KitchenQual','MSZoning', 'SaleType', 'Functional'):
    whole_data[t] = whole_data[t].fillna(whole_data[t].mode()[0])


# In[203]:


whole_data['MasVnrType'] = whole_data['MasVnrType'].fillna('None')


# In[204]:


whole_data['MasVnrArea'] = whole_data['MasVnrArea'].fillna(0)


# In[205]:


#Check whether we have removed all of the missing values:
missi = whole_data.isnull().sum() / len(whole_data)
missi = missi[missi>0].sort_values(ascending = False)
missi


# In[206]:


whole_data['MSSubClass'] = whole_data['MSSubClass'].astype(str)
whole_data['OverallCond'] = whole_data['OverallCond'].astype(str)
whole_data['YrSold'] = whole_data['YrSold'].astype(str)
whole_data['MoSold'] = whole_data['MoSold'].astype(str)


# In[207]:


#Add new features:
whole_data['Total_porch_sf'] = (whole_data['OpenPorchSF'] + whole_data['3SsnPorch'] +
                              whole_data['EnclosedPorch'] + whole_data['ScreenPorch'] +
                              whole_data['WoodDeckSF'])
whole_data['Total_Bathrooms'] = (whole_data['FullBath'] + (0.5 * whole_data['HalfBath']) +
                               whole_data['BsmtFullBath'] + (0.5 * whole_data['BsmtHalfBath']))


# In[208]:


#Perform the label encoder:
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(whole_data[c].values))
    whole_data[c] = lbl.transform(list(whole_data[c].values))
print('shape: {}'.format(whole_data.shape))
whole_data['TotalSF'] = whole_data['TotalBsmtSF'] + whole_data['1stFlrSF'] + whole_data['2ndFlrSF']
whole_data['Total_Home_Quality'] = whole_data['OverallQual'] + whole_data['OverallCond']

# In[209]:


whole_data['1stFlrSF'].skew()


# In[210]:


#Operation on skewed features:
numerical_features = whole_data.dtypes[whole_data.dtypes != 'object'].index
skewed_features = whole_data[numerical_features].apply(lambda X: skew(X)).sort_values(ascending = False)
skewness = pd.DataFrame({'Skew': skewed_features})
skewness.head(15)


# In[211]:


skewness = skewness[abs(skewness) > 0.7]
print('There are {} skewed features to modify'.format(len(skewness)))


# In[212]:


from scipy.special import boxcox1p
lam = 0.15
for feat in skewness.index:
    whole_data[feat] = boxcox1p(whole_data[feat], lam)


# In[213]:


#Transform it to the dummy variable case:
whole_data = pd.get_dummies(whole_data)
print('The data has {} features'.format(whole_data.shape[1]))


# In[214]:


X_train = whole_data[:n_train]
X_test = whole_data[n_train:]
#Save the data after feature engineering:


os.chdir('/Users/ruanhq/Desktop/Davis_PhD_Study/Jobs/Spark')
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
Y_train = pd.DataFrame(Y_train, columns = ['SalePrice'])
train_df = pd.concat([Y_train, X_train], axis = 1)
train_df.to_csv('train_df.csv')

#Y_train.to_csv('Y_train.csv')


# In[215]:


#Y_train


# In[216]:


#Adding some more features may help.


# ### Starting to model:

# In[217]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
#import lightgbm as lgb


# In[218]:


#Calculate the CV-RMSE:
n_folds = 10
def CV_RMSE(model):
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(X_train.values)
    rmse = np.sqrt(-cross_val_score(model, X_train.values, Y_train, scoring = 'neg_mean_squared_error', cv = kf))
    return rmse


# In[219]:
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lasso_pip = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 42))
elas_net_pip = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR_pip = KernelRidge(alpha = 0.5, kernel = 'polynomial', degree = 2, coef0 = 2.5)
GBoo_pip = GradientBoostingRegressor(n_estimators = 2000,
                                     learning_rate = 0.05,
                                     max_depth = 4,
                                     max_features = 'sqrt',
                                     min_samples_leaf = 15,
                                     min_samples_split = 8,
                                     loss = 'huber',
                                     random_state = 42)
Bridge_pip = make_pipeline(RobustScaler(), BayesianRidge())
SGD_pip = make_pipeline(RobustScaler(), SGDRegressor(max_iter = 10000, tol = 1e-3))



# In[220]:


score = CV_RMSE(lasso_pip)
print('For Lasso the CV-RMSE mean is {:.4f} +- {:.4f}\n'.format(score.mean(), score.std()))


# In[221]:


score = CV_RMSE(elas_net_pip)
print('For Elastic Net the CV-RMSE mean is {:.4f} +- {:.4f}\n'.format(score.mean(), score.std()))


# In[222]:


score = CV_RMSE(KRR_pip)
print('For Kernel Ridge Regression the CV-RMSE mean is {:.4f} +- {:.4f}\n'.format(score.mean(), score.std()))


# In[223]:


score = CV_RMSE(GBoo_pip)
print('For GB-Regressor the CV-RMSE mean is {:.4f} +- {:.4f}\n'.format(score.mean(), score.std()))


# In[224]:


XGB_pipeline = xgb.XGBRegressor(colsample_bytree = 0.5,
                               gamma = 0.05,
                               learning_rate = 0.05,
                               max_depth = 3,
                               min_child_weight = 1.5, n_estimators = 2500, reg_alpha = 0.5, reg_lambda = 0.85, subsample = 0.4, random_state = 42)


# In[225]:


score = CV_RMSE(XGB_pipeline)
print('For XGB-Regressor the CV-RMSE mean is {:.4f}+- {:.4f}\n'.format(score.mean(), score.std()))


# In[226]:


XGB_pipeline2 = xgb.XGBRegressor(colsample_bytree = 0.3, gamma = 0.05, learning_rate = 0.01, max_depth = 5, min_child_weight = 1.5, n_estimators = 3000, reg_alpha = 0.5, reg_lambda = 0.8, subsample = 0.35, random_state = 1)


# In[227]:


score = CV_RMSE(XGB_pipeline2)
print('FOR XGB Regressor the CV-RMSE mean is {:.4f}+- {:.4f}\n'.format(score.mean(), score.std()))


# In[228]:


rf_pipeline = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators = 5000, min_samples_split = 2, oob_score = True, warm_start = True, min_samples_leaf = 2, max_features = 12, max_depth = 20, random_state = 1))
score = CV_RMSE(rf_pipeline)
print('For random forest the CV-RMSE is {:.4f}+-{:.4f}'.format(score.mean(), score.std()))


# In[229]:


#Huber regression:
from sklearn.linear_model import HuberRegressor
huber_pip = make_pipeline(RobustScaler(), HuberRegressor(epsilon = 5.5, max_iter = 1000, alpha = 0.01))
score = CV_RMSE(huber_pip)
print('For Huber Regressor the CV-RMSE is {:.4f}+-{:.4f}'.format(score.mean(), score.std()))


# In[230]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[231]:
########
#The stacking averaged models is to do a second level model fitting given the first level prediction(out-of-fold prediction)
#run on the features extracted from the out-of-fold prediction which serves as the input of the downstream model fitting.
#To generalize better

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)



# In[232]:



#Stacking Average models:
stacked_aver_model1 = StackingAveragedModels(base_models = (elas_net_pip, GBoo_pip, KRR_pip, model_lgb2, huber_pip, Bridge_pip, SGD_pip), meta_model = lasso_pip)
score = CV_RMSE(stacked_aver_model1)
print('Stacking Averaged models score: {:.4f} +- {:.4f}'.format(score.mean(), score.std()))


# In[234]:


# In[235]:


########
#Perform the second level prediction:
stacked_aver_model1.fit(X_train.values, Y_train)
stacked_train_pred = stacked_aver_model1.predict(X_train.values)
stacked_pred1 = np.expm1(stacked_aver_model1.predict(X_test.values))
print(np.sqrt(mean_squared_error(stacked_train_pred, Y_train)))


# In[236]:





# In[237]:


#####
#Integrate XGBoost and LightGBM:
XGB_pipeline.fit(X_train.values, Y_train)
XGB_train_pred = XGB_pipeline.predict(X_train.values)
XGB_pred = np.expm1(XGB_pipeline.predict(X_test.values))
print(np.sqrt(mean_squared_error(XGB_train_pred, Y_train)))

model_lgb.fit(X_train.values, Y_train)
lgb_train_pred = model_lgb.predict(X_train.values)
lgb_pred = np.expm1(model_lgb.predict(X_test.values))
print(rmsle(Y_train, lgb_train_pred))

# In[251]:
ensemble = stacked_pred1 * 0.7 + XGB_pred * 0.15 + lgb_pred * 0.15




# In[252]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission11.csv', index = False)




# In[ ]:


#Output the data:

