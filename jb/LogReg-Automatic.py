#%%

import collections
import gc
import IPython
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import plotly.graph_objs as go
import plotly.plotly as py
import random 
#import seaborn as sb
import seaborn as sns
#import seaborn as snss
#import scipy
import scipy as sp
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import warnings

from datetime import date
from IPython.display import HTML
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from pandas import DataFrame
from pandas import Series
from patsy import dmatrices
from plotly import tools
from plotly.offline import init_notebook_mode
from plotly.offline import iplot
from pylab import rcParams
from random import choice
from random import choices # Python 3.6+
from random import sample
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from wordcloud import WordCloud

init_notebook_mode(connected=True)
plt.rcParams.update({'figure.max_open_warning': 200})
# Suppress warnings
warnings.filterwarnings("ignore")

# In a notebook environment, display the plots inline
#%matplotlib inline
# Set some parameters to apply to all plots. These can be overridden in each plot if desired.
# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')
# emulate the aesthetics of ggplot (a popular plotting package for R)
matplotlib.style.use('ggplot')

np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

#import C:/Development/kaggle--home-credit-default-risk/rand_jitter
#import C:/Development/kaggle--home-credit-default-risk/draw_feature_distribution

# sys.path.insert(0, 'C:/Development/kaggle--home-credit-default-risk/') # ~= sys.path.prepend
sys.path.append('C:/Development/kaggle--home-credit-default-risk/')
# import rand_jitter
# import draw_feature_distribution
##from rand_jitter import * # NOTE: added directly to draw_feature_distribution_v2
# from draw_feature_distribution import *
# from draw_feature_distribution_v1 import *
#from draw_feature_distribution_v2 import *

# C:\Users\jbalcomb\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning:
# The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.

#%%

print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))
print("SciPy version: {}".format(sp.__version__))
print("scikit-learn version: {}".format(sklearn.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("IPython version: {}".format(IPython.__version__))
print()


#%%

random_seed = 1234567890
random.seed(random_seed)
np.random.seed(random_seed)


#%%

# Set configuration based on environment variables
if os.getenv('HCDR_DATA_FILE_PATH'):
    print('Using Environment Variable for data_file_path')
    data_file_path = os.getenv('HCDR_DATA_FILE_PATH')
    data_file_path = os.path.join(data_file_path, '')
#TODO(JamesBalcomb): add code to fall back on .config file
#else:
#    'kaggle--home-credit-default-risk.config'
else:
    print('Using Hard-Coded Configuration for data_file_path')
    data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
    data_file_path = os.path.join(data_file_path, '')
print('data_file_path: ', data_file_path)
print()


#%%

# Set configuration based on environment variables
if os.getenv('HCDR_WORKING_DIRECTORY'):
    print('Using Environment Variable for working_directory')
    working_directory = os.getenv('HCDR_WORKING_DIRECTORY')
    working_directory = os.path.join(working_directory, '')
#TODO(JamesBalcomb): add code to fall back on .config file
#else:
#    'kaggle--home-credit-default-risk.config'
else:
    print('Using Hard-Coded Configuration for working_directory')
    working_directory = 'C:/Development/kaggle--home-credit-default-risk/'
    working_directory = os.path.join(working_directory, '')
print('working_directory: ', working_directory)
print()


#%%

# TODO(JamesBalcomb): add a function that handles specifying multiple files
data_file_name = 'application_train.csv'
print('data_file_name: ', data_file_name)
print()


#%%

print('Importing data file...')
print()

df = pd.read_table(data_file_path + data_file_name, sep=',')


#%%

print('Fixing dtypes...')
print()

df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('category')  
df['TARGET'] = df['TARGET'].astype('category')  
df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].astype('category')  
df['CODE_GENDER'] = df['CODE_GENDER'].astype('category')  
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].astype('category')  
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].astype('category')  
df['CNT_CHILDREN'] = df['CNT_CHILDREN'].astype('float64')  
df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].astype('float64')  
df['AMT_CREDIT'] = df['AMT_CREDIT'].astype('float64')  
df['AMT_ANNUITY'] = df['AMT_ANNUITY'].astype('float64')  
df['AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'].astype('float64')  
df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].astype('category')  
df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].astype('category')  
df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype('category', ordered=True)  
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].astype('category')  
df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('category')  
df['REGION_POPULATION_RELATIVE'] = df['REGION_POPULATION_RELATIVE'].astype('float64')  
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].astype('float64')  
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].astype('float64')  
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].astype('float64')  
df['DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH'].astype('float64')  
df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].astype('float64')  
df['FLAG_MOBIL'] = df['FLAG_MOBIL'].astype('category')  
df['FLAG_EMP_PHONE'] = df['FLAG_EMP_PHONE'].astype('category')  
df['FLAG_WORK_PHONE'] = df['FLAG_WORK_PHONE'].astype('category')  
df['FLAG_CONT_MOBILE'] = df['FLAG_CONT_MOBILE'].astype('category')  
df['FLAG_PHONE'] = df['FLAG_PHONE'].astype('category')  
df['FLAG_EMAIL'] = df['FLAG_EMAIL'].astype('category')  
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].astype('category')  
df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype('float64')  
df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].astype('category', ordered=True)  
df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].astype('category', ordered=True)  
df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].astype('category')  
df['HOUR_APPR_PROCESS_START'] = df['HOUR_APPR_PROCESS_START'].astype('float64')  
df['REG_REGION_NOT_LIVE_REGION'] = df['REG_REGION_NOT_LIVE_REGION'].astype('category')  
df['REG_REGION_NOT_WORK_REGION'] = df['REG_REGION_NOT_WORK_REGION'].astype('category')  
df['LIVE_REGION_NOT_WORK_REGION'] = df['LIVE_REGION_NOT_WORK_REGION'].astype('category')  
df['REG_CITY_NOT_LIVE_CITY'] = df['REG_CITY_NOT_LIVE_CITY'].astype('category')  
df['REG_CITY_NOT_WORK_CITY'] = df['REG_CITY_NOT_WORK_CITY'].astype('category')  
df['LIVE_CITY_NOT_WORK_CITY'] = df['LIVE_CITY_NOT_WORK_CITY'].astype('category')  
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')  
df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].astype('float64')  
df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].astype('float64')  
df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].astype('float64')  
df['APARTMENTS_AVG'] = df['APARTMENTS_AVG'].astype('float64')  
df['BASEMENTAREA_AVG'] = df['BASEMENTAREA_AVG'].astype('float64')  
df['YEARS_BEGINEXPLUATATION_AVG'] = df['YEARS_BEGINEXPLUATATION_AVG'].astype('float64')  
df['YEARS_BUILD_AVG'] = df['YEARS_BUILD_AVG'].astype('float64')  
df['COMMONAREA_AVG'] = df['COMMONAREA_AVG'].astype('float64')  
df['ELEVATORS_AVG'] = df['ELEVATORS_AVG'].astype('float64')  
df['ENTRANCES_AVG'] = df['ENTRANCES_AVG'].astype('float64')  
df['FLOORSMAX_AVG'] = df['FLOORSMAX_AVG'].astype('float64')  
df['FLOORSMIN_AVG'] = df['FLOORSMIN_AVG'].astype('float64')  
df['LANDAREA_AVG'] = df['LANDAREA_AVG'].astype('float64')  
df['LIVINGAPARTMENTS_AVG'] = df['LIVINGAPARTMENTS_AVG'].astype('float64')  
df['LIVINGAREA_AVG'] = df['LIVINGAREA_AVG'].astype('float64')  
df['NONLIVINGAPARTMENTS_AVG'] = df['NONLIVINGAPARTMENTS_AVG'].astype('float64')  
df['NONLIVINGAREA_AVG'] = df['NONLIVINGAREA_AVG'].astype('float64')  
df['APARTMENTS_MODE'] = df['APARTMENTS_MODE'].astype('float64')  
df['BASEMENTAREA_MODE'] = df['BASEMENTAREA_MODE'].astype('float64')  
df['YEARS_BEGINEXPLUATATION_MODE'] = df['YEARS_BEGINEXPLUATATION_MODE'].astype('float64')  
df['YEARS_BUILD_MODE'] = df['YEARS_BUILD_MODE'].astype('float64')  
df['COMMONAREA_MODE'] = df['COMMONAREA_MODE'].astype('float64')  
df['ELEVATORS_MODE'] = df['ELEVATORS_MODE'].astype('float64')  
df['ENTRANCES_MODE'] = df['ENTRANCES_MODE'].astype('float64')  
df['FLOORSMAX_MODE'] = df['FLOORSMAX_MODE'].astype('float64')  
df['FLOORSMIN_MODE'] = df['FLOORSMIN_MODE'].astype('float64')  
df['LANDAREA_MODE'] = df['LANDAREA_MODE'].astype('float64')  
df['LIVINGAPARTMENTS_MODE'] = df['LIVINGAPARTMENTS_MODE'].astype('float64')  
df['LIVINGAREA_MODE'] = df['LIVINGAREA_MODE'].astype('float64')  
df['NONLIVINGAPARTMENTS_MODE'] = df['NONLIVINGAPARTMENTS_MODE'].astype('float64')  
df['NONLIVINGAREA_MODE'] = df['NONLIVINGAREA_MODE'].astype('float64')  
df['APARTMENTS_MEDI'] = df['APARTMENTS_MEDI'].astype('float64')  
df['BASEMENTAREA_MEDI'] = df['BASEMENTAREA_MEDI'].astype('float64')  
df['YEARS_BEGINEXPLUATATION_MEDI'] = df['YEARS_BEGINEXPLUATATION_MEDI'].astype('float64')  
df['YEARS_BUILD_MEDI'] = df['YEARS_BUILD_MEDI'].astype('float64')  
df['COMMONAREA_MEDI'] = df['COMMONAREA_MEDI'].astype('float64')  
df['ELEVATORS_MEDI'] = df['ELEVATORS_MEDI'].astype('float64')  
df['ENTRANCES_MEDI'] = df['ENTRANCES_MEDI'].astype('float64')  
df['FLOORSMAX_MEDI'] = df['FLOORSMAX_MEDI'].astype('float64')  
df['FLOORSMIN_MEDI'] = df['FLOORSMIN_MEDI'].astype('float64')  
df['LANDAREA_MEDI'] = df['LANDAREA_MEDI'].astype('float64')  
df['LIVINGAPARTMENTS_MEDI'] = df['LIVINGAPARTMENTS_MEDI'].astype('float64')  
df['LIVINGAREA_MEDI'] = df['LIVINGAREA_MEDI'].astype('float64')  
df['NONLIVINGAPARTMENTS_MEDI'] = df['NONLIVINGAPARTMENTS_MEDI'].astype('float64')  
df['NONLIVINGAREA_MEDI'] = df['NONLIVINGAREA_MEDI'].astype('float64')  
df['FONDKAPREMONT_MODE'] = df['FONDKAPREMONT_MODE'].astype('category')  
df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].astype('category')  
df['TOTALAREA_MODE'] = df['TOTALAREA_MODE'].astype('float64')  
df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].astype('category')  
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')  
df['OBS_30_CNT_SOCIAL_CIRCLE'] = df['OBS_30_CNT_SOCIAL_CIRCLE'].astype('float64')  
df['DEF_30_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'].astype('float64')  
df['OBS_60_CNT_SOCIAL_CIRCLE'] = df['OBS_60_CNT_SOCIAL_CIRCLE'].astype('float64')  
df['DEF_60_CNT_SOCIAL_CIRCLE'] = df['DEF_60_CNT_SOCIAL_CIRCLE'].astype('float64')  
df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].astype('float64')  
df['FLAG_DOCUMENT_2'] = df['FLAG_DOCUMENT_2'].astype('category')  
df['FLAG_DOCUMENT_3'] = df['FLAG_DOCUMENT_3'].astype('category')  
df['FLAG_DOCUMENT_4'] = df['FLAG_DOCUMENT_4'].astype('category')  
df['FLAG_DOCUMENT_5'] = df['FLAG_DOCUMENT_5'].astype('category')  
df['FLAG_DOCUMENT_6'] = df['FLAG_DOCUMENT_6'].astype('category')  
df['FLAG_DOCUMENT_7'] = df['FLAG_DOCUMENT_7'].astype('category')  
df['FLAG_DOCUMENT_8'] = df['FLAG_DOCUMENT_8'].astype('category')  
df['FLAG_DOCUMENT_9'] = df['FLAG_DOCUMENT_9'].astype('category')  
df['FLAG_DOCUMENT_10'] = df['FLAG_DOCUMENT_10'].astype('category')  
df['FLAG_DOCUMENT_11'] = df['FLAG_DOCUMENT_11'].astype('category')  
df['FLAG_DOCUMENT_12'] = df['FLAG_DOCUMENT_12'].astype('category')  
df['FLAG_DOCUMENT_13'] = df['FLAG_DOCUMENT_13'].astype('category')  
df['FLAG_DOCUMENT_14'] = df['FLAG_DOCUMENT_14'].astype('category')  
df['FLAG_DOCUMENT_15'] = df['FLAG_DOCUMENT_15'].astype('category')  
df['FLAG_DOCUMENT_16'] = df['FLAG_DOCUMENT_16'].astype('category')  
df['FLAG_DOCUMENT_17'] = df['FLAG_DOCUMENT_17'].astype('category')  
df['FLAG_DOCUMENT_18'] = df['FLAG_DOCUMENT_18'].astype('category')  
df['FLAG_DOCUMENT_19'] = df['FLAG_DOCUMENT_19'].astype('category')  
df['FLAG_DOCUMENT_20'] = df['FLAG_DOCUMENT_20'].astype('category')  
df['FLAG_DOCUMENT_21'] = df['FLAG_DOCUMENT_21'].astype('category')  
df['AMT_REQ_CREDIT_BUREAU_HOUR'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'].astype('float64')  
df['AMT_REQ_CREDIT_BUREAU_DAY'] = df['AMT_REQ_CREDIT_BUREAU_DAY'].astype('float64')  
df['AMT_REQ_CREDIT_BUREAU_WEEK'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'].astype('float64')  
df['AMT_REQ_CREDIT_BUREAU_MON'] = df['AMT_REQ_CREDIT_BUREAU_MON'].astype('float64')  
df['AMT_REQ_CREDIT_BUREAU_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'].astype('float64')  
df['AMT_REQ_CREDIT_BUREAU_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'].astype('float64')  


#%%

print('Applying transformations...')
print()

#df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
df.loc[df.SK_ID_CURR == 141289, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 319880, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 196708, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 144669, 'CODE_GENDER'] = 'M'
df['CODE_GENDER'] = pd.Series(np.where(df['CODE_GENDER'].values == 'M', 1, 0), df.index)

df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

#df['EMERGENCYSTATE_MODE'].add_categories('MISSING')
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('object')
df['EMERGENCYSTATE_MODE'].fillna('MISSING', inplace=True)
df['EMERGENCYSTATE_MODE__MISSING'] = pd.Series(np.where(df['EMERGENCYSTATE_MODE'].values == 'MISSING', 1, 0), df.index)
#df['EMERGENCYSTATE_MODE__Yes'] = pd.Series(np.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
df['EMERGENCYSTATE_MODE'] = pd.Series(np.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')

df['EXT_SOURCE_mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_median'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)

#df.drop('FLAG_MOBIL', inplace=True)

df['FLAG_OWN_CAR'] = pd.Series(np.where(df['FLAG_OWN_CAR'].values == 'Y', 1, 0), df.index)

df['FLAG_OWN_REALTY'] = pd.Series(np.where(df['FLAG_OWN_REALTY'].values == 'Y', 1, 0), df.index)

df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace('XNA', np.nan)
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('object')  
df['ORGANIZATION_TYPE'].fillna('MISSING', inplace=True)
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')  

#df.rename(columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE__Revolving_loans'}, inplace=True)
#df['NAME_CONTRACT_TYPE__Revolving_loans'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE__Revolving_loans'].values == 'Revolving loans', 1, 0), df.index)
df['NAME_CONTRACT_TYPE'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE'].values == 'Revolving loans', 1, 0), df.index)


#%%

# Machine Learning Pipe-Line
# Logistic Regression
# Paramters

# pandas.DateFrame
#    Y  Column Name
#    X  Column Name(s)

# sklearn.model_selection.train_test_split
#     *arrays
#     test_size=0.25
#     train_size=None
#     random_state=None
#     shuffle=True
#     stratify=None

# sklearn.linear_model.LogisticRegression
#     penalty='l2'
#     dual=False
#     tol=0.0001
#     C=1.0
#     fit_intercept=True
#     intercept_scaling=1
#     class_weight=None
#     random_state=None
#     solver='liblinear'
#     max_iter=100
#     multi_class='ovr'
#     verbose=0
#     warm_start=False
#     n_jobs=1

# sklearn.linear_model.LogisticRegression.fit()
#     Parameters
#         X: array-like
#         y: array-like
#         sample_weight
#     Returns
#         self

# sklearn.linear_model.LogisticRegression.predict()
#     Parameters
#         X: array-like
#     Returns
#         C: array-like

# sklearn.linear_model.LogisticRegression.predict_proba()
#     Parameters
#         X: array-like
#     Returns
#         T: array-like

# sklearn.linear_model.LogisticRegression.predict_log_proba()
#     Parameters
#         X: array-like
#     Returns
#         T: array-like

# sklearn.linear_model.LogisticRegression.decision_function()
#     Parameters
#         X: array-like
#     Returns
#         array

# sklearn.linear_model.LogisticRegression.score()
#     Parameters
#         X
#     Returns
#         C

# sklearn.linear_model.LogisticRegression.get_params(deep=True)
#     Parameters
#         deep=True
#     Returns
#         dictionary of LogisticRegression parameters

# roc_auc_score(y_test, logreg.predict_proba(X_test))
#     Parameters
#         X
#     Returns
#         C


#%%

# Machine Learning Pipe-Line
# Logistic Regression
# Outputs

# sklearn.linear_model.LogisticRegression
#     coef_
#     intercept_
#     n_iter_
#     classes_

# sklearn.linear_model.LogisticRegression.fit()
#     N/A

# sklearn.linear_model.LogisticRegression.predict()
#     

# sklearn.linear_model.LogisticRegression.predict_proba()
#     

# sklearn.linear_model.LogisticRegression.predict_log_proba()
#     

# sklearn.linear_model.LogisticRegression.decision_function()
#     

# sklearn.linear_model.LogisticRegression.score()
#     

# sklearn.linear_model.LogisticRegression.get_params(deep=True)
#     


#%%
class LogRegModel(object):
    pass


#%% 

y = df['TARGET']

X = df.loc[:,[
'CODE_GENDER',
'DAYS_EMPLOYED',
'EMERGENCYSTATE_MODE',
'EXT_SOURCE_1',
'EXT_SOURCE_2',
'EXT_SOURCE_3',
'FLAG_CONT_MOBILE',
'FLAG_EMAIL',
'FLAG_EMP_PHONE',
'FLAG_OWN_CAR',
'FLAG_OWN_REALTY',
'FLAG_PHONE',
'FLAG_WORK_PHONE',
'NAME_CONTRACT_TYPE'
#'ORGANIZATION_TYPE'
]]


#%%

print('Imputing Missing Values...')
print()

#X.isnull().any()

X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].fillna((X['DAYS_EMPLOYED'].mean()))

X['EXT_SOURCE_1'] = X['EXT_SOURCE_1'].fillna((X['EXT_SOURCE_1'].mean()))
X['EXT_SOURCE_2'] = X['EXT_SOURCE_2'].fillna((X['EXT_SOURCE_2'].mean()))
X['EXT_SOURCE_3'] = X['EXT_SOURCE_3'].fillna((X['EXT_SOURCE_3'].mean()))

#X = X.fillna(X.mean())

#df[column_name] = df[column_name].fillna((df[column_name].mean()))
#df[column_name] = df[column_name].fillna('MISSING_VALUE')

#pd.Series([X[c].value_counts().index[0]
#X.fillna(self.fill)

#pd.Series([X['FLAG_CONT_MOBILE'].value_counts().index[0])
#X['FLAG_CONT_MOBILE'].value_counts().index[0]
#df['FLAG_CONT_MOBILE'] = df['FLAG_CONT_MOBILE'].fillna(X['FLAG_CONT_MOBILE'].value_counts().index[0])


#%%

print(X.head())


#%%

# Test for NaN, infinity, or values to large for dtype float64
#> ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

#X.isnull().any()

#print(X.isinf().any())


#%%

#print('Standardizing Features...')
#print()

#scaler = StandardScaler()
#X = scaler.fit_transform(X)

#print(pd.DataFrame(X).head())


#%%

print()
print('Splitting Trainging and Testing data-sets...')
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)


#%%

# Logistic Regression Model
# Fitted Values
# Predicted Values
# Model Evaluation Scoring Metrics

print()
print('Instantiating LogisticRegression class...')
print()

logreg = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=0.0001,
        #C=1.0,
        C=20.0,
        fit_intercept=True,
        intercept_scaling=1,
        #class_weight=None,
        class_weight='balanced',
        #random_state=None,
        random_state=random_seed,
        solver='liblinear',
        max_iter=100,
        multi_class='ovr',
        #verbose=0,
        verbose=1,
        warm_start=False,
        n_jobs=1
        )

#%% 

print()
print('Training the model...')
print()

logreg.fit(X_train, y_train, sample_weight=None)


#%%

print()
print('Instantiating LogRegModel class...')
print()

log_reg_model = LogRegModel()


#%%

print()
print('Assigning log_reg_model.params via a call to the LogisticRegression.get_params method...')
print()

log_reg_model.params = logreg.get_params()


#%%

print()
print('Logistic Regression Estimator/Model Parameters')
print(logreg.get_params())
print()
print(log_reg_model.params)
print()


#%%

logreg.score(X_train, y_train, sample_weight=None)
logreg.score(X_test, y_test, sample_weight=None)


#%%

# Make predictions
X_test_predicted_class_labels = logreg.predict(X_test)
df_X_test_predicted_class_labels = pd.DataFrame(X_test_predicted_class_labels)
df_X_test_predicted_class_labels.columns = ['Late_Payments']

X_test_predicted_probability_estimates = logreg.predict_proba(X_test)
df_X_test_predicted_probability_estimates = pd.DataFrame(X_test_predicted_probability_estimates)
df_X_test_predicted_probability_estimates.columns = ['OnTime_Payments_Probability', 'Late_Payments_Probability']

X_test_predicted_log_of_probability_estimates = logreg.predict_log_proba(X_test)
df_X_test_predicted_log_of_probability_estimates = pd.DataFrame(X_test_predicted_log_of_probability_estimates)
#df_X_test_predicted_log_of_probability_estimates.columns = ['']

X_test_predicted_confidence_scores = logreg.decision_function(X_test)
df_X_test_predicted_confidence_scores = pd.DataFrame(X_test_predicted_confidence_scores)
#df_X_test_predicted_confidence_scores.columns = ['']


#%%

# Generate table of predictions
#pd.crosstab(X_train['FLAG_CONT_MOBILE'], df_X_test_predicted_probability_estimates.ix[:, 'Late_Payments_Probability'])

# Generate table of predictions vs actual
pd.crosstab(X_test_predicted_class_labels, y_test)

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, X_test_predicted_class_labels)
print(confusion_matrix)

# The confusion matrix below is not visually super informative or visually appealing.
cm = metrics.confusion_matrix(y_test, X_test_predicted_class_labels)
print(cm)

# Compute precision, recall, F-measure and support
print(classification_report(y_test, X_test_predicted_class_labels))

# ROC Curve
#logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test))
logit_roc_auc = roc_auc_score(y_test, X_test_predicted_class_labels)
print('logit_roc_auc: ', logit_roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, X_test_predicted_probability_estimates[:,1])
#print('fpr, tpr, thresholds: ', fpr, tpr, thresholds)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.5f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#%%

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# sklearn.linear_model.LogisticRegression
# class sklearn.linear_model.LogisticRegression
#   (
#       penalty=’l2’,
#       dual=False,
#       tol=0.0001,
#       C=1.0,
#       fit_intercept=True,
#       intercept_scaling=1,
#       class_weight=None,
#       random_state=None,
#       solver=’liblinear’,
#       max_iter=100,
#       multi_class=’ovr’,
#       verbose=0,
#       warm_start=False,
#       n_jobs=1
#   )
# Logistic Regression (aka logit, MaxEnt) classifier.
# This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers.
# It can handle both dense and sparse input.
# Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance;
#   any other input format will be converted (and copied).
#
# Parameters:	
#
#   penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
#       Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
#       New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)
#   dual : bool, default: False
#       Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
#   tol : float, default: 1e-4
#       Tolerance for stopping criteria.
#   C : float, default: 1.0
#       Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
#   fit_intercept : bool, default: True
#       Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
#   intercept_scaling : float, default 1.
#       Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.
#       Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
#   class_weight : dict or ‘balanced’, default: None
#       Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
#       The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
#       Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
#       New in version 0.17: class_weight=’balanced’
#   random_state : int, RandomState instance or None, optional, default: None
#       The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.
#   solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
#       default: ‘liblinear’ Algorithm to use in the optimization problem.
#       For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
#       For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes. ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
#       Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
#       New in version 0.17: Stochastic Average Gradient descent solver.
#       New in version 0.19: SAGA solver.
#   max_iter : int, default: 100
#       Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
#   multi_class : str, {‘ovr’, ‘multinomial’}, default: ‘ovr’
#       Multiclass option can be either ‘ovr’ or ‘multinomial’. If the option chosen is ‘ovr’, then a binary problem is fit for each label. Else the loss minimised is the multinomial loss fit across the entire probability distribution. Does not work for liblinear solver.
#       New in version 0.18: Stochastic Average Gradient descent solver for ‘multinomial’ case.
#   verbose : int, default: 0
#       For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
#   warm_start : bool, default: False
#       When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver.
#       New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.
#   n_jobs : int, default: 1
#       Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the ``solver``is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. If given a value of -1, all cores are used.
#
#
# Attributes:	
#
#   coef_ : array, shape (1, n_features) or (n_classes, n_features)
#       Coefficient of the features in the decision function.
#       coef_ is of shape (1, n_features) when the given problem is binary.
#   intercept_ : array, shape (1,) or (n_classes,)
#    Intercept (a.k.a. bias) added to the decision function.
#       If fit_intercept is set to False, the intercept is set to zero. intercept_ is of shape(1,) when the problem is binary.
#   n_iter_ : array, shape (n_classes,) or (1, )
#       Actual number of iterations for all classes. If binary or multinomial, it returns only 1 element. For liblinear solver, only the maximum number of iteration across all classes is given.
#
#
#
# http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
