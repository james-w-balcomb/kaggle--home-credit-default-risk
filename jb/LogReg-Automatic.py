#%%

import matplotlib
import numpy
import os
import pandas
import random
import sklearn
import statsmodels.api as sm

from patsy import dmatrices
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

import ats_functions
import hcdr_functions

import SklearnLogisticRegressionModel
import StatsmodelsLogisticRegressionModel


#%%

RANDOM_SEED = 1234567890
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)


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

df = pandas.read_table(
        data_file_path + data_file_name,
        sep=',',
        dtype={
                'AMT_ANNUITY':'float64',
                'AMT_CREDIT':'float64',
                'AMT_GOODS_PRICE':'float64',
                'AMT_INCOME_TOTAL':'float64',
                'AMT_REQ_CREDIT_BUREAU_DAY':'float64',
                'AMT_REQ_CREDIT_BUREAU_HOUR':'float64',
                'AMT_REQ_CREDIT_BUREAU_MON':'float64',
                'AMT_REQ_CREDIT_BUREAU_QRT':'float64',
                'AMT_REQ_CREDIT_BUREAU_WEEK':'float64',
                'AMT_REQ_CREDIT_BUREAU_YEAR':'float64',
                'APARTMENTS_AVG':'float64',
                'APARTMENTS_MEDI':'float64',
                'APARTMENTS_MODE':'float64',
                'BASEMENTAREA_AVG':'float64',
                'BASEMENTAREA_MEDI':'float64',
                'BASEMENTAREA_MODE':'float64',
                'CNT_CHILDREN':'float64',
                'CNT_FAM_MEMBERS':'float64',
                'CODE_GENDER':'object',
                'COMMONAREA_AVG':'float64',
                'COMMONAREA_MEDI':'float64',
                'COMMONAREA_MODE':'float64',
                'DAYS_BIRTH':'float64',
                'DAYS_EMPLOYED':'float64',
                'DAYS_ID_PUBLISH':'float64',
                'DAYS_LAST_PHONE_CHANGE':'float64',
                'DAYS_REGISTRATION':'float64',
                'DEF_30_CNT_SOCIAL_CIRCLE':'float64',
                'DEF_60_CNT_SOCIAL_CIRCLE':'float64',
                'ELEVATORS_AVG':'float64',
                'ELEVATORS_MEDI':'float64',
                'ELEVATORS_MODE':'float64',
                'EMERGENCYSTATE_MODE':'object',
                'ENTRANCES_AVG':'float64',
                'ENTRANCES_MEDI':'float64',
                'ENTRANCES_MODE':'float64',
                'EXT_SOURCE_1':'float64',
                'EXT_SOURCE_2':'float64',
                'EXT_SOURCE_3':'float64',
                'FLAG_CONT_MOBILE':'object',
                'FLAG_DOCUMENT_2':'object',
                'FLAG_DOCUMENT_3':'object',
                'FLAG_DOCUMENT_4':'object',
                'FLAG_DOCUMENT_5':'object',
                'FLAG_DOCUMENT_6':'object',
                'FLAG_DOCUMENT_7':'object',
                'FLAG_DOCUMENT_8':'object',
                'FLAG_DOCUMENT_9':'object',
                'FLAG_DOCUMENT_10':'object',
                'FLAG_DOCUMENT_11':'object',
                'FLAG_DOCUMENT_12':'object',
                'FLAG_DOCUMENT_13':'object',
                'FLAG_DOCUMENT_14':'object',
                'FLAG_DOCUMENT_15':'object',
                'FLAG_DOCUMENT_16':'object',
                'FLAG_DOCUMENT_17':'object',
                'FLAG_DOCUMENT_18':'object',
                'FLAG_DOCUMENT_19':'object',
                'FLAG_DOCUMENT_20':'object',
                'FLAG_DOCUMENT_21':'object',
                'FLAG_EMAIL':'object',
                'FLAG_EMP_PHONE':'object',
                'FLAG_MOBIL':'object',
                'FLAG_OWN_CAR':'object',
                'FLAG_OWN_REALTY':'object',
                'FLAG_PHONE':'object',
                'FLAG_WORK_PHONE':'object',
                'FLOORSMAX_AVG':'float64',
                'FLOORSMAX_MEDI':'float64',
                'FLOORSMAX_MODE':'float64',
                'FLOORSMIN_AVG':'float64',
                'FLOORSMIN_MEDI':'float64',
                'FLOORSMIN_MODE':'float64',
                'FONDKAPREMONT_MODE':'object',
                'HOUR_APPR_PROCESS_START':'float64',
                'HOUSETYPE_MODE':'object',
                'LANDAREA_AVG':'float64',
                'LANDAREA_MEDI':'float64',
                'LANDAREA_MODE':'float64',
                'LIVE_CITY_NOT_WORK_CITY':'object',
                'LIVE_REGION_NOT_WORK_REGION':'object',
                'LIVINGAPARTMENTS_AVG':'float64',
                'LIVINGAPARTMENTS_MEDI':'float64',
                'LIVINGAPARTMENTS_MODE':'float64',
                'LIVINGAREA_AVG':'float64',
                'LIVINGAREA_MEDI':'float64',
                'LIVINGAREA_MODE':'float64',
                'NAME_CONTRACT_TYPE':'object',
                'NAME_EDUCATION_TYPE':'object',
                'NAME_FAMILY_STATUS':'object',
                'NAME_HOUSING_TYPE':'object',
                'NAME_INCOME_TYPE':'object',
                'NAME_TYPE_SUITE':'object',
                'NONLIVINGAPARTMENTS_AVG':'float64',
                'NONLIVINGAPARTMENTS_MEDI':'float64',
                'NONLIVINGAPARTMENTS_MODE':'float64',
                'NONLIVINGAREA_AVG':'float64',
                'NONLIVINGAREA_MEDI':'float64',
                'NONLIVINGAREA_MODE':'float64',
                'OBS_30_CNT_SOCIAL_CIRCLE':'float64',
                'OBS_60_CNT_SOCIAL_CIRCLE':'float64',
                'OCCUPATION_TYPE':'object',
                'ORGANIZATION_TYPE':'object',
                'OWN_CAR_AGE':'float64',
                'REG_CITY_NOT_LIVE_CITY':'object',
                'REG_CITY_NOT_WORK_CITY':'object',
                'REG_REGION_NOT_LIVE_REGION':'object',
                'REG_REGION_NOT_WORK_REGION':'object',
                'REGION_POPULATION_RELATIVE':'float64',
                'REGION_RATING_CLIENT':'object',
                'REGION_RATING_CLIENT_W_CITY':'object',
                'SK_ID_CURR':'object',
                'TARGET':'object',
                'TOTALAREA_MODE':'float64',
                'WALLSMATERIAL_MODE':'object',
                'WEEKDAY_APPR_PROCESS_START':'object',
                'YEARS_BEGINEXPLUATATION_AVG':'float64',
                'YEARS_BEGINEXPLUATATION_MEDI':'float64',
                'YEARS_BEGINEXPLUATATION_MODE':'float64',
                'YEARS_BUILD_AVG':'float64',
                'YEARS_BUILD_MEDI':'float64',
                'YEARS_BUILD_MODE':'float64'
                }
        )


#%%

def fix__CODE_GENDER():
    #df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
    df.loc[df.SK_ID_CURR == 141289, 'CODE_GENDER'] = 'F'
    df.loc[df.SK_ID_CURR == 319880, 'CODE_GENDER'] = 'F'
    df.loc[df.SK_ID_CURR == 196708, 'CODE_GENDER'] = 'F'
    df.loc[df.SK_ID_CURR == 144669, 'CODE_GENDER'] = 'M'
    df['CODE_GENDER'] = pandas.Series(numpy.where(df['CODE_GENDER'].values == 'M', 1, 0), df.index)

def fix__DAYS_EMPLOYED():
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, numpy.nan)

def fix__EMERGENCYSTATE_MODE():
    #df['EMERGENCYSTATE_MODE'].add_categories('MISSING')
    df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('object')
    df['EMERGENCYSTATE_MODE'].fillna('MISSING', inplace=True)
    df['EMERGENCYSTATE_MODE__MISSING'] = pandas.Series(numpy.where(df['EMERGENCYSTATE_MODE'].values == 'MISSING', 1, 0), df.index)
    #df['EMERGENCYSTATE_MODE__Yes'] = pd.Series(np.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
    df['EMERGENCYSTATE_MODE'] = pandas.Series(numpy.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
    df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')

#df.drop('FLAG_MOBIL', inplace=True)

def fix__FLAG_OWN_CAR():
    df['FLAG_OWN_CAR'] = pandas.Series(numpy.where(df['FLAG_OWN_CAR'].values == 'Y', 1, 0), df.index)

def fix__FLAG_OWN_REALTY():
    df['FLAG_OWN_REALTY'] = pandas.Series(numpy.where(df['FLAG_OWN_REALTY'].values == 'Y', 1, 0), df.index)

def fix__ORGANIZATION_TYPE():
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace('XNA', numpy.nan)
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('object')  
    df['ORGANIZATION_TYPE'].fillna('MISSING', inplace=True)
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')  

def fix__NAME_CONTRACT_TYPE():
    #df.rename(columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE__Revolving_loans'}, inplace=True)
    #df['NAME_CONTRACT_TYPE__Revolving_loans'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE__Revolving_loans'].values == 'Revolving loans', 1, 0), df.index)
    df['NAME_CONTRACT_TYPE'] = pandas.Series(numpy.where(df['NAME_CONTRACT_TYPE'].values == 'Revolving loans', 1, 0), df.index)


#%%

print('Applying transformations...')
print()

fix__CODE_GENDER()
fix__DAYS_EMPLOYED()
fix__EMERGENCYSTATE_MODE()
fix__FLAG_OWN_CAR()
fix__FLAG_OWN_REALTY()
fix__ORGANIZATION_TYPE()
fix__NAME_CONTRACT_TYPE()


#%%

df['EXT_SOURCE_mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_median'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)


#%%

def make__capped_iqr__AMT_INCOME_TOTAL():
    column_name = 'AMT_INCOME_TOTAL'
    # Computing IQR
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    IQR150 = IQR * 1.50
    Q1IQR150 = Q1 - IQR150
    Q3IQR150 = Q3 + IQR150
    print(Q1IQR150)
    print(Q3IQR150)
    
    df['capped_iqr__' + column_name] = df[column_name]
    #df._is_copy
    #df._is_view
    #df2.values.base is df.values.base
    #df['capped_iqr__' + column_name][df['capped_iqr__' + column_name] < Q1IQR150] = Q1IQR150
    #df['capped_iqr__' + column_name][df['capped_iqr__' + column_name] > Q3IQR150] = Q3IQR150
    # https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-part-4-c4216f84d388
    #df.loc[df['age'] > 10, 'score'] = 99
    df.loc[df['capped_iqr__' + column_name] < Q1IQR150, 'capped_iqr__' + column_name] = Q1IQR150
    df.loc[df['capped_iqr__' + column_name] < Q3IQR150, 'capped_iqr__' + column_name] = Q3IQR150


#%%

make__capped_iqr__AMT_INCOME_TOTAL()


#%%

df['CODE_GENDER__male'] = pandas.Series(numpy.where(df['CODE_GENDER'].values == 'M', 1, 0), df.index)


#%%

df['EXT_SOURCE_1__imputed_mean'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].mean())
df['EXT_SOURCE_2__imputed_mean'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].mean())
df['EXT_SOURCE_3__imputed_mean'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].mean())

df['EXT_SOURCE_1__imputed_median'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
df['EXT_SOURCE_2__imputed_median'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
df['EXT_SOURCE_3__imputed_median'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())


#%%

df['EXT_SOURCE__mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE__median'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)

df['EXT_SOURCE__mean_imputed_mean'] = df['EXT_SOURCE__mean'].fillna(df['EXT_SOURCE__mean'].mean())
df['EXT_SOURCE__median_imputed_median'] = df['EXT_SOURCE__median'].fillna(df['EXT_SOURCE__median'].median())


#%%

# ... [1]
df['ratio__AMT_CREDIT__AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

#
df['ratio__AMT_INCOME_TOTAL__AMT_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

# How much of the price of the good was financed?
df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

# ... [2]
df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

# ... [2]
df['ratio__DAYS_ID_PUBLISH__DAYS_BIRTH'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']

# ... [2]
df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

# ... [2]
df['ratio__DAYS_REGISTRATION__DAYS_BIRTH'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']


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

print('Imputing Missing Values...')
print()

#df.isnull().any()

#df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna((df['DAYS_EMPLOYED'].mean()))

df['DAYS_EMPLOYED__imputed_mean'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].mean())
df['DAYS_EMPLOYED__imputed_median'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())


df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna((df['EXT_SOURCE_1'].mean()))
df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].fillna((df['EXT_SOURCE_2'].mean()))
df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna((df['EXT_SOURCE_3'].mean()))

#X = X.fillna(X.mean())

#df[column_name] = df[column_name].fillna((df[column_name].mean()))
#df[column_name] = df[column_name].fillna('MISSING_VALUE')

#pd.Series([X[c].value_counts().index[0]
#X.fillna(self.fill)

#pd.Series([X['FLAG_CONT_MOBILE'].value_counts().index[0])
#X['FLAG_CONT_MOBILE'].value_counts().index[0]
#df['FLAG_CONT_MOBILE'] = df['FLAG_CONT_MOBILE'].fillna(X['FLAG_CONT_MOBILE'].value_counts().index[0])


#%%

# Column Name                                   Count of Missing Values
# AMT_ANNUITY                                     12
# AMT_GOODS_PRICE                                278
# DAYS_EMPLOYED                                55374
# DAYS_LAST_PHONE_CHANGE                           1

df['AMT_ANNUITY__imputed_mean'] = df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].mean())
df['AMT_ANNUITY__imputed_median'] = df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())

df['AMT_GOODS_PRICE__imputed_mean'] = df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].mean())
df['AMT_GOODS_PRICE__imputed_median'] = df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())

df['DAYS_EMPLOYED__imputed_mean'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].mean())
df['DAYS_EMPLOYED__imputed_median'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

df['DAYS_LAST_PHONE_CHANGE__imputed_mean'] = df['DAYS_LAST_PHONE_CHANGE'].fillna(df['DAYS_LAST_PHONE_CHANGE'].mean())
df['DAYS_LAST_PHONE_CHANGE__imputed_median'] = df['DAYS_LAST_PHONE_CHANGE'].fillna(df['DAYS_LAST_PHONE_CHANGE'].median())


#%%

# Column Name                                   Count of Missing Values
# ratio__AMT_CREDIT__AMT_ANNUITY                  12
# ratio__AMT_CREDIT_AMT_GOODS_PRICE              278
# ratio__DAYS_EMPLOYED__DAYS_BIRTH             55374
# ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH        1

df['ratio__AMT_CREDIT__AMT_ANNUITY__imputed_mean'] = df['ratio__AMT_CREDIT__AMT_ANNUITY'].fillna(df['ratio__AMT_CREDIT__AMT_ANNUITY'].mean())
df['ratio__AMT_CREDIT__AMT_ANNUITY__imputed_median'] = df['ratio__AMT_CREDIT__AMT_ANNUITY'].fillna(df['ratio__AMT_CREDIT__AMT_ANNUITY'].median())

df['ratio__AMT_CREDIT__AMT_GOODS_PRICE__imputed_mean'] = df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'].fillna(df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'].mean())
df['ratio__AMT_CREDIT__AMT_GOODS_PRICE__imputed_median'] = df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'].fillna(df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'].median())

df['ratio__DAYS_EMPLOYED__DAYS_BIRTH__imputed_mean'] = df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'].fillna(df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'].mean())
df['ratio__DAYS_EMPLOYED__DAYS_BIRTH__imputed_median'] = df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'].fillna(df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'].median())

df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH__imputed_mean'] = df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'].fillna(df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'].mean())
df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH__imputed_median'] = df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'].fillna(df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'].median())


#%%

flag_document_column_names = [
'FLAG_DOCUMENT_2',
'FLAG_DOCUMENT_3',
'FLAG_DOCUMENT_4',
'FLAG_DOCUMENT_5',
'FLAG_DOCUMENT_6',
'FLAG_DOCUMENT_7',
'FLAG_DOCUMENT_8',
'FLAG_DOCUMENT_9',
'FLAG_DOCUMENT_10',
'FLAG_DOCUMENT_11',
'FLAG_DOCUMENT_12',
'FLAG_DOCUMENT_13',
'FLAG_DOCUMENT_14',
'FLAG_DOCUMENT_15',
'FLAG_DOCUMENT_16',
'FLAG_DOCUMENT_17',
'FLAG_DOCUMENT_18',
'FLAG_DOCUMENT_19',
'FLAG_DOCUMENT_20',
'FLAG_DOCUMENT_21'
]

df['FLAG_DOCUMENT__count'] = df[flag_document_column_names].astype('int').sum(axis=1)


#%%

#df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('category')  
#df['TARGET'] = df['TARGET'].astype('category')  

df['CODE_GENDER'] = df['CODE_GENDER'].astype('category')  
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')  
df['FLAG_CONT_MOBILE'] = df['FLAG_CONT_MOBILE'].astype('category')  
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
df['FLAG_EMAIL'] = df['FLAG_EMAIL'].astype('category')  
df['FLAG_EMP_PHONE'] = df['FLAG_EMP_PHONE'].astype('category')  
df['FLAG_MOBIL'] = df['FLAG_MOBIL'].astype('category')  
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].astype('category')  
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].astype('category')  
df['FLAG_PHONE'] = df['FLAG_PHONE'].astype('category')  
df['FLAG_WORK_PHONE'] = df['FLAG_WORK_PHONE'].astype('category')  
df['FONDKAPREMONT_MODE'] = df['FONDKAPREMONT_MODE'].astype('category')  
df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].astype('category')  
df['LIVE_CITY_NOT_WORK_CITY'] = df['LIVE_CITY_NOT_WORK_CITY'].astype('category')  
df['LIVE_REGION_NOT_WORK_REGION'] = df['LIVE_REGION_NOT_WORK_REGION'].astype('category')  
df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].astype('category')  
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].astype('category')  
df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('category')  
df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].astype('category')  
df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].astype('category')  
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].astype('category')  
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')  
df['REG_CITY_NOT_LIVE_CITY'] = df['REG_CITY_NOT_LIVE_CITY'].astype('category')  
df['REG_CITY_NOT_WORK_CITY'] = df['REG_CITY_NOT_WORK_CITY'].astype('category')  
df['REG_REGION_NOT_LIVE_REGION'] = df['REG_REGION_NOT_LIVE_REGION'].astype('category')  
df['REG_REGION_NOT_WORK_REGION'] = df['REG_REGION_NOT_WORK_REGION'].astype('category')  
df['TARGET'] = df['TARGET'].astype('category')  
df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].astype('category')  
df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].astype('category')  

df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype(
        'category',
        ordered=True,
        categories=[
                'Academic degree',
                'Higher education',
                'Incomplete higher',
                'Secondary / secondary special',
                'Lower secondary'
                ]
        )
df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].astype(
        'category',
        ordered=True,
        categories=[
                3,
                2,
                1
                ]
        )
df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].astype(
        'category',
        ordered=True,
        categories=[
                3,
                2,
                1
                ]
        )


#%%

#df['capped_iqr__AMT_INCOME_TOTAL']
#df['ratio__AMT_CREDIT__AMT_ANNUITY']
#df['ratio__AMT_INCOME_TOTAL__AMT_CREDIT']
#df['ratio__AMT_CREDIT_AMT_GOODS_PRICE']
#df['ratio__DAYS_EMPLOYED__DAYS_BIRTH']
#df['ratio__DAYS_ID_PUBLISH__DAYS_BIRTH']
#df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH']
#df['ratio__DAYS_REGISTRATION__DAYS_BIRTH']


#%% 

#dependent_column_name = ['TARGET']
dependent_column_name = 'TARGET'


#%%

independent_column_names = [
##'AMT_ANNUITY',
'AMT_ANNUITY__imputed_mean',
#'AMT_ANNUITY__imputed_median',
'AMT_CREDIT',
##'AMT_GOODS_PRICE',
'AMT_GOODS_PRICE__imputed_mean',
#'AMT_GOODS_PRICE__imputed_median',
##'AMT_INCOME_TOTAL',
'capped_iqr__AMT_INCOME_TOTAL',
##'CODE_GENDER',
'CODE_GENDER__male',
'DAYS_BIRTH',
##'DAYS_EMPLOYED',
'DAYS_EMPLOYED__imputed_mean',
#'DAYS_EMPLOYED__imputed_median',
'DAYS_ID_PUBLISH',
##'DAYS_LAST_PHONE_CHANGE',
'DAYS_LAST_PHONE_CHANGE__imputed_mean',
#'DAYS_LAST_PHONE_CHANGE__imputed_median',
'DAYS_REGISTRATION',
'EMERGENCYSTATE_MODE',
##'EXT_SOURCE_1',
##'EXT_SOURCE_2',
##'EXT_SOURCE_3',
'EXT_SOURCE_1__imputed_mean',
'EXT_SOURCE_2__imputed_mean',
'EXT_SOURCE_3__imputed_mean',
#'EXT_SOURCE_1__imputed_median',
#'EXT_SOURCE_2__imputed_median',
#'EXT_SOURCE_3__imputed_median',
'EXT_SOURCE__mean_imputed_mean',
#'EXT_SOURCE__median_imputed_median',
'FLAG_CONT_MOBILE',
'FLAG_EMAIL',
'FLAG_EMP_PHONE',
'FLAG_OWN_CAR',
'FLAG_OWN_REALTY',
'FLAG_PHONE',
'FLAG_WORK_PHONE',
'NAME_CONTRACT_TYPE',
##'ratio__AMT_CREDIT__AMT_ANNUITY',
'ratio__AMT_CREDIT__AMT_ANNUITY__imputed_mean',
#'ratio__AMT_CREDIT__AMT_ANNUITY__imputed_median',
'ratio__AMT_INCOME_TOTAL__AMT_CREDIT',
##'ratio__AMT_CREDIT__AMT_GOODS_PRICE',
'ratio__AMT_CREDIT__AMT_GOODS_PRICE__imputed_mean',
#'ratio__AMT_CREDIT__AMT_GOODS_PRICE__imputed_median',
##'ratio__DAYS_EMPLOYED__DAYS_BIRTH',
'ratio__DAYS_EMPLOYED__DAYS_BIRTH__imputed_mean',
#'ratio__DAYS_EMPLOYED__DAYS_BIRTH__imputed_median',
'ratio__DAYS_ID_PUBLISH__DAYS_BIRTH',
##'ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH',
'ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH__imputed_mean',
#'ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH__imputed_median',
'ratio__DAYS_REGISTRATION__DAYS_BIRTH'
]


#%%

#numpy.any(numpy.isnan(df[independent_column_names]))
#> TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
numpy.any(pandas.isnull(df[independent_column_names]))
#pandas.isnull(df[independent_column_names]).values_counts()
df[independent_column_names].isnull().sum()

#%%

#numpy.all(numpy.isfinite(df[independent_column_names]))
#> TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


#%%
    
class LogRegModel(object):
    pass


#%%

logistic_regression_models = dict()

# sklearn.linear_model.LogisticRegression
# solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘liblinear’
solver_list = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
#solver_list = ['liblinear', 'sag']
#solver_list = ['liblinear']
# penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
penalty_list = ['l1', 'l2']
# C : float, default: 1.0
#c_list = [1.0, 20.0, 400.0, 8000.0]
c_list = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


# Configuration for testing purposes
solver_list = ['liblinear']
penalty_list = ['l1']
c_list = [1.0]



#%%

#range_end = (len(solver) * len(penalty) * len(c) * len(independent_column_names)) + 1
#print(range_end)

#model_count = (len(solver_list) * len(penalty_list) * len(c_list) * len(independent_column_names))
model_count = (len(solver_list) * len(penalty_list) * len(c_list))


#%%

print('Total Model Count: {}'.format(model_count))


#%%

#for i in range(1, range_end, 1):
current_model_number = 0

for solver in solver_list:
    for penalty in penalty_list:
        for c in c_list:
            
            current_model_number = current_model_number  + 1
            
            print()
            print('Logistic Regression Model: {} of {}'.format(current_model_number, model_count))
            print('Solver: {}'.format(solver))
            print('Penalty: {}'.format(penalty))
            print('C: {}'.format(c))
            print()
            
            #.../site-packages/sklearn/linear_model/tests/test_logistic.py
            #if solver == 'liblinear' and multi_class == 'multinomial':
            #        continue
            if solver == 'lbfgs' and penalty == 'l1':
                print()
                print('ValueError: Solver lbfgs supports only l2 penalties, got l1 penalty.')
                print()
                continue
            if solver == 'newton-cg' and penalty == 'l1':
                print()
                print('ValueError: Solver newton-cg supports only l2 penalties, got l1 penalty.')
                print()
                continue
            if solver == 'sag' and penalty == 'l1':
                print()
                print('ValueError: Solver sag supports only l2 penalties, got l1 penalty.')
                print()
                continue
            
            logistic_regression_models[current_model_number] = LogRegModel()  # sklearn
            logistic_regression_models[current_model_number].dependent_column_name = dependent_column_name
            logistic_regression_models[current_model_number].independent_column_names = independent_column_names
        
            print()
            print('Splitting Trainging and Testing data-sets...')
            print()
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                    df.loc[:,logistic_regression_models[current_model_number].independent_column_names],
                    df[logistic_regression_models[current_model_number].dependent_column_name],
                    test_size=0.3,
                    random_state=RANDOM_SEED
                    )
        
            logistic_regression_models[current_model_number].logistic_regression_parameters = {
                    #'penalty':'l2',
                    'penalty':penalty,
                    'dual':False,
                    'tol':0.0001,
                    #'C':1.0,
                    #'C':20.0,
                    'C':c,
                    'fit_intercept':True,
                    'intercept_scaling':1,
                    #'class_weight':None,
                    'class_weight':'balanced',
                    #'random_state':None,
                    'random_state':RANDOM_SEED,
                    #'solver':'liblinear',
                    'solver':solver,
                    'max_iter':100,
                    'multi_class':'ovr',
                    #'verbose':0,
                    'verbose':1,
                    'warm_start':False,
                    'n_jobs':1
                    }
            
            print()
            print('Instantiating LogisticRegression class...')
            print()
            
            logreg = sklearn.linear_model.LogisticRegression(**logistic_regression_models[current_model_number].logistic_regression_parameters)
            
            # TODO(JamesBalcomb): figure out how to capture and handle failure to converge
            #Solver: newton-cg
            #Penalty: l2
            #C: 20.0
            #LineSearchWarning: The line search algorithm did not converge
            
            print()
            print('Training the model...')
            print()
            
            logreg.fit(X_train, y_train, sample_weight=None)
            
            print()
            print('Collecting the model outputs...')
            print()
            
            logistic_regression_models[current_model_number].coefficients = logreg.coef_
            logistic_regression_models[current_model_number].intercept = logreg.intercept_
            logistic_regression_models[current_model_number].number_of_iterations = logreg.n_iter_
            
            logistic_regression_models[current_model_number].params = logreg.get_params()
            
            logistic_regression_models[current_model_number].X_test_predicted_class_labels = logreg.predict(X_test)
            #logistic_regression_models[i].df_X_test_predicted_class_labels = pandas.DataFrame(logistic_regression_models[i].X_test_predicted_class_labels)
            #logistic_regression_models[i].df_X_test_predicted_class_labels.columns = ['Late_Payments']
            
            logistic_regression_models[current_model_number].X_test_predicted_log_probability_estimates = logreg.predict_log_proba(X_test)
            #logistic_regression_models[i].df_X_test_predicted_log_probability_estimates = pandas.DataFrame(logistic_regression_models[i].X_test_predicted_log_probability_estimates)
            
            logistic_regression_models[current_model_number].X_test_predicted_probability_estimates = logreg.predict_proba(X_test)
            #logistic_regression_models[i].df_X_test_predicted_probability_estimates = pandas.DataFrame(logistic_regression_models[i].X_test_predicted_probability_estimates)
            #logistic_regression_models[i].df_X_test_predicted_probability_estimates.columns = ['OnTime_Payments_Probability', 'Late_Payments_Probability']
            
            logistic_regression_models[current_model_number].X_test_predicted_confidence_scores = logreg.decision_function(X_test)
            #logistic_regression_models[i].df_X_test_predicted_confidence_scores = pandas.DataFrame(logistic_regression_models[i].X_test_predicted_confidence_scores)
            
            logistic_regression_models[current_model_number].score_train = logreg.score(X_train, y_train, sample_weight=None)
            logistic_regression_models[current_model_number].score_test = logreg.score(X_test, y_test, sample_weight=None)
            logistic_regression_models[current_model_number].confusion_matrix = sklearn.metrics.confusion_matrix(y_test, logistic_regression_models[current_model_number].X_test_predicted_class_labels)
            logistic_regression_models[current_model_number].classification_report = sklearn.metrics.classification_report(y_test, logistic_regression_models[current_model_number].X_test_predicted_class_labels)
            logistic_regression_models[current_model_number].logit_roc_auc = sklearn.metrics.roc_auc_score(y_test, logistic_regression_models[current_model_number].X_test_predicted_class_labels)
            logistic_regression_models[current_model_number].fpr, logistic_regression_models[current_model_number].tpr, logistic_regression_models[current_model_number].thresholds = sklearn.metrics.roc_curve(y_test, logistic_regression_models[current_model_number].X_test_predicted_probability_estimates[:,1])
            logistic_regression_models[current_model_number].true_negative_count, logistic_regression_models[current_model_number].false_positive_count, logistic_regression_models[current_model_number].false_negative_count, logistic_regression_models[current_model_number].true_positive_count = logistic_regression_models[current_model_number].confusion_matrix.ravel()
            
            print()
            print('Displaying the model outputs...')
            print()
            
            print(logistic_regression_models[current_model_number].params)
            print(logistic_regression_models[current_model_number].number_of_iterations)
            print(logistic_regression_models[current_model_number].intercept)
            logistic_regression_models[current_model_number].coefficients_dict = dict(zip(logistic_regression_models[current_model_number].independent_column_names, list(logistic_regression_models[current_model_number].coefficients[0])))
            print(logistic_regression_models[current_model_number].coefficients_dict)
            
            print()
            print('Displaying the model results...')
            print()
            
            print('Mean Accuracy on training data-set: {:.5f}'.format(logistic_regression_models[current_model_number].score_train))
            print('Mean Accuracy on testing data-set: {:.5f}'.format(logistic_regression_models[current_model_number].score_test))
            #print(logistic_regression_models[i].confusion_matrix)
            print('True Positves:   {:>6,}'.format(logistic_regression_models[current_model_number].true_positive_count))
            print('False Positves:  {:>6,}'.format(logistic_regression_models[current_model_number].false_positive_count))
            print('False Negatives: {:>6,}'.format(logistic_regression_models[current_model_number].false_negative_count))
            print('True Negatives:  {:>6,}'.format(logistic_regression_models[current_model_number].true_negative_count))
            print('Classification Report:', '\n', logistic_regression_models[current_model_number].classification_report)
            print('Area Under the ROC Curve (AUC): {:.5f}'.format(logistic_regression_models[current_model_number].logit_roc_auc))
            
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(
                    logistic_regression_models[current_model_number].fpr,
                    logistic_regression_models[current_model_number].tpr,
                    label='Logistic Regression (AUC = %0.5f)' % logistic_regression_models[current_model_number].logit_roc_auc
                    )
            matplotlib.pyplot.plot([0, 1], [0, 1],'r--')
            matplotlib.pyplot.xlim([0.0, 1.0])
            matplotlib.pyplot.ylim([0.0, 1.05])
            matplotlib.pyplot.xlabel('False Positive Rate')
            matplotlib.pyplot.ylabel('True Positive Rate')
            matplotlib.pyplot.title('Receiver Operating Characteristic')
            matplotlib.pyplot.legend(loc="lower right")
            #matplotlib.pyplot.savefig('Log_ROC')
            matplotlib.pyplot.show()

print()
print('Completed Logistic Regression Models: {} of {}'.format(current_model_number, model_count))
print()
