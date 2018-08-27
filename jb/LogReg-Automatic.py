#%%

import matplotlib
import numpy
import os
import pandas
import random 
import sklearn
from sklearn import linear_model
#from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
#from sklearn.model_selection import train_test_split


#%%

random_seed = 1234567890
random.seed(random_seed)
numpy.random.seed(random_seed)


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

df = pandas.read_table(data_file_path + data_file_name, sep=',')


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
df['CODE_GENDER'] = pandas.Series(numpy.where(df['CODE_GENDER'].values == 'M', 1, 0), df.index)

df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, numpy.nan)

#df['EMERGENCYSTATE_MODE'].add_categories('MISSING')
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('object')
df['EMERGENCYSTATE_MODE'].fillna('MISSING', inplace=True)
df['EMERGENCYSTATE_MODE__MISSING'] = pandas.Series(numpy.where(df['EMERGENCYSTATE_MODE'].values == 'MISSING', 1, 0), df.index)
#df['EMERGENCYSTATE_MODE__Yes'] = pd.Series(np.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
df['EMERGENCYSTATE_MODE'] = pandas.Series(numpy.where(df['EMERGENCYSTATE_MODE'].values == 'Yes', 1, 0), df.index)
df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')

df['EXT_SOURCE_mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_median'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)

#df.drop('FLAG_MOBIL', inplace=True)

df['FLAG_OWN_CAR'] = pandas.Series(numpy.where(df['FLAG_OWN_CAR'].values == 'Y', 1, 0), df.index)

df['FLAG_OWN_REALTY'] = pandas.Series(numpy.where(df['FLAG_OWN_REALTY'].values == 'Y', 1, 0), df.index)

df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace('XNA', numpy.nan)
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('object')  
df['ORGANIZATION_TYPE'].fillna('MISSING', inplace=True)
df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')  

#df.rename(columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE__Revolving_loans'}, inplace=True)
#df['NAME_CONTRACT_TYPE__Revolving_loans'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE__Revolving_loans'].values == 'Revolving loans', 1, 0), df.index)
df['NAME_CONTRACT_TYPE'] = pandas.Series(numpy.where(df['NAME_CONTRACT_TYPE'].values == 'Revolving loans', 1, 0), df.index)


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

df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna((df['DAYS_EMPLOYED'].mean()))

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

print(df.head())


#%% 

dependent_column_name = 'TARGET'


#%%

independent_column_names = [
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
        ]


#%%
    
class LogRegModel(object):
    pass


#%% 

print()
print('Instantiating the LogRegModel class...')
print()

log_reg_model_01 = LogRegModel()


#%%

log_reg_model_01.dependent_column_name = dependent_column_name
log_reg_model_01.independent_column_names = independent_column_names


#%%

#y = df[dependent_column_name]
#y = df[log_reg_model_01.dependent_column_name]

#X = df.loc[:,independent_column_names]
#X = df.loc[:,log_reg_model_01.independent_column_names]


#%%

print()
print('Splitting Trainging and Testing data-sets...')
print()

#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=random_seed)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        df.loc[:,log_reg_model_01.independent_column_names],
        df[log_reg_model_01.dependent_column_name],
        test_size=0.3,
        random_state=random_seed
        )


#%%

log_reg_model_01.logistic_regression_parameters = {
        'penalty':'l2',
        'dual':False,
        'tol':0.0001,
        #'C':1.0,
        'C':20.0,
        'fit_intercept':True,
        'intercept_scaling':1,
        #'class_weight':None,
        'class_weight':'balanced',
        #'random_state':None,
        'random_state':random_seed,
        'solver':'liblinear',
        'max_iter':100,
        'multi_class':'ovr',
        #'verbose':0,
        'verbose':1,
        'warm_start':False,
        'n_jobs':1
        }


#%%

# Logistic Regression Model
# Fitted Values
# Predicted Values
# Model Evaluation Scoring Metrics

print()
print('Instantiating LogisticRegression class...')
print()

logreg = sklearn.linear_model.LogisticRegression(**log_reg_model_01.logistic_regression_parameters)


#%% 

print()
print('Training the model...')
print()

logreg.fit(X_train, y_train, sample_weight=None)


#%%

log_reg_model_01.coefficients = logreg.coef_
log_reg_model_01.intercept = logreg.intercept_
log_reg_model_01.number_of_iterations = logreg.n_iter_

log_reg_model_01.params = logreg.get_params()

log_reg_model_01.X_test_predicted_class_labels = logreg.predict(X_test)
log_reg_model_01.X_test_predicted_log_probability_estimates = logreg.predict_log_proba(X_test)
log_reg_model_01.X_test_predicted_probability_estimates = logreg.predict_proba(X_test)
log_reg_model_01.X_test_predicted_confidence_scores = logreg.decision_function(X_test)

log_reg_model_01.score_train = logreg.score(X_train, y_train, sample_weight=None)
log_reg_model_01.score_test = logreg.score(X_test, y_test, sample_weight=None)
log_reg_model_01.confusion_matrix = sklearn.metrics.confusion_matrix(y_test, log_reg_model_01.X_test_predicted_class_labels)
log_reg_model_01.classification_report = sklearn.metrics.classification_report(y_test, log_reg_model_01.X_test_predicted_class_labels)
log_reg_model_01.logit_roc_auc = sklearn.metrics.roc_auc_score(y_test, log_reg_model_01.X_test_predicted_class_labels)
log_reg_model_01.fpr, log_reg_model_01.tpr, log_reg_model_01.thresholds = sklearn.metrics.roc_curve(y_test, log_reg_model_01.X_test_predicted_probability_estimates[:,1])


#%%

log_reg_model_01.true_negative_count, log_reg_model_01.false_positive_count, log_reg_model_01.false_negative_count, log_reg_model_01.true_positive_count = log_reg_model_01.confusion_matrix.ravel()


#%%

log_reg_model_01.df_X_test_predicted_class_labels = pandas.DataFrame(log_reg_model_01.X_test_predicted_class_labels)
log_reg_model_01.df_X_test_predicted_class_labels.columns = ['Late_Payments']

log_reg_model_01.df_X_test_predicted_probability_estimates = pandas.DataFrame(log_reg_model_01.X_test_predicted_probability_estimates)
log_reg_model_01.df_X_test_predicted_probability_estimates.columns = ['OnTime_Payments_Probability', 'Late_Payments_Probability']

log_reg_model_01.df_X_test_predicted_log_probability_estimates = pandas.DataFrame(log_reg_model_01.X_test_predicted_log_probability_estimates)
#df_X_test_predicted_log_of_probability_estimates.columns = ['']

log_reg_model_01.df_X_test_predicted_confidence_scores = pandas.DataFrame(log_reg_model_01.X_test_predicted_confidence_scores)
#df_X_test_predicted_confidence_scores.columns = ['']


#%%

print(log_reg_model_01.params)


#%%

print(log_reg_model_01.number_of_iterations)


#%%

print(log_reg_model_01.intercept)


#%%

log_reg_model_01.coefficients_dict = dict(zip(log_reg_model_01.independent_column_names, list(log_reg_model_01.coefficients[0])))
print(log_reg_model_01.coefficients_dict)


#%%

print('Mean Accuracy on training data-set: {:.5f}'.format(log_reg_model_01.score_train))
print('Mean Accuracy on testing data-set: {:.5f}'.format(log_reg_model_01.score_test))


#%%

#print(log_reg_model_01.confusion_matrix)
print('True Positves:   {:>6,}'.format(log_reg_model_01.true_positive_count))
print('False Positves:  {:>6,}'.format(log_reg_model_01.false_positive_count))
print('False Negatives: {:>6,}'.format(log_reg_model_01.false_negative_count))
print('True Negatives:  {:>6,}'.format(log_reg_model_01.true_negative_count))


#%%

print('Classification Report:', '\n', log_reg_model_01.classification_report)


#%%

print('Area Under the ROC Curve (AUC): {:.5f}'.format(log_reg_model_01.logit_roc_auc))


#%%

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(log_reg_model_01.fpr, log_reg_model_01.tpr, label='Logistic Regression (AUC = %0.5f)' % log_reg_model_01.logit_roc_auc)
matplotlib.pyplot.plot([0, 1], [0, 1],'r--')
matplotlib.pyplot.xlim([0.0, 1.0])
matplotlib.pyplot.ylim([0.0, 1.05])
matplotlib.pyplot.xlabel('False Positive Rate')
matplotlib.pyplot.ylabel('True Positive Rate')
matplotlib.pyplot.title('Receiver Operating Characteristic')
matplotlib.pyplot.legend(loc="lower right")
#matplotlib.pyplot.savefig('Log_ROC')
matplotlib.pyplot.show()


#%% 

print()
print('Instantiating the LogRegModel class...')
print()

log_reg_model_02 = LogRegModel()


#%%

dependent_column_name = 'TARGET'

independent_column_names = [
'EXT_SOURCE_1',
'EXT_SOURCE_2',
'EXT_SOURCE_3'
]

log_reg_model_02.dependent_column_name = dependent_column_name
log_reg_model_02.independent_column_names = independent_column_names


#%%

#y = df[dependent_column_name]
#y = df[log_reg_model_02.dependent_column_name]

#X = df.loc[:,independent_column_names]
#X = df.loc[:,log_reg_model_02.independent_column_names]


#%%

print()
print('Splitting Trainging and Testing data-sets...')
print()

#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=random_seed)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        df.loc[:,log_reg_model_02.independent_column_names],
        df[log_reg_model_02.dependent_column_name],
        test_size=0.3,
        random_state=random_seed
        )


#%% 

log_reg_model_02.logistic_regression_parameters = {
        'penalty':'l2',
        'dual':False,
        'tol':0.0001,
        #'C':1.0,
        'C':20.0,
        'fit_intercept':True,
        'intercept_scaling':1,
        #'class_weight':None,
        'class_weight':'balanced',
        #'random_state':None,
        'random_state':random_seed,
        'solver':'liblinear',
        'max_iter':100,
        'multi_class':'ovr',
        #'verbose':0,
        'verbose':1,
        'warm_start':False,
        'n_jobs':1
        }


#%%

# Logistic Regression Model
# Fitted Values
# Predicted Values
# Model Evaluation Scoring Metrics

print()
print('Instantiating LogisticRegression class...')
print()

logreg = sklearn.linear_model.LogisticRegression(**log_reg_model_02.logistic_regression_parameters)


#%% 

print()
print('Training the model...')
print()

logreg.fit(X_train, y_train, sample_weight=None)


#%%

log_reg_model_02.coefficients = logreg.coef_
log_reg_model_02.intercept = logreg.intercept_
log_reg_model_02.number_of_iterations = logreg.n_iter_

log_reg_model_02.params = logreg.get_params()

log_reg_model_02.X_test_predicted_class_labels = logreg.predict(X_test)
log_reg_model_02.X_test_predicted_log_probability_estimates = logreg.predict_log_proba(X_test)
log_reg_model_02.X_test_predicted_probability_estimates = logreg.predict_proba(X_test)
log_reg_model_02.X_test_predicted_confidence_scores = logreg.decision_function(X_test)

log_reg_model_02.score_train = logreg.score(X_train, y_train, sample_weight=None)
log_reg_model_02.score_test = logreg.score(X_test, y_test, sample_weight=None)
log_reg_model_02.confusion_matrix = sklearn.metrics.confusion_matrix(y_test, log_reg_model_02.X_test_predicted_class_labels)
log_reg_model_02.classification_report = sklearn.metrics.classification_report(y_test, log_reg_model_02.X_test_predicted_class_labels)
log_reg_model_02.logit_roc_auc = sklearn.metrics.roc_auc_score(y_test, log_reg_model_02.X_test_predicted_class_labels)
log_reg_model_02.fpr, log_reg_model_02.tpr, log_reg_model_02.thresholds = sklearn.metrics.roc_curve(y_test, log_reg_model_02.X_test_predicted_probability_estimates[:,1])


#%%

log_reg_model_02.true_negative_count, log_reg_model_02.false_positive_count, log_reg_model_02.false_negative_count, log_reg_model_02.true_positive_count = log_reg_model_02.confusion_matrix.ravel()


#%%

log_reg_model_02.df_X_test_predicted_class_labels = pandas.DataFrame(log_reg_model_02.X_test_predicted_class_labels)
log_reg_model_02.df_X_test_predicted_class_labels.columns = ['Late_Payments']

log_reg_model_02.df_X_test_predicted_probability_estimates = pandas.DataFrame(log_reg_model_02.X_test_predicted_probability_estimates)
log_reg_model_02.df_X_test_predicted_probability_estimates.columns = ['OnTime_Payments_Probability', 'Late_Payments_Probability']

log_reg_model_02.df_X_test_predicted_log_probability_estimates = pandas.DataFrame(log_reg_model_02.X_test_predicted_log_probability_estimates)
#df_X_test_predicted_log_of_probability_estimates.columns = ['']

log_reg_model_02.df_X_test_predicted_confidence_scores = pandas.DataFrame(log_reg_model_02.X_test_predicted_confidence_scores)
#df_X_test_predicted_confidence_scores.columns = ['']


#%%

print(log_reg_model_02.params)


#%%

print(log_reg_model_02.number_of_iterations)


#%%

print(log_reg_model_02.intercept)


#%%

log_reg_model_02.coefficients_dict = dict(zip(log_reg_model_02.independent_column_names, list(log_reg_model_02.coefficients[0])))
log_reg_model_02.coefficients_dict


#%%

print('Mean Accuracy on training data-set: {:.5f}'.format(log_reg_model_02.score_train))
print('Mean Accuracy on testing data-set: {:.5f}'.format(log_reg_model_02.score_test))


#%%

#print(log_reg_model_02.confusion_matrix)
print('True Positves:   {:>6,}'.format(log_reg_model_02.true_positive_count))
print('False Positves:  {:>6,}'.format(log_reg_model_02.false_positive_count))
print('False Negatives: {:>6,}'.format(log_reg_model_02.false_negative_count))
print('True Negatives:  {:>6,}'.format(log_reg_model_02.true_negative_count))


#%%

print('Classification Report:', '\n', log_reg_model_02.classification_report)


#%%

print('Area Under the ROC Curve (AUC): {:.5f}'.format(log_reg_model_02.logit_roc_auc))


#%%

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(log_reg_model_02.fpr, log_reg_model_02.tpr, label='Logistic Regression (AUC = %0.5f)' % log_reg_model_02.logit_roc_auc)
matplotlib.pyplot.plot([0, 1], [0, 1],'r--')
matplotlib.pyplot.xlim([0.0, 1.0])
matplotlib.pyplot.ylim([0.0, 1.05])
matplotlib.pyplot.xlabel('False Positive Rate')
matplotlib.pyplot.ylabel('True Positive Rate')
matplotlib.pyplot.title('Receiver Operating Characteristic')
matplotlib.pyplot.legend(loc="lower right")
#matplotlib.pyplot.savefig('Log_ROC')
matplotlib.pyplot.show()

##%%
#
##If I exponentiate it, I get exp(.0885629)=1.092603.
##This tells me that black college graduates are 1.09 times more likely...
#
#math.exp(log_reg_model_02.intercept)
##> 27.061477123450594
#
#math.exp(log_reg_model_02.coefficients[0][0])
##> 1.4418632857489724
## Men (CODE_GENDER=1) are 1.44 times more likely to have a late payment
#
#math.exp(log_reg_model_02.intercept) + math.exp(log_reg_model_02.coefficients[0][0])
##> 28.503340409199566
## Overall, men's late payment rate is 28.50

#%%

print(log_reg_model_01.coefficients_dict)
print(log_reg_model_02.coefficients_dict)

print('Area Under the ROC Curve (AUC): {:.5f}'.format(log_reg_model_01.logit_roc_auc))
print('Area Under the ROC Curve (AUC): {:.5f}'.format(log_reg_model_02.logit_roc_auc))
