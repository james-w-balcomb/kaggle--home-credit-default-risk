# hcdr_functions.py

import numpy
import os
import pandas
import random

#import ats_functions
import hcdr_functions


#DATA_FILE_NAMES = list(
#'application_test.csv',
#'application_train.csv',
#'bureau.csv',
#'bureau_balance.csv',
#'credit_card_balance.csv',
#'installments_payments.csv',
#'POS_CASH_balance.csv',
#'previous_application.csv'
#)


## Set configuration based on environment variables
#if os.getenv('HCDR_DATA_FILE_PATH'):
#    print('Using Environment Variable for data_file_path')
#    data_file_path = os.getenv('HCDR_DATA_FILE_PATH')
#    data_file_path = os.path.join(data_file_path, '')
##TODO(JamesBalcomb): add code to fall back on .config file
##else:
##    'kaggle--home-credit-default-risk.config'
#else:
#    print('Using Hard-Coded Configuration for data_file_path')
#    data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
#    data_file_path = os.path.join(data_file_path, '')
#print('data_file_path: ', data_file_path)
#print()
#
#
## Set configuration based on environment variables
#if os.getenv('HCDR_WORKING_DIRECTORY'):
#    print('Using Environment Variable for working_directory')
#    working_directory = os.getenv('HCDR_WORKING_DIRECTORY')
#    working_directory = os.path.join(working_directory, '')
##TODO(JamesBalcomb): add code to fall back on .config file
##else:
##    'kaggle--home-credit-default-risk.config'
#else:
#    print('Using Hard-Coded Configuration for working_directory')
#    working_directory = 'C:/Development/kaggle--home-credit-default-risk/'
#    working_directory = os.path.join(working_directory, '')
#print('working_directory: ', working_directory)
#print()


def get_data_file_path():
    
    # Set configuration based on environment variables
    if os.getenv('HCDR_DATA_FILE_PATH'):
        print('Using Environment Variable for data_file_path')
        data_file_path = os.getenv('HCDR_DATA_FILE_PATH')
        data_file_path = os.path.join(data_file_path, '')
    #TODO(JamesBalcomb): add code to fall back on .cfg file
    #else:
    #    'kaggle--home-credit-default-risk.cfg'
    else:
        print('Using Hard-Coded Configuration for data_file_path')
        data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
        data_file_path = os.path.join(data_file_path, '')
    
    return data_file_path


def load_configuration():
    
    RANDOM_SEED = 1234567890
    random.seed(RANDOM_SEED)
    numpy.random.seed(RANDOM_SEED)

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


def load_data_file(data_file_path, data_file_name, dtypes_dictionary):
    # TODO(JamesBalcomb): build this load_data_file() function
    # TODO(JamesBalcomb): add a function that handles specifying multiple files
    pass


def load_data_file__application_train_csv():
    
    data_file_name = 'application_train.csv'
    
    data_file_path = get_data_file_path()
    
    print('Importing data file...')
    print('Data File Path: '  + data_file_path)
    print('Data File Name: '  + data_file_name)
    print()
    
    df = pandas.read_csv(
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
                    'HOUR_APPR_PROCESS_START':'object',
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

    return df

credit_bureau_enquiries_column_names   = list()
credit_scores_column_names             = list()
asset_ownership_flags_column_names     = list()
contact_information_flags_column_names = list()
document_flags_column_names            = list()
housing_characteristics_column_names   = list()

# Categorical Polychotomous Variable Feature Sets
credit_bureau_enquiries_column_names = [
'AMT_REQ_CREDIT_BUREAU_DAY',
'AMT_REQ_CREDIT_BUREAU_HOUR',
'AMT_REQ_CREDIT_BUREAU_MON',
'AMT_REQ_CREDIT_BUREAU_QRT',
'AMT_REQ_CREDIT_BUREAU_WEEK',
'AMT_REQ_CREDIT_BUREAU_YEAR'
]
credit_scores_column_names = [
'EXT_SOURCE_1',
'EXT_SOURCE_2',
'EXT_SOURCE_3'
]

# Categorical Dichotomous Variable Feature Sets
asset_ownership_flags_column_names = [
'FLAG_OWN_REALTY',
'FLAG_OWN_CAR'
]
contact_information_flags_column_names = [
'FLAG_PHONE',
'FLAG_WORK_PHONE',
'FLAG_EMP_PHONE',
'FLAG_MOBIL',
'FLAG_EMAIL',
'FLAG_CONT_MOBILE'
]
document_flags_column_names = [
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
housing_characteristics_column_names = [
'APARTMENTS_AVG',
'APARTMENTS_MEDI',
'APARTMENTS_MODE',
'BASEMENTAREA_AVG',
'BASEMENTAREA_MEDI',
'BASEMENTAREA_MODE',
'COMMONAREA_AVG',
'COMMONAREA_MEDI',
'COMMONAREA_MODE',
'ELEVATORS_AVG',
'ELEVATORS_MEDI',
'ELEVATORS_MODE',
'EMERGENCYSTATE_MODE',
'ENTRANCES_AVG',
'ENTRANCES_MEDI',
'ENTRANCES_MODE',
'FLOORSMAX_AVG',
'FLOORSMAX_MEDI',
'FLOORSMAX_MODE',
'FLOORSMIN_AVG',
'FLOORSMIN_MEDI',
'FLOORSMIN_MODE',
'FONDKAPREMONT_MODE',
'HOUSETYPE_MODE',
'LANDAREA_AVG',
'LANDAREA_MEDI',
'LANDAREA_MODE',
'LIVINGAPARTMENTS_AVG',
'LIVINGAPARTMENTS_MEDI',
'LIVINGAPARTMENTS_MODE',
'LIVINGAREA_AVG',
'LIVINGAREA_MEDI',
'LIVINGAREA_MODE',
'NONLIVINGAPARTMENTS_AVG',
'NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAPARTMENTS_MODE',
'NONLIVINGAREA_AVG',
'NONLIVINGAREA_MEDI',
'NONLIVINGAREA_MODE',
'TOTALAREA_MODE',
'WALLSMATERIAL_MODE',
'YEARS_BEGINEXPLUATATION_AVG',
'YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BEGINEXPLUATATION_MODE',
'YEARS_BUILD_AVG',
'YEARS_BUILD_MEDI',
'YEARS_BUILD_MODE'
]


def load_prepared_data_set__application_train_csv():
    
    df = hcdr_functions.load_data_file__application_train_csv()
    
    # make preliminary manual fixes
    print("Manually replace the 4 out of 300,000 values missing for CODE_GENDER...")
    
    df.loc[df['SK_ID_CURR'] == '141289', 'CODE_GENDER'] = 'F'
    df.loc[df['SK_ID_CURR'] == '319880', 'CODE_GENDER'] = 'F'
    df.loc[df['SK_ID_CURR'] == '196708', 'CODE_GENDER'] = 'F'
    df.loc[df['SK_ID_CURR'] == '144669', 'CODE_GENDER'] = 'M'
    
    df.loc[df['HOUR_APPR_PROCESS_START'] == '141289', 'HOUR_APPR_PROCESS_START'] = 'F'
    
        
    # make Missing Values Place-Holder flags
    print("Adding columns with flags for Missing Values Place-Holders...")
    
    column_name_prefix = 'missing_value_placeholder_flag__'
    
    column_name = 'DAYS_EMPLOYED'
    missing_value_placeholder_value = 365243
    df[column_name_prefix + column_name] = False
    df.loc[df[column_name] == missing_value_placeholder_value, column_name_prefix + column_name] = True
    df[column_name_prefix + column_name] = df[column_name_prefix + column_name].astype(int).astype(str)
    
    column_name = 'ORGANIZATION_TYPE'
    missing_value_placeholder_value = 'XNA'
    df[column_name_prefix + column_name] = False
    df.loc[df[column_name] == missing_value_placeholder_value, column_name_prefix + column_name] = True
    df[column_name_prefix + column_name] = df[column_name_prefix + column_name].astype(int).astype(str)
    
    # replace Missing Values Place-Holders
    print("Replacing Missing Value Place-Holders with numpy.nan...")
    
    column_name = 'DAYS_EMPLOYED'
    missing_value_placeholder_value = 365243
    df[column_name] = df[column_name].replace(missing_value_placeholder_value, numpy.nan)
    
    column_name = 'ORGANIZATION_TYPE'
    missing_value_placeholder_value = 'XNA'
    df[column_name] = df[column_name].replace(missing_value_placeholder_value, numpy.nan)
    
    # make Missing Values column count
    print("Adding column with count of columns with Missing Values...")
    
    df['missing_values_column_count'] = df.isnull().sum(axis=1)
    
    # make Missing Values flags
    print("Adding columns with flags for Missing Values...")
    
    column_name_prefix = 'missing_value_flag__'
    
    for column_name in sorted(df.columns.tolist()):
        if df[column_name].isnull().any():
            df[column_name_prefix + column_name] = df[column_name].isnull().astype(int).astype(str)
    
    # make feature-set Missing Values counts
    print("Adding columns with counts of columns with Missing Values per feature set...")
    
    # NA # asset_ownership_flags_column_names
    # NA # contact_information_flags_column_names
    df['missing_values_count__credit_bureau_enquiries'] = df[credit_bureau_enquiries_column_names].isnull().sum(axis=1)
    df['missing_values_count__credit_scores'] = df[credit_scores_column_names].isnull().sum(axis=1)
    # NA # document_flags_column_names
    df['missing_values_count__housing_characteristics'] = df[housing_characteristics_column_names].isnull().sum(axis=1)
    
    # Convert categorical dichotomous variables to {0,1}
    print("Converting categorical dichotomous variable to {0,1}")
    
    df['FLAG_OWN_CAR'] = pandas.Series(numpy.where(df['FLAG_OWN_CAR'].values == 'Y', 1, 0), df.index)
    df['FLAG_OWN_REALTY'] = pandas.Series(numpy.where(df['FLAG_OWN_REALTY'].values == 'Y', 1, 0), df.index)
    
    # make feature-set TRUE counts
    print("Adding columns with counts of columns with TRUE for Categorical Dichotomous variable feature sets...")
    
    df['asset_ownership_flags_count'] = df[asset_ownership_flags_column_names].astype('int').sum(axis=1)
    df['contact_information_flags_count'] = df[contact_information_flags_column_names].astype('int').sum(axis=1)
    # NA # credit_bureau_enquiries_column_names
    # NA # credit_scores_column_names
    df['document_flags_count'] = df[document_flags_column_names].astype('int').sum(axis=1)
    # NA # housing_characteristics_column_names
        
    # make engineered features
    print("Adding engineered features of means, differences, and ratios...")
    
    df['EXT_SOURCE__mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True)      
    df['difference__DAYS_BIRTH__OWN_CAR_AGE'] = df['DAYS_BIRTH'] - df['OWN_CAR_AGE']
    df['difference__DAYS_BIRTH__DAYS_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
    df['ratio__AMT_CREDIT__AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['ratio__AMT_INCOME_TOTAL__AMT_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['ratio__AMT_CREDIT__AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['ratio__DAYS_ID_PUBLISH__DAYS_BIRTH'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['ratio__DAYS_REGISTRATION__DAYS_BIRTH'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    
    # Replace categorical missing values with "MISSING"
    print("Replace categorical variable's missing values with \"MISSING\"...")
    
    df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].fillna('MISSING')
    df['FONDKAPREMONT_MODE'] = df['FONDKAPREMONT_MODE'].fillna('MISSING')
    df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].fillna('MISSING')
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('MISSING')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('MISSING')
    df['ORGANIZATION_TYPE'] = df['EMERGENCYSTATE_MODE'].fillna('MISSING')
    df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].fillna('MISSING')
    
    # Replace numerical missing values with the mean
    print("Replacing numerical missing values with the mean...")
    
    for column_name in sorted(df.select_dtypes(include=[numpy.number]).columns.tolist()):
        if df[column_name].isnull().any():
            df[column_name] = df[column_name].fillna(df[column_name].mean())
    
#    # Encode Categorical Nominal as so-called "dummy variables"
#    df_dummies = ats_functions.ats_get_dummies(df,
#                                           prefix=None,
#                                           prefix_sep='_',
#                                           dummy_na=False,
#                                           sparse=False,
#                                           included_column_names=None,
#                                           excluded_column_names=['SK_ID_CURR'])
#    pandas.concat([df, df_dummies], axis=1)
    
    # Finalize dtypes
    print("Finalizing the dtypes for each column...")
    
    df['CODE_GENDER'] = df['CODE_GENDER'].astype('category')
    df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype('category')
    #df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].astype(pandas.api.types.CategoricalDtype(categories=['No', 'MISSING', 'Yes',], ordered=True))
    df['FLAG_CONT_MOBILE'] = df['FLAG_CONT_MOBILE'].astype('uint8')
    df['FLAG_DOCUMENT_2'] = df['FLAG_DOCUMENT_2'].astype('uint8')
    df['FLAG_DOCUMENT_3'] = df['FLAG_DOCUMENT_3'].astype('uint8')
    df['FLAG_DOCUMENT_4'] = df['FLAG_DOCUMENT_4'].astype('uint8')
    df['FLAG_DOCUMENT_5'] = df['FLAG_DOCUMENT_5'].astype('uint8')
    df['FLAG_DOCUMENT_6'] = df['FLAG_DOCUMENT_6'].astype('uint8')
    df['FLAG_DOCUMENT_7'] = df['FLAG_DOCUMENT_7'].astype('uint8')
    df['FLAG_DOCUMENT_8'] = df['FLAG_DOCUMENT_8'].astype('uint8')
    df['FLAG_DOCUMENT_9'] = df['FLAG_DOCUMENT_9'].astype('uint8')
    df['FLAG_DOCUMENT_10'] = df['FLAG_DOCUMENT_10'].astype('uint8')
    df['FLAG_DOCUMENT_11'] = df['FLAG_DOCUMENT_11'].astype('uint8')
    df['FLAG_DOCUMENT_12'] = df['FLAG_DOCUMENT_12'].astype('uint8')
    df['FLAG_DOCUMENT_13'] = df['FLAG_DOCUMENT_13'].astype('uint8')
    df['FLAG_DOCUMENT_14'] = df['FLAG_DOCUMENT_14'].astype('uint8')
    df['FLAG_DOCUMENT_15'] = df['FLAG_DOCUMENT_15'].astype('uint8')
    df['FLAG_DOCUMENT_16'] = df['FLAG_DOCUMENT_16'].astype('uint8')
    df['FLAG_DOCUMENT_17'] = df['FLAG_DOCUMENT_17'].astype('uint8')
    df['FLAG_DOCUMENT_18'] = df['FLAG_DOCUMENT_18'].astype('uint8')
    df['FLAG_DOCUMENT_19'] = df['FLAG_DOCUMENT_19'].astype('uint8')
    df['FLAG_DOCUMENT_20'] = df['FLAG_DOCUMENT_20'].astype('uint8')
    df['FLAG_DOCUMENT_21'] = df['FLAG_DOCUMENT_21'].astype('uint8')
    df['FLAG_EMAIL'] = df['FLAG_EMAIL'].astype('uint8')
    df['FLAG_EMP_PHONE'] = df['FLAG_EMP_PHONE'].astype('uint8')
    df['FLAG_MOBIL'] = df['FLAG_MOBIL'].astype('uint8')
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].astype('uint8')
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].astype('uint8')
    df['FLAG_PHONE'] = df['FLAG_PHONE'].astype('uint8')
    df['FLAG_WORK_PHONE'] = df['FLAG_WORK_PHONE'].astype('uint8')
    df['FONDKAPREMONT_MODE'] = df['FONDKAPREMONT_MODE'].astype('category')
    df['HOUR_APPR_PROCESS_START'] = df['HOUR_APPR_PROCESS_START'].astype('category')
    df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].astype('category')
    df['LIVE_CITY_NOT_WORK_CITY'] = df['LIVE_CITY_NOT_WORK_CITY'].astype('int')
    df['LIVE_REGION_NOT_WORK_REGION'] = df['LIVE_REGION_NOT_WORK_REGION'].astype('int')
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].astype('category')
    #df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype('category')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype(pandas.api.types.CategoricalDtype(categories=['Academic degree', 'Higher education', 'Incomplete higher', 'Secondary / secondary special', 'Lower secondary'], ordered=True))
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].astype('category')
    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('category')
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].astype('category')
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].astype('category')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].astype('category')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].astype('category')
    df['REG_CITY_NOT_LIVE_CITY'] = df['REG_CITY_NOT_LIVE_CITY'].astype('uint8')
    df['REG_CITY_NOT_WORK_CITY'] = df['REG_CITY_NOT_WORK_CITY'].astype('uint8')
    df['REG_REGION_NOT_LIVE_REGION'] = df['REG_REGION_NOT_LIVE_REGION'].astype('uint8')
    df['REG_REGION_NOT_WORK_REGION'] = df['REG_REGION_NOT_WORK_REGION'].astype('uint8')
    #df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].astype('category')
    df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].astype(pandas.api.types.CategoricalDtype(categories=['3', '2', '1'], ordered=True))
    #df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].astype('category')
    df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].astype(pandas.api.types.CategoricalDtype(categories=['3', '2', '1'], ordered=True))
    #df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('category')
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('object')
    #df['TARGET'] = df['TARGET'].astype('category')
    df['TARGET'] = df['TARGET'].astype(int)
    df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].astype('category')
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].astype('category')
    
    df['missing_value_placeholder_flag__DAYS_EMPLOYED'] = df['missing_value_placeholder_flag__DAYS_EMPLOYED'].astype('uint8')
    df['missing_value_placeholder_flag__ORGANIZATION_TYPE'] = df['missing_value_placeholder_flag__ORGANIZATION_TYPE'].astype('uint8')
    
    df['missing_value_flag__AMT_ANNUITY'] = df['missing_value_flag__AMT_ANNUITY'].astype('uint8')
    df['missing_value_flag__AMT_GOODS_PRICE'] = df['missing_value_flag__AMT_GOODS_PRICE'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_DAY'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_DAY'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_HOUR'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_HOUR'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_MON'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_MON'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_QRT'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_QRT'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_WEEK'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_WEEK'].astype('uint8')
    df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_YEAR'] = df['missing_value_flag__AMT_REQ_CREDIT_BUREAU_YEAR'].astype('uint8')
    df['missing_value_flag__APARTMENTS_AVG'] = df['missing_value_flag__APARTMENTS_AVG'].astype('uint8')
    df['missing_value_flag__APARTMENTS_MEDI'] = df['missing_value_flag__APARTMENTS_MEDI'].astype('uint8')
    df['missing_value_flag__APARTMENTS_MODE'] = df['missing_value_flag__APARTMENTS_MODE'].astype('uint8')
    df['missing_value_flag__BASEMENTAREA_AVG'] = df['missing_value_flag__BASEMENTAREA_AVG'].astype('uint8')
    df['missing_value_flag__BASEMENTAREA_MEDI'] = df['missing_value_flag__BASEMENTAREA_MEDI'].astype('uint8')
    df['missing_value_flag__BASEMENTAREA_MODE'] = df['missing_value_flag__BASEMENTAREA_MODE'].astype('uint8')
    df['missing_value_flag__CNT_FAM_MEMBERS'] = df['missing_value_flag__CNT_FAM_MEMBERS'].astype('uint8')
    df['missing_value_flag__COMMONAREA_AVG'] = df['missing_value_flag__COMMONAREA_AVG'].astype('uint8')
    df['missing_value_flag__COMMONAREA_MEDI'] = df['missing_value_flag__COMMONAREA_MEDI'].astype('uint8')
    df['missing_value_flag__COMMONAREA_MODE'] = df['missing_value_flag__COMMONAREA_MODE'].astype('uint8')
    df['missing_value_flag__DAYS_EMPLOYED'] = df['missing_value_flag__DAYS_EMPLOYED'].astype('uint8')
    df['missing_value_flag__DAYS_LAST_PHONE_CHANGE'] = df['missing_value_flag__DAYS_LAST_PHONE_CHANGE'].astype('uint8')
    df['missing_value_flag__DEF_30_CNT_SOCIAL_CIRCLE'] = df['missing_value_flag__DEF_30_CNT_SOCIAL_CIRCLE'].astype('uint8')
    df['missing_value_flag__DEF_60_CNT_SOCIAL_CIRCLE'] = df['missing_value_flag__DEF_60_CNT_SOCIAL_CIRCLE'].astype('uint8')
    df['missing_value_flag__ELEVATORS_AVG'] = df['missing_value_flag__ELEVATORS_AVG'].astype('uint8')
    df['missing_value_flag__ELEVATORS_MEDI'] = df['missing_value_flag__ELEVATORS_MEDI'].astype('uint8')
    df['missing_value_flag__ELEVATORS_MODE'] = df['missing_value_flag__ELEVATORS_MODE'].astype('uint8')
    df['missing_value_flag__EMERGENCYSTATE_MODE'] = df['missing_value_flag__EMERGENCYSTATE_MODE'].astype('uint8')
    df['missing_value_flag__ENTRANCES_AVG'] = df['missing_value_flag__ENTRANCES_AVG'].astype('uint8')
    df['missing_value_flag__ENTRANCES_MEDI'] = df['missing_value_flag__ENTRANCES_MEDI'].astype('uint8')
    df['missing_value_flag__ENTRANCES_MODE'] = df['missing_value_flag__ENTRANCES_MODE'].astype('uint8')
    df['missing_value_flag__EXT_SOURCE_1'] = df['missing_value_flag__EXT_SOURCE_1'].astype('uint8')
    df['missing_value_flag__EXT_SOURCE_2'] = df['missing_value_flag__EXT_SOURCE_2'].astype('uint8')
    df['missing_value_flag__EXT_SOURCE_3'] = df['missing_value_flag__EXT_SOURCE_3'].astype('uint8')
    df['missing_value_flag__FLOORSMAX_AVG'] = df['missing_value_flag__FLOORSMAX_AVG'].astype('uint8')
    df['missing_value_flag__FLOORSMAX_MEDI'] = df['missing_value_flag__FLOORSMAX_MEDI'].astype('uint8')
    df['missing_value_flag__FLOORSMAX_MODE'] = df['missing_value_flag__FLOORSMAX_MODE'].astype('uint8')
    df['missing_value_flag__FLOORSMIN_AVG'] = df['missing_value_flag__FLOORSMIN_AVG'].astype('uint8')
    df['missing_value_flag__FLOORSMIN_MEDI'] = df['missing_value_flag__FLOORSMIN_MEDI'].astype('uint8')
    df['missing_value_flag__FLOORSMIN_MODE'] = df['missing_value_flag__FLOORSMIN_MODE'].astype('uint8')
    df['missing_value_flag__FONDKAPREMONT_MODE'] = df['missing_value_flag__FONDKAPREMONT_MODE'].astype('uint8')
    df['missing_value_flag__HOUSETYPE_MODE'] = df['missing_value_flag__HOUSETYPE_MODE'].astype('uint8')
    df['missing_value_flag__LANDAREA_AVG'] = df['missing_value_flag__LANDAREA_AVG'].astype('uint8')
    df['missing_value_flag__LANDAREA_MEDI'] = df['missing_value_flag__LANDAREA_MEDI'].astype('uint8')
    df['missing_value_flag__LANDAREA_MODE'] = df['missing_value_flag__LANDAREA_MODE'].astype('uint8')
    df['missing_value_flag__LIVINGAPARTMENTS_AVG'] = df['missing_value_flag__LIVINGAPARTMENTS_AVG'].astype('uint8')
    df['missing_value_flag__LIVINGAPARTMENTS_MEDI'] = df['missing_value_flag__LIVINGAPARTMENTS_MEDI'].astype('uint8')
    df['missing_value_flag__LIVINGAPARTMENTS_MODE'] = df['missing_value_flag__LIVINGAPARTMENTS_MODE'].astype('uint8')
    df['missing_value_flag__LIVINGAREA_AVG'] = df['missing_value_flag__LIVINGAREA_AVG'].astype('uint8')
    df['missing_value_flag__LIVINGAREA_MEDI'] = df['missing_value_flag__LIVINGAREA_MEDI'].astype('uint8')
    df['missing_value_flag__LIVINGAREA_MODE'] = df['missing_value_flag__LIVINGAREA_MODE'].astype('uint8')
    df['missing_value_flag__NAME_TYPE_SUITE'] = df['missing_value_flag__NAME_TYPE_SUITE'].astype('uint8')
    df['missing_value_flag__NONLIVINGAPARTMENTS_AVG'] = df['missing_value_flag__NONLIVINGAPARTMENTS_AVG'].astype('uint8')
    df['missing_value_flag__NONLIVINGAPARTMENTS_MEDI'] = df['missing_value_flag__NONLIVINGAPARTMENTS_MEDI'].astype('uint8')
    df['missing_value_flag__NONLIVINGAPARTMENTS_MODE'] = df['missing_value_flag__NONLIVINGAPARTMENTS_MODE'].astype('uint8')
    df['missing_value_flag__NONLIVINGAREA_AVG'] = df['missing_value_flag__NONLIVINGAREA_AVG'].astype('uint8')
    df['missing_value_flag__NONLIVINGAREA_MEDI'] = df['missing_value_flag__NONLIVINGAREA_MEDI'].astype('uint8')
    df['missing_value_flag__NONLIVINGAREA_MODE'] = df['missing_value_flag__NONLIVINGAREA_MODE'].astype('uint8')
    df['missing_value_flag__OBS_30_CNT_SOCIAL_CIRCLE'] = df['missing_value_flag__OBS_30_CNT_SOCIAL_CIRCLE'].astype('uint8')
    df['missing_value_flag__OBS_60_CNT_SOCIAL_CIRCLE'] = df['missing_value_flag__OBS_60_CNT_SOCIAL_CIRCLE'].astype('uint8')
    df['missing_value_flag__OCCUPATION_TYPE'] = df['missing_value_flag__OCCUPATION_TYPE'].astype('uint8')
    df['missing_value_flag__ORGANIZATION_TYPE'] = df['missing_value_flag__ORGANIZATION_TYPE'].astype('uint8')
    df['missing_value_flag__OWN_CAR_AGE'] = df['missing_value_flag__OWN_CAR_AGE'].astype('uint8')
    df['missing_value_flag__TOTALAREA_MODE'] = df['missing_value_flag__TOTALAREA_MODE'].astype('uint8')
    df['missing_value_flag__WALLSMATERIAL_MODE'] = df['missing_value_flag__WALLSMATERIAL_MODE'].astype('uint8')
    df['missing_value_flag__YEARS_BEGINEXPLUATATION_AVG'] = df['missing_value_flag__YEARS_BEGINEXPLUATATION_AVG'].astype('uint8')
    df['missing_value_flag__YEARS_BEGINEXPLUATATION_MEDI'] = df['missing_value_flag__YEARS_BEGINEXPLUATATION_MEDI'].astype('uint8')
    df['missing_value_flag__YEARS_BEGINEXPLUATATION_MODE'] = df['missing_value_flag__YEARS_BEGINEXPLUATATION_MODE'].astype('uint8')
    df['missing_value_flag__YEARS_BUILD_AVG'] = df['missing_value_flag__YEARS_BUILD_AVG'].astype('uint8')
    df['missing_value_flag__YEARS_BUILD_MEDI'] = df['missing_value_flag__YEARS_BUILD_MEDI'].astype('uint8')
    df['missing_value_flag__YEARS_BUILD_MODE'] = df['missing_value_flag__YEARS_BUILD_MODE'].astype('uint8')
    
    df['missing_values_column_count'] = df['missing_values_column_count'].astype('int')
    df['missing_values_count__credit_bureau_enquiries'] = df['missing_values_count__credit_bureau_enquiries'].astype('int')
    df['missing_values_count__credit_scores'] = df['missing_values_count__credit_scores'].astype('int')
    df['missing_values_count__housing_characteristics'] = df['missing_values_count__housing_characteristics'].astype('int')
    
    df['asset_ownership_flags_count'] = df['asset_ownership_flags_count'].astype('int')
    df['contact_information_flags_count'] = df['contact_information_flags_count'].astype('int')
    df['document_flags_count'] = df['document_flags_count'].astype('int')
    
    #df['EXT_SOURCE__mean']
    #df['difference__DAYS_BIRTH__OWN_CAR_AGE']
    #df['difference__DAYS_BIRTH__DAYS_EMPLOYED']
    #df['ratio__AMT_CREDIT__AMT_ANNUITY']
    #df['ratio__AMT_INCOME_TOTAL__AMT_CREDIT']
    #df['ratio__AMT_CREDIT__AMT_GOODS_PRICE']
    #df['ratio__DAYS_EMPLOYED__DAYS_BIRTH']
    #df['ratio__DAYS_ID_PUBLISH__DAYS_BIRTH']
    #df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH']
    #df['ratio__DAYS_REGISTRATION__DAYS_BIRTH']
    
    return df
