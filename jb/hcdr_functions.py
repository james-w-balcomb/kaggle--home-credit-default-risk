# hcdr_functions.py

import numpy
import os
import pandas
import random


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

    return df
