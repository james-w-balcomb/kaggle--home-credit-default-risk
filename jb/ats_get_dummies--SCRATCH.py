#%%

import numpy as np
import pandas as pd

#from pandas.api.types import is_string_dtype
#from pandas.api.types import is_numeric_dtype
#from pandas.core.dtypes.common import is_list_like
from pandas.core.frame import DataFrame
from pandas.core.series import Series



#%%

data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
data_file_name = 'application_train.csv'

df = pd.read_table(
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
                'TARGET':'int64',
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


independent_column_names = [
'CODE_GENDER',
'DAYS_EMPLOYED',
'EMERGENCYSTATE_MODE',
'EXT_SOURCE_1',
'FLAG_CONT_MOBILE',
'NAME_EDUCATION_TYPE'
]

df_X = df.loc[:,independent_column_names]



#%%

#data = df_X

##def get_dummies(data,
##                prefix=None,
##                prefix_sep='_',
##                dummy_na=False,
##                columns=None,
##                sparse=False,
##                drop_first=False):
#
##def _get_dummies_1d(data,
##                    prefix,
##                    prefix_sep='_',
##                    dummy_na=False,
##                    sparse=False,
##                    drop_first=False):
#def _get_dummies_1d():
#    pass
#
#
#prefix=None
#prefix_sep='_'
#dummy_na=False
#columns=None
#sparse=False
#drop_first=False


#%%

#if isinstance(data, DataFrame):
#        # determine columns being encoded
#
#        if columns is None:
#            columns_to_encode = data.select_dtypes(include=['object', 'category']).columns
#        else:
#            columns_to_encode = columns
#else:
#    result = _get_dummies_1d(data,
#                             prefix,
#                             prefix_sep,
#                             dummy_na,
#                             sparse=sparse,
#                             drop_first=drop_first)

#%%
    
#pd.get_dummies(data)
#>         DAYS_EMPLOYED  EXT_SOURCE_1  CODE_GENDER_F  CODE_GENDER_M  \
#> 0              -637.0      0.083037              0              1
#>         CODE_GENDER_XNA  EMERGENCYSTATE_MODE_No  EMERGENCYSTATE_MODE_Yes  \
#> 0                     0                       1                        0
#>         FLAG_CONT_MOBILE_0  FLAG_CONT_MOBILE_1  
#> 0                        0                   1
#> ...
#> [307511 rows x 9 columns]


#%%

#pd.get_dummies(data['CODE_GENDER'])
#>         F  M  XNA
#> 0       0  1    0
#> ...
#>[307511 rows x 3 columns]

#%%

#pd.get_dummies(data['CODE_GENDER'], columns='CODE_GENDER')


#%%

#pd.get_dummies(df.loc[:,['CODE_GENDER', 'EMERGENCYSTATE_MODE']], columns=['CODE_GENDER'])

#pd.get_dummies(df.loc[:,['CODE_GENDER', 'EMERGENCYSTATE_MODE']], columns=['EMERGENCYSTATE_MODE'])


#%%

#pd.get_dummies(data['EXT_SOURCE_1'])
#> File "C:\Users\jbalcomb\Anaconda3\lib\site-packages\pandas\core\reshape\reshape.py", line 1287, in _get_dummies_1d
#> dummy_mat = np.eye(number_of_cols, dtype=np.uint8).take(codes, axis=0)
#> MemoryError


#%%

#def is_category_ordered(data, column_name):
#    try:
#        data[column_name].min()
#    except TypeError:
#        return False
#    else:
#        return True


#%%

#if is_category_ordered(df, 'NAME_EDUCATION_TYPE'):
#    print("is_category_ordered: True")
#else:
#    print("is_category_ordered: False")
#
#if is_category_ordered(df, 'EMERGENCYSTATE_MODE'):
#    print("is_category_ordered: True")
#else:
#    print("is_category_ordered: False")


#%%

# pandas.core.frame.DataFrame
# pandas.core.series.Series

def ats_get_dummies(data,
                    prefix=None,
                    prefix_sep='_',
                    dummy_na=False,
                    sparse=False,
                    drop_first=False,
                    included_column_names=None,
                    excluded_column_names=None,
                    dtype=None):
    """
    Generates columns with flags indicating missing values in the existing columns
    
    Parameters
    ----------
    data : array-like, Series, or DataFrame
    prefix : string, list of strings, or dict of strings, default None
    prefix_sep : string, default '_'
    dummy_na : bool, default False
    column_names : list-like, default None
    sparse : bool, default False
    drop_first : bool, default False
    dtype : , default None
    
    Returns
    -------
    ats_df_dummies : DataFrame
    """
    
    #ats_dummies_metadata = dict(column_name=None, reference_level=None)
    
    ats_df_dummies = pd.DataFrame()
    
    if isinstance(data, DataFrame):
        print("pandas.core.frame.DataFrame")
        
        if included_column_names is None:
            print("included_column_names is None.")
            if excluded_column_names is not None:
                print("excluded_column_names is not None.")
                columns_to_encode = list(set(data.select_dtypes(include=['object', 'category']).columns) - set(excluded_column_names))
            else:
                print("excluded_column_names is None.")
                columns_to_encode = list(set(data.select_dtypes(include=['object', 'category']).columns))
        else:
            print("included_column_names is not None.")
            if excluded_column_names is not None:
                print("excluded_column_names is not None.")
                columns_to_encode = list(set(included_column_names) - set(excluded_column_names))
            else:
                print("excluded_column_names is None.")
                columns_to_encode = list(set(included_column_names))
        
        for column_name in columns_to_encode:
            print("column_name: {}".format(column_name))
            print("data[column_name].dtype: {}".format(data[column_name].dtype))
            if(data[column_name].dtype == np.number):
                print("{} is Numerical. Excluding...".format(column_name))
                columns_to_encode.remove(column_name)
            elif data[column_name].dtype.name == 'category':
                if data[column_name].cat.ordered:
                    print("{} is ordered. Excluding...".format(column_name))
                    columns_to_encode.remove(column_name)
            #if data[column_name].dtype.name == 'object':
            #    pass
        #columns_to_encode = [column_name for column_name in columns_to_encode if not data[column_name].cat.ordered]
        
        for column_name in columns_to_encode:
            print("Getting dummies for column: {}".format(column_name))
            dummy = pd.get_dummies(data[column_name],
                                   prefix=column_name,
                                   prefix_sep='__',
                                   dummy_na=False,
                                   columns=None,
                                   sparse=False,
                                   drop_first=False)
        
            ats_df_dummies = pd.concat([ats_df_dummies, dummy], axis=1)
            
            # drop most frequent variable for reference level
            dropped_column_name = column_name + '__' + data.groupby([column_name]).size().idxmax()
            print(column_name + " dropping " + dropped_column_name)
            #print(data.groupby([column_name]).size())
            ats_df_dummies.drop(dropped_column_name, axis=1, inplace=True)
    
    elif isinstance(data, Series):
        #column_name = data.name
        print("pandas.core.series.Series")
        print("column_name: {}".format(data.name))
        print("data[column_name].dtype: {}".format(data.dtype))
        if(data.dtype == np.number):
            print("You can't get_dummies on a numerical variable, dummy.")
        else:
            dummy = pd.get_dummies(data,
                                   prefix=data.name,
                                   prefix_sep='__',
                                   dummy_na=False,
                                   columns=None,
                                   sparse=False,
                                   drop_first=False)
            ats_df_dummies = pd.concat([ats_df_dummies, dummy], axis=1)
            # drop most frequent variable for reference level
            #dropped_column_name = column_name + '__' + data.groupby([column_name]).size().idxmax()
            dropped_column_name = data.name + '__' + data.value_counts().index[0]
            print(data.name + " dropping " + dropped_column_name)
            #print(data.groupby([column_name]).size())
            print(data.value_counts())
            ats_df_dummies.drop(dropped_column_name, axis=1, inplace=True)
    else:
        print("Mistakes were made.")
    
    return ats_df_dummies



#%%

independent_column_names = [
'CODE_GENDER',
'DAYS_EMPLOYED',
'EMERGENCYSTATE_MODE',
'EXT_SOURCE_1',
'FLAG_CONT_MOBILE',
'NAME_EDUCATION_TYPE'
]

ats_get_dummies(df.loc[:,independent_column_names])


#%%

ats_get_dummies(df['EXT_SOURCE_1'])


#%%

#df_X['CODE_GENDER'] = df['CODE_GENDER'].astype('category')

ats_get_dummies(df['CODE_GENDER'])


#%%

#df_X['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype(
#        'category',
#        ordered=True,
#        categories=[
#                'Academic degree',
#                'Higher education',
#                'Incomplete higher',
#                'Secondary / secondary special',
#                'Lower secondary'
#                ]
#        )

ats_get_dummies(df['NAME_EDUCATION_TYPE'])


#%%

ats_get_dummies(dict())


#%%

df_dummies = ats_get_dummies(df, excluded_column_names=['SK_ID_CURR', 'TARGET'])


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

df_dummies = ats_get_dummies(df, excluded_column_names=['SK_ID_CURR', 'TARGET'])


#%%
        
#df_nonmodel_column_names


#%%

#columns = set(df_dtype_object_column_names) - set(df_nonmodel_column_names)
#columns


#%%

#df['combined'] = df['bar'].astype(str) + '_' + df['foo'] + '_' + df['new']

#df["combined"] = df["foo"].str.cat(df[["bar", "new"]].astype(str), sep="_")

