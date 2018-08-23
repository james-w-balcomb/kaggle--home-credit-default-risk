
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'NB01-Load.ipynb')


# In[ ]:


# Current Data-Frame
df_current = 'application_train'


# In[ ]:


# Master Data-Frame
df_master = 'application_train'


# In[ ]:


# Count of Rows
# Count of Columns
df_row_count, df_column_count = df.shape
df_row_count, df_column_count


# In[ ]:


# Total Values Count
df_values_count_total = df_row_count * df_column_count
df_values_count_total


# In[ ]:


# Column Names
df_column_names = sorted(df.columns.tolist())
df_column_names


# In[ ]:


# Column Data Types
df_column_dtypes = df.dtypes
df_column_dtypes


# In[ ]:


# Column Data Types Groups
#df_column_dtype_groups = df.columns.to_series().groupby(df.dtypes).groups
#df_column_dtype_groups

#> TypeError: data type not understood
df_column_dtype_groups = df.columns.groupby(df.dtypes)
df_column_dtype_groups


# In[ ]:


# Any Missing Values
df_missing_values_flag = df.isnull().values.any()
df_missing_values_flag


# In[ ]:


# Total Missing Values Count
df_missing_values_count_total = df.isnull().sum().sum()
df_missing_values_count_total


# In[ ]:


# Total Missing Values Percentage
df_missing_values_percentage_total = df_missing_values_count_total / df_values_count_total
df_missing_values_percentage_total


# In[ ]:


# Count of Columns with/without Missing Values
#df_missing_values_column_count


# In[ ]:


# Count of Rows with/without Missing Values
#df_missing_values_row_count


# In[ ]:


# Percentage of Columns with/without Missing Values
#df_missing_values_column_percentage


# In[ ]:


# Percentage of Rows with/without Missing Values
#df_missing_values_row_percentage


# In[ ]:


# Count of Rows Per Column
df_columns_row_count = {column_name:df_row_count for column_name in df_column_names}
df_columns_row_count


# In[ ]:


# Count of Unique Values Per Column
df_columns_number_of_unique_values = {column_name:None for column_name in df_column_names}
for column_name in df_columns_number_of_unique_values:
    df_columns_number_of_unique_values[column_name] = df[column_name].nunique()
df_columns_number_of_unique_values


# In[ ]:


# Percentage of Unique Values Per Column
df_columns_percentage_of_unique_values = {column_name:None for column_name in df_column_names}
for column_name in df_columns_percentage_of_unique_values:
    df_columns_percentage_of_unique_values[column_name] = df_columns_number_of_unique_values[column_name]/df_row_count
df_columns_percentage_of_unique_values


# In[ ]:


# Any Missing Values Per Column
df_columns_missing_values_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_flag:
    df_columns_missing_values_flag[column_name] = df[column_name].isnull().any()

for key, value in df_columns_missing_values_flag.items():
    if value:
        print(key)


# In[ ]:


# Count of Missing Values Count Per Column
df_columns_missing_values_count = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_count:
    df_columns_missing_values_count[column_name] = df[column_name].isnull().sum()
df_columns_missing_values_count


# In[ ]:


# Percentage of Missing Values Per Column
df_columns_missing_values_percentage = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_percentage:
    df_columns_missing_values_percentage[column_name] = df_columns_missing_values_count[column_name]/df_row_count
df_columns_missing_values_percentage


# In[ ]:


# Any Missing Values Per Row
#df_rows_missing_values_flag


# In[ ]:


# Count of Missing Values Per Row
#df_rows_missing_values_count


# In[ ]:


# Percentage of Missing Values Per Row
#df_rows_missing_values_count


# In[ ]:


#df['SK_ID_CURR'].str.isdigit().all()


# In[ ]:


#df['SK_ID_CURR'].str.isalpha().all()


# In[ ]:


df_dtype_boolean_column_names = sorted(list(df.select_dtypes(include=['bool']).columns))
df_dtype_boolean_column_names


# In[ ]:


df_dtype_float64_column_names = sorted(list(df.select_dtypes(include=['float64']).columns))
df_dtype_float64_column_names


# In[ ]:


df_dtype_int64_column_names = sorted(list(df.select_dtypes(include=['int64']).columns))
df_dtype_int64_column_names


# In[ ]:


df_dtype_object_column_names = sorted(list(df.select_dtypes(include=['O']).columns))
df_dtype_object_column_names


# In[ ]:


df_dtype_category_column_names = sorted(list(df.select_dtypes(include=['category']).columns))
df_dtype_category_column_names


# # Test if values are letters or numbers
# # Then, test if numbers are integers or decimals
# 
# ```python
# # pandas.Series.str.isnumeric
# # Series.str.isnumeric()
# # Check whether all characters in each string in the Series/Index are numeric. Equivalent to str.isnumeric().
# # Returns:	is : Series/array of boolean values
# 
# string.isdecimal()  
# string.isdigit()
# string.isnumeric()
# ```
# 
# ...\pandas\core\strings.py  
# ```python
# from pandas.core.dtypes.common import (
#     is_bool_dtype,
#     is_categorical_dtype,
#     is_object_dtype,
#     is_string_like,
#     is_list_like,
#     is_scalar,
#     is_integer,
#     is_re
# )
# isalnum
# isalpha
# isdecimal
# isdigit
# isnumeric
# islower
# isspace
# isupper
# istitle
# ```
# 
# ...\string.py
# ```python
# # Some strings for ctype-style character classification
# whitespace = ' \t\n\r\v\f'
# ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
# ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# ascii_letters = ascii_lowercase + ascii_uppercase
# digits = '0123456789'
# hexdigits = digits + 'abcdef' + 'ABCDEF'
# octdigits = '01234567'
# punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# printable = digits + ascii_letters + punctuation + whitespace
# ```
# 

# In[ ]:


try:
    i = float(str)
except (ValueError, TypeError):
    print('\nNot numeric')

print(string.isdecimal())

def isNumeric(s):
return s.isnumeric()

print(isNumeric("1234124"))

if string.isdigit():
    print("Your message includes numbers only.")
else:
    print("Your message does not include numbers.")

# Iterating the string and checking for numeric characters
# Incrementing the counter if a numeric character is found
# And adding the character to new string if not numeric
# Finally printing the count and the newstring
for a in string:
    if (a.isnumeric()) == True:
        count+= 1
    else:
        newstring1+= a
print(count)
print(newstring1)


# In[ ]:


# column_name = 'ORGANIZATION_TYPE'
# float(df[column_name])
# #> TypeError: cannot convert the series to <class 'float'>

column_name = 'ORGANIZATION_TYPE'
try:
    float(df[column_name])
except (TypeError):
    print('Non-Numeric: ' + column_name)


# In[ ]:


def is_numeric_scalar(scalar_value):
    # Iterating the string and checking for numeric characters
    # Incrementing the counter if a numeric character is found
    # And adding the character to new string if not numeric
    # NOTE: iteration over a string is actually iteration over the individual characters
    
    scalar_value_is_numeric_scalar = None
    
    for single_character in scalar_value:
        #print(scalar_value)
        #print(single_character)
        if (single_character.isnumeric()) == True:
            scalar_value_is_numeric_scalar = True
            #print(True)
        else:
            scalar_value_is_numeric_scalar = False
            #print(False)
            #continue
            break
        
        #if scalar_value_is_numeric_scalar == False:
        #    continue
    
    return scalar_value_is_numeric_scalar


# In[ ]:


# column_name = 'ORGANIZATION_TYPE'
# df[column_name].apply(is_numeric_each_character)
#> Business Entity Type 3
#> B
#> False
#> Business Entity Type 3
#> u
#> False
#> Business Entity Type 3
#> s
#> False
#> Business Entity Type 3
#> i
#> False
#> Business Entity Type 3
#> n
#> False
#> Business Entity Type 3
#> e
#> False
#> Business Entity Type 3
#> s
#> False
#> Business Entity Type 3
#> s
#> False
#> Business Entity Type 3
#>  
#> False
#> Business Entity Type 3
#> E
#> False
#> Business Entity Type 3
#> n
#> False
#> Business Entity Type 3
#> t
#> False
#> Business Entity Type 3
#> i
#> False
#> Business Entity Type 3
#> t
#> False
#> Business Entity Type 3
#> y
#> False
#> Business Entity Type 3
#>  
#> False
#> Business Entity Type 3
#> T
#> False
#> Business Entity Type 3
#> y
#> False
#> Business Entity Type 3
#> p
#> False
#> Business Entity Type 3
#> e
#> False
#> Business Entity Type 3
#>  
#> False
#> Business Entity Type 3
#> 3
#> True


# In[ ]:


# column_name = 'ORGANIZATION_TYPE'
# df[column_name].apply(is_numeric_scalar)
#> 0          True
#> 1         False
#> 2         False
#> 3          True
#> 4         False
#> 5         False
#> 6          True
#> 7         False
#> 8         False
#> 9         False
#> 10        False
#> 11        False
#> 12         True
#> 13        False
#> 14         True
#> 15         True
#> 16        False
#> 17        False
#> 18        False
#> 19        False
#> 20        False
#> 21         True
#> 22        False
#> 23        False
#> 24         True
#> 25         True
#> 26         True
#> 27         True
#> 28        False
#> 29         True
#>           ...  
#> 307481     True
#> 307482    False
#> 307483    False
#> 307484     True
#> 307485     True
#> 307486     True
#> 307487    False
#> 307488    False
#> 307489     True
#> 307490     True
#> 307491    False
#> 307492    False
#> 307493    False
#> 307494    False
#> 307495     True
#> 307496    False
#> 307497     True
#> 307498    False
#> 307499    False
#> 307500     True
#> 307501     True
#> 307502    False
#> 307503    False
#> 307504    False
#> 307505    False
#> 307506    False
#> 307507    False
#> 307508    False
#> 307509     True
#> 307510     True
#> Name: ORGANIZATION_TYPE, Length: 307511, dtype: bool


# In[ ]:


def is_numeric_series(pandas_series):
    # Iterating the string and checking for numeric characters
    # Incrementing the counter if a numeric character is found
    # And adding the character to new string if not numeric
    # NOTE: iteration over a string is actually iteration over the individual characters
    
    pandas_series_is_numeric_series = None
    
    #TODO(JamesBalcomb): decide on early-exit when dtype is bool, float, int, etc. (.:. TypeError: 'numpy.float64' object is not iterable)
    #if pandas_series.dtype == numpy.number:
    #if pandas.api.types.is_numeric_dtype(pandas_series):
    
    if pandas.api.types.is_numeric_dtype(pandas_series):
        pandas_series_is_numeric_series = True
    else:
        for index_number in pandas_series.index:
            pandas_series_value = pandas_series.loc[index_number]
            #print(pandas_series_value)
            if is_numeric_scalar(pandas_series_value):
                pandas_series_is_numeric_series = True
                #print(True)
            else:
                pandas_series_is_numeric_series = False
                #print(False)
                break
    
    return pandas_series_is_numeric_series


# In[ ]:


# column_name = 'ORGANIZATION_TYPE'
# pandas_series = df[column_name]

# type(pandas_series)
#> <class 'pandas.core.series.Series'>
# pandas_series.iloc[0]
#> 'Business Entity Type 3'
# pandas_series.iloc[1]
#> 'School'
# pandas_series.iloc[0].iat[0]
#> AttributeError: 'str' object has no attribute 'iat'

# pandas.Series.iloc
# Series.iloc
# Purely integer-location based indexing for selection by position.
# .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.

# len(pandas_series)
#> 307511

# pandas_series.index
#> RangeIndex(start=0, stop=307511, step=1)

# for index_id in pandas_series.index:
#     print(pandas_series.loc[index_id])
#> Business Entity Type 3
#> School
#> Government
#> Business Entity Type 3
#> Religion
#> ...
#> Services
#> XNA
#> School
#> Business Entity Type 1
#> Business Entity Type 3


# In[ ]:


# column_name = 'ORGANIZATION_TYPE'
# pandas_series = df[column_name]
# is_numeric_series(pandas_series)
#> False

# column_name = 'AMT_INCOME_TOTAL'
# pandas_series = df[column_name]
# is_numeric_series(pandas_series)
#> TypeError                                 Traceback (most recent call last)
#> <ipython-input-49-ff847204bea6> in <module>()
#> ---> 10 is_numeric_series(pandas_series)
#> <ipython-input-46-45f2861e88f6> in is_numeric_series(pandas_series)
#> ---> 12         if is_numeric_scalar(pandas_series_value):
#> <ipython-input-43-33965b580f01> in is_numeric_scalar(scalar_value)
#> ----> 9     for single_character in scalar_value:
#> TypeError: 'numpy.float64' object is not iterable

# column_name = 'ORGANIZATION_TYPE'
# pandas_series = df[column_name]
##import numpy
# pandas_series.dtype == numpy.number
#> False
##import pandas
####from pandas.api.types import is_string_dtype
####from pandas.api.types import is_numeric_dtype
# pandas.api.types.is_numeric_dtype(pandas_series)
#> False

# column_name = 'AMT_INCOME_TOTAL'
# pandas_series = df[column_name]
# pandas_series.dtype == numpy.number
#> True
# pandas.api.types.is_numeric_dtype(pandas_series)
#> True

# column_name = 'ORGANIZATION_TYPE'
# pandas_series = df[column_name]
# is_numeric_series(pandas_series)
#> False

# column_name = 'AMT_INCOME_TOTAL'
# pandas_series = df[column_name]
# is_numeric_series(pandas_series)
#> True


# # EDA - Meta-Data - Manual
# 

# In[ ]:


df_target_column_name = ['TARGET']
df_target_column_name


# In[ ]:


df_record_id_column_name = ['SK_ID_CURR']
df_record_id_column_name


# In[ ]:


df_record_key_column_name = ['SK_ID_CURR']
df_record_key_column_name


# In[ ]:


df_nondata_column_names = ['SK_ID_CURR']
df_nondata_column_names


# In[ ]:


df_useless_column_names = ['FLAG_MOBIL']
df_useless_column_names


# In[ ]:


df_nonmodel_column_names = []
df_nonmodel_column_names = sorted(list(set(df_target_column_name + df_record_id_column_name + df_record_key_column_name + df_nondata_column_names + df_useless_column_names)))
df_nonmodel_column_names


# # Lists of Column Names
# df_nondata_column_names  
# df_useless_column_names  
# df_nonmodel_column_names  
# df_int_column_names  
# df_float_column_names  
# df_datetime_column_names  
# df_date_column_names  
# df_object_column_names  
# df_boolean_column_names  
# df_categorical_column_names  
# df_numerical_column_names  

# In[ ]:


df_int_column_names = []
df_int_column_names


# In[ ]:


df_float_column_names = []
df_float_column_names


# In[ ]:


df_currency_column_names = []
df_currency_column_names


# In[ ]:


df_timestamp_column_names = []
df_timestamp_column_names


# In[ ]:


df_datetime_column_names = []
df_datetime_column_names


# In[ ]:


df_date_column_names = []
df_date_column_names


# In[ ]:


df_object_column_names = [
'CODE_GENDER',
'EMERGENCYSTATE_MODE',
'FONDKAPREMONT_MODE',
'HOUSETYPE_MODE',
'NAME_CONTRACT_TYPE',
'NAME_TYPE_SUITE',
'NAME_INCOME_TYPE',
'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE',
'OCCUPATION_TYPE',
'ORGANIZATION_TYPE',
'WALLSMATERIAL_MODE',
'WEEKDAY_APPR_PROCESS_START',
]
df_object_column_names


# In[ ]:


df_boolean_column_names = [
'FLAG_CONT_MOBILE',
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
'FLAG_DOCUMENT_21',
'FLAG_EMAIL',
'FLAG_EMP_PHONE',
'FLAG_MOBIL',
'FLAG_OWN_CAR',
'FLAG_OWN_REALTY',
'FLAG_PHONE',
'FLAG_WORK_PHONE',
'LIVE_CITY_NOT_WORK_CITY',
'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION'
]
df_boolean_column_names


# In[ ]:


df_categorical_column_names = [
'CODE_GENDER',
'EMERGENCYSTATE_MODE',
'FLAG_CONT_MOBILE',
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
'FLAG_DOCUMENT_21',
'FLAG_EMAIL',
'FLAG_EMP_PHONE',
'FLAG_MOBIL',
'FLAG_OWN_CAR',
'FLAG_OWN_REALTY',
'FLAG_PHONE',
'FLAG_WORK_PHONE',
'FONDKAPREMONT_MODE',
'HOUSETYPE_MODE',
'HOUR_APPR_PROCESS_START',
'LIVE_CITY_NOT_WORK_CITY',
'LIVE_REGION_NOT_WORK_REGION',
'NAME_CONTRACT_TYPE',
'NAME_EDUCATION_TYPE',
'NAME_FAMILY_STATUS',
'NAME_HOUSING_TYPE',
'NAME_INCOME_TYPE',
'NAME_TYPE_SUITE',
'OCCUPATION_TYPE',
'ORGANIZATION_TYPE',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION',
'WALLSMATERIAL_MODE',
'WEEKDAY_APPR_PROCESS_START'
]
df_categorical_column_names


# In[ ]:


df_numerical_column_names = [
'AMT_ANNUITY',
'AMT_CREDIT',
'AMT_GOODS_PRICE',
'AMT_INCOME_TOTAL',
'AMT_REQ_CREDIT_BUREAU_DAY',
'AMT_REQ_CREDIT_BUREAU_HOUR',
'AMT_REQ_CREDIT_BUREAU_MON',
'AMT_REQ_CREDIT_BUREAU_QRT',
'AMT_REQ_CREDIT_BUREAU_WEEK',
'AMT_REQ_CREDIT_BUREAU_YEAR',
'APARTMENTS_AVG',
'APARTMENTS_MEDI',
'APARTMENTS_MODE',
'BASEMENTAREA_AVG',
'BASEMENTAREA_MEDI',
'BASEMENTAREA_MODE',
'CNT_CHILDREN',
'CNT_FAM_MEMBERS',
'COMMONAREA_AVG',
'COMMONAREA_MEDI',
'COMMONAREA_MODE',
'DAYS_BIRTH',
'DAYS_EMPLOYED',
'DAYS_ID_PUBLISH',
'DAYS_LAST_PHONE_CHANGE',
'DAYS_REGISTRATION',
'DEF_30_CNT_SOCIAL_CIRCLE',
'DEF_60_CNT_SOCIAL_CIRCLE',
'ELEVATORS_AVG',
'ELEVATORS_MEDI',
'ELEVATORS_MODE',
'ENTRANCES_AVG',
'ENTRANCES_MEDI',
'ENTRANCES_MODE',
'EXT_SOURCE_1',
'EXT_SOURCE_2',
'EXT_SOURCE_3',
'FLOORSMAX_AVG',
'FLOORSMAX_MEDI',
'FLOORSMAX_MODE',
'FLOORSMIN_AVG',
'FLOORSMIN_MEDI',
'FLOORSMIN_MODE',
'HOUR_APPR_PROCESS_START',
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
'OBS_30_CNT_SOCIAL_CIRCLE',
'OBS_60_CNT_SOCIAL_CIRCLE',
'OWN_CAR_AGE',
'REGION_POPULATION_RELATIVE',
'REGION_RATING_CLIENT',
'REGION_RATING_CLIENT_W_CITY',
'TOTALAREA_MODE',
'YEARS_BEGINEXPLUATATION_AVG',
'YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BEGINEXPLUATATION_MODE',
'YEARS_BUILD_AVG',
'YEARS_BUILD_MEDI',
'YEARS_BUILD_MODE'
]
df_numerical_column_names


# 1) Always Positive / Never Negative  
# 2) Always Negative / Never Positive  
# 3) Always Between 1 and 0  
# 4) Has Zero  
# 5) Never Zero  
# 6) Has Mean of 0  
# 7) Has Standard Deviation of 1  
# 8) Is Mean Centered  
# 9) Is Scaled  
# 10) Is Z-Score (AKA Standardized, Normalized, Centered and Scaled)

# ```python
# # Test if column has dtype of category
# if df[column_name].dtype.name == 'category':
# # Or
# if isinstance(df.[column_name].dtype, pd.core.common.CategoricalDtype):
# # Or
# if pd.core.common.is_categorical_dtype(df.[column_name]):
# ```

# In[ ]:


# Always Positive / Never Negative
#all(df['TARGET'] >= 0)
df_columns_always_positive_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_always_positive_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_always_positive_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_always_positive_flag[column_name] = None
    else:
        df_columns_always_positive_flag[column_name] = all(df[column_name] > 0)
df_columns_always_positive_flag


# In[ ]:


# Always Negative / Never Positive
#all(df['TARGET'] <= 0)
df_columns_always_negative_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_always_negative_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_always_negative_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_always_negative_flag[column_name] = None
    else:
        df_columns_always_negative_flag[column_name] = all(df[column_name] < 0)
df_columns_always_negative_flag


# In[ ]:


# Always Between 1 and 0
#all(df[(df['TARGET'] >= 0) & (df['TARGET'] <= 1)])
# all(df['TARGET'].between(0, 1, inclusive=True))
# df['TARGET'].between(0, 1, inclusive=True).any()
df_columns_always_between_one_and_zero_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_always_between_one_and_zero_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_always_between_one_and_zero_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_always_between_one_and_zero_flag[column_name] = None
    else:
        df_columns_always_between_one_and_zero_flag[column_name] = all(df[(df[column_name] >= 0) & (df[column_name] <= 1)])
df_columns_always_between_one_and_zero_flag


# In[ ]:


# Has Zero
#any(df['TARGET'] == 0)
df_columns_has_zero_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_has_zero_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_has_zero_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_has_zero_flag[column_name] = None
    else:
        df_columns_has_zero_flag[column_name] = any(df[column_name] == 0)
df_columns_has_zero_flag


# In[ ]:


# Never Zero
#all(df['TARGET'] != 0)
df_columns_never_zero_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_never_zero_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_never_zero_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_never_zero_flag[column_name] = None
    else:
        df_columns_never_zero_flag[column_name] = all(df[column_name] != 0)
df_columns_never_zero_flag


# In[ ]:


# Has Mean of 0
#df['TARGET'].mean() == 0
df_columns_has_mean_of_zero_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_has_mean_of_zero_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_has_mean_of_zero_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_has_mean_of_zero_flag[column_name] = None
    else:
        df_columns_has_mean_of_zero_flag[column_name] = df[column_name].mean() == 0
df_columns_has_mean_of_zero_flag


# In[ ]:


# Has Standard Deviation of 1
#df['TARGET'].std() == 1
df_columns_has_standard_deviation_of_one_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_has_standard_deviation_of_one_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_has_standard_deviation_of_one_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_has_standard_deviation_of_one_flag[column_name] = None
    else:
        df_columns_has_standard_deviation_of_one_flag[column_name] = df[column_name].std() == 1
df_columns_has_standard_deviation_of_one_flag


# In[ ]:


# Is Mean Centered (i.e., the mean of all values has been subtracted from each value)
#df['TARGET'].mean() == 0
df_columns_is_mean_centered_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_is_mean_centered_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_is_mean_centered_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_is_mean_centered_flag[column_name] = None
    else:
        df_columns_is_mean_centered_flag[column_name] = df[column_name].mean() == 0
df_columns_is_mean_centered_flag


# In[ ]:


# Is Scaled
#df['TARGET'].std() == 1
df_columns_is_scaled_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_is_scaled_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_is_scaled_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_is_scaled_flag[column_name] = None
    else:
        df_columns_is_scaled_flag[column_name] = df[column_name].std() == 1
df_columns_is_scaled_flag


# In[ ]:


# Is Z-Score (AKA Standardized, Normalized, Centered and Scaled)
#((df['TARGET'].mean() == 0) & (df['TARGET'].std() == 1))
df_columns_is_z_score_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_is_z_score_flag:
    if df[column_name].dtype.name == 'object':
        df_columns_is_z_score_flag[column_name] = None
    elif df[column_name].dtype.name == 'category':
        df_columns_is_z_score_flag[column_name] = None
    else:
        df_columns_is_z_score_flag[column_name] = ((df[column_name].mean() == 0) & (df[column_name].std() == 1))
df_columns_is_z_score_flag


# # Skew & Kurtosis

# In[ ]:


df_columns_skew_value = {column_name:None for column_name in df_column_names}
for column_name in df_columns_skew_value:
    if df[column_name].dtype.name == 'object':
        # df_columns_skew_value[column_name] = None
        df_columns_skew_value[column_name] = 0
    elif df[column_name].dtype.name == 'category':
        #df_columns_skew_value[column_name] = None
        df_columns_skew_value[column_name] = 0
    else:
        df_columns_skew_value[column_name] = sp.stats.skew(df[column_name], nan_policy='omit')
df_columns_skew_value


# In[ ]:


for key in sorted(df_columns_skew_value, key=df_columns_skew_value.get):
    print("{key}: {value}".format(key=key, value=df_columns_skew_value[key]))
#> TypeError: '<' not supported between instances of 'NoneType' and 'float'


# In[ ]:


df_columns_kurtosis_value = {column_name:None for column_name in df_column_names}
for column_name in df_columns_kurtosis_value:
    if df[column_name].dtype.name == 'object':
        #df_columns_kurtosis_value[column_name] = None
        df_columns_kurtosis_value[column_name] = 0
    elif df[column_name].dtype.name == 'category':
        #df_columns_kurtosis_value[column_name] = None
        df_columns_kurtosis_value[column_name] = 0
    else:
        df_columns_kurtosis_value[column_name] = sp.stats.kurtosis(df[column_name], nan_policy='omit')
df_columns_kurtosis_value


# In[ ]:


for key in sorted(df_columns_kurtosis_value, key=df_columns_kurtosis_value.get):
    print("{key}: {value}".format(key=key, value=df_columns_kurtosis_value[key]))
#> TypeError: '<' not supported between instances of 'NoneType' and 'float'


# # Outliers

# In[ ]:


print(np.std(df['AMT_INCOME_TOTAL']))
print(np.std(df['AMT_INCOME_TOTAL']) * 3)


# In[ ]:


print(sp.stats.iqr(df['AMT_INCOME_TOTAL']))
print(sp.stats.iqr(df['AMT_INCOME_TOTAL']) * 1.5)


# In[ ]:


column_name = 'AMT_INCOME_TOTAL'
# Computing IQR
Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)
IQR = Q3 - Q1
IQR150 = IQR * 1.50
Q1IQR150 = Q1 - IQR150
Q3IQR150 = Q3 + IQR150
#df[column_name] < Q1 - IQR * 1.5
#df[column_name] > Q3 + IQR * 1.5
((df[column_name] < Q1IQR150) | (df[column_name] > Q3IQR150))


# In[ ]:


df_columns_outlier_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_outlier_flag:
    if df[column_name].dtype.name == 'object':
        #df_columns_outlier_flag[column_name] = None
        df_columns_outlier_flag[column_name] = 0
    elif df[column_name].dtype.name == 'category':
        #df_columns_outlier_flag[column_name] = None
        df_columns_outlier_flag[column_name] = 0
    else:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        IQR150 = IQR * 1.50
        Q1IQR150 = Q1 - IQR150
        Q3IQR150 = Q3 + IQR150
        df_columns_outlier_flag[column_name] = ((df[column_name] < Q1IQR150) | (df[column_name] > Q3IQR150))
df_columns_outlier_flag

