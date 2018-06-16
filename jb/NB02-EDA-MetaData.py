
# coding: utf-8

# In[1]:


get_ipython().magic('run NB01-Load.ipynb')


# In[2]:


# Current Data-Frame
df_current = 'application_train'


# In[3]:


# Master Data-Frame
df_master = 'application_train'


# In[4]:


# Count of Rows
# Count of Columns
df_row_count, df_column_count = df.shape
print('\n', 'df_row_count: ', df_row_count, '\n', 'df_column_count: ', df_column_count, '\n')


# In[5]:


# Total Values Count
df_values_count_total = df_row_count * df_column_count
print('\n', 'df_values_count_total: ', df_values_count_total, '\n')


# In[6]:


# Column Names
df_column_names = df.columns.tolist()
print('\n', 'df_column_names: ', '\n', df_column_names, '\n')


# In[7]:


# Column Data Types
df_column_dtypes = df.dtypes
print('\n', 'df_column_dtypes: ', '\n', df_column_dtypes, '\n')


# In[8]:


# Column Data Types Groups
df_column_dtype_groups = df.columns.to_series().groupby(df.dtypes).groups
print('\n', 'df_column_dtype_groups: ', '\n', df_column_dtype_groups, '\n')


# In[9]:


# Any Missing Values
df_missing_values_flag = df.isnull().values.any()
print('\n', 'df_missing_values_flag: ', df_missing_values_flag, '\n')


# In[10]:


# Total Missing Values Count
df_missing_values_count_total = df.isnull().sum().sum()
print('\n', 'df_missing_values_count_total: ', df_missing_values_count_total, '\n')


# In[11]:


# Total Missing Values Percentage
df_missing_values_percentage_total = df_missing_values_count_total / df_values_count_total
print('\n', 'df_missing_values_percentage_total: ', df_missing_values_percentage_total, '\n')


# In[12]:


# Count of Columns with/without Missing Values
#df_missing_values_column_count


# In[13]:


# Count of Rows with/without Missing Values
#df_missing_values_row_count


# In[14]:


# Percentage of Columns with/without Missing Values
#df_missing_values_column_percentage


# In[15]:


# Percentage of Rows with/without Missing Values
#df_missing_values_row_percentage


# In[16]:


# Count of Rows Per Column
df_columns_row_count = {column_name:df_row_count for column_name in df_column_names}
print('\n', 'df_columns_row_count: ', '\n', df_columns_row_count, '\n')


# In[17]:


# Count of Unique Values Per Column
df_columns_number_of_unique_values = {column_name:None for column_name in df_column_names}
for column_name in df_columns_number_of_unique_values:
    df_columns_number_of_unique_values[column_name] = df[column_name].nunique()
print('\n', 'df_columns_number_of_unique_values: ', '\n', df_columns_number_of_unique_values, '\n')


# In[18]:


# Percentage of Unique Values Per Column
df_columns_percentage_of_unique_values = {column_name:None for column_name in df_column_names}
for column_name in df_columns_percentage_of_unique_values:
    df_columns_percentage_of_unique_values[column_name] = df_columns_number_of_unique_values[column_name]/df_row_count
print('\n', 'df_columns_percentage_of_unique_values: ', '\n', df_columns_percentage_of_unique_values, '\n')


# In[19]:


# Any Missing Values Per Column
df_columns_missing_values_flag = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_flag:
    df_columns_missing_values_flag[column_name] = df[column_name].isnull().any()
print('\n', 'df_columns_missing_values_flag: ', '\n', df_columns_missing_values_flag, '\n')


# In[20]:


# Count of Missing Values Count Per Column
df_columns_missing_values_count = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_count:
    df_columns_missing_values_count[column_name] = df[column_name].isnull().sum()
print('\n', 'df_columns_missing_values_count: ', '\n', df_columns_missing_values_count, '\n')


# In[21]:


# Percentage of Missing Values Per Column
df_columns_missing_values_percentage = {column_name:None for column_name in df_column_names}
for column_name in df_columns_missing_values_percentage:
    df_columns_missing_values_percentage[column_name] = df_columns_missing_values_count[column_name]/df_row_count
print('\n', 'df_columns_missing_values_percentage: ', '\n', df_columns_missing_values_percentage, '\n')


# In[22]:


# Any Missing Values Per Row
#df_rows_missing_values_flag


# In[23]:


# Count of Missing Values Per Row
#df_rows_missing_values_count


# In[24]:


# Percentage of Missing Values Per Row
#df_rows_missing_values_count


# In[25]:


#df['SK_ID_CURR'].str.isdigit().all()


# In[26]:


#df['SK_ID_CURR'].str.isalpha().all()


# In[27]:


df_dtype_boolean_column_names = list(df.select_dtypes(include=['bool']).columns)
df_dtype_boolean_column_names


# In[28]:


df_dtype_float64_column_names = list(df.select_dtypes(include=['float64']).columns)
df_dtype_float64_column_names


# In[29]:


df_dtype_int64_column_names = list(df.select_dtypes(include=['int64']).columns)
df_dtype_int64_column_names


# In[30]:


df_dtype_object_column_names = list(df.select_dtypes(include=['O']).columns)
df_dtype_object_column_names


# # EDA - Meta-Data - Manual
# 

# In[31]:


df_target_column_name = 'TARGET'
print('\n', 'df_target_column_name: ', df_target_column_name, '\n')


# In[32]:


df_record_id_column_name = 'SK_ID_CURR'
print('\n', 'df_record_id_column_name: ', df_record_id_column_name, '\n')


# In[33]:


df_record_key_column_name = 'SK_ID_CURR'
print('\n', 'df_record_key_column_name: ', df_record_key_column_name, '\n')


# # Lists of Column Names
# df_nondata_column_names
# df_useless_column_names
# df_int_column_names
# df_float_column_names
# df_datetime_column_names
# df_date_column_names
# df_object_column_names
# df_boolean_column_names
# df_categorical_column_names
# df_numerical_column_names

# In[34]:


df_nondata_column_names = ['SK_ID_CURR']


# In[35]:


df_useless_column_names = ['FLAG_MOBIL']
#> 1    307510
#> 0         1
#> Name: FLAG_MOBIL, dtype: int64


# In[36]:


df_int_column_names = []


# In[37]:


df_float_column_names = []


# In[38]:


df_datetime_column_names = []


# In[39]:


df_date_column_names = []


# In[40]:


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


# In[41]:


df_boolean_column_names = [
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
    'LIVE_CITY_NOT_WORK_CITY',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'TARGET'
]


# In[42]:


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
    'WEEKDAY_APPR_PROCESS_START',
    
'HOUR_APPR_PROCESS_START',
'HOUSETYPE_MODE',
'LIVE_CITY_NOT_WORK_CITY',
'LIVE_REGION_NOT_WORK_REGION'
]


# In[43]:


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
    'AMT_REQ_CREDIT_BUREAU_YEAR'
    'CNT_CHILDREN',
    'CNT_FAM_MEMBERS',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_ID_PUBLISH',
    'DAYS_LAST_PHONE_CHANGE',
    'DAYS_REGISTRATION',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
'FLAG_MOBIL',
'FLAG_EMP_PHONE',
'FLAG_WORK_PHONE',
'FLAG_CONT_MOBILE',
'FLAG_PHONE',
'FLAG_EMAIL',
    'HOUR_APPR_PROCESS_START',
'SK_ID_CURR',
'TARGET',

'REGION_RATING_CLIENT',
'REGION_RATING_CLIENT_W_CITY',

'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION',
'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'LIVE_CITY_NOT_WORK_CITY',
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
'REGION_POPULATION_RELATIVE',

'OWN_CAR_AGE',


'APARTMENTS_AVG',
'BASEMENTAREA_AVG',
'YEARS_BEGINEXPLUATATION_AVG',
'YEARS_BUILD_AVG',
'COMMONAREA_AVG',
'ELEVATORS_AVG',
'ENTRANCES_AVG',
'FLOORSMAX_AVG',
'FLOORSMIN_AVG',
'LANDAREA_AVG',
'LIVINGAPARTMENTS_AVG',
'LIVINGAREA_AVG',
'NONLIVINGAPARTMENTS_AVG',
'NONLIVINGAREA_AVG',
'APARTMENTS_MODE',
'BASEMENTAREA_MODE',
'YEARS_BEGINEXPLUATATION_MODE',
'YEARS_BUILD_MODE',
'COMMONAREA_MODE',
'ELEVATORS_MODE',
'ENTRANCES_MODE',
'FLOORSMAX_MODE',
'FLOORSMIN_MODE',
'LANDAREA_MODE',
'LIVINGAPARTMENTS_MODE',
'LIVINGAREA_MODE',
'NONLIVINGAPARTMENTS_MODE',
'NONLIVINGAREA_MODE',
'APARTMENTS_MEDI',
'BASEMENTAREA_MEDI',
'YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BUILD_MEDI',
'COMMONAREA_MEDI',
'ELEVATORS_MEDI',
'ENTRANCES_MEDI',
'FLOORSMAX_MEDI',
'FLOORSMIN_MEDI',
'LANDAREA_MEDI',
'LIVINGAPARTMENTS_MEDI',
'LIVINGAREA_MEDI',
'NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAREA_MEDI',
'TOTALAREA_MODE',
'OBS_30_CNT_SOCIAL_CIRCLE',
'DEF_30_CNT_SOCIAL_CIRCLE',
'OBS_60_CNT_SOCIAL_CIRCLE',
'DEF_60_CNT_SOCIAL_CIRCLE'
]

