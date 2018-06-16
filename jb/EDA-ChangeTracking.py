
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import random


# In[2]:


seed=1234567890
random.seed(seed)
np.random.seed(seed)


# In[3]:


data_file_path = 'DATA_FILE_PATH' in os.environ
data_file_path


# In[4]:


# Set configuration based on environment variables
# os.environ['DATA_FILE_PATH']
data_file_path = 'DATA_FILE_PATH' in os.environ
if data_file_path:
    print('Using Environment Variable for data_file_path')
else:
    data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
    print('Using Hard-Coded Configuration for data_file_path')
data_file_path


# In[5]:


data_file_name = 'application_train.csv'
data_file_name


# In[6]:


df = pd.read_table(data_file_path + data_file_name, sep=',')


# In[11]:


df_current='df'


# In[16]:


df_record_id_column_name = 'SK_ID_CURR'


# In[22]:


dict_data_changes = {
    'DataFileName':'application_train.csv',
    'DataFrameName':'application_train',
    'RecordIdColumnName':'SK_ID_CURR',
    'RecordId':'144669',
    'ValueChangedColumnName':'CODE_GENDER',
    'ValueChangedFrom':'XNA',
    'ValueChangedTo':'F',
    'AuthorName':'JamesBalcomb',
    'DateTime':'2018-06-15 12:03:00'
}
dict_data_changes


# In[23]:


list_dict_data_changes = [{
    'DataFileName':'application_train.csv',
    'DataFrameName':'application_train',
    'RecordIdColumnName':'SK_ID_CURR',
    'RecordId':'144669',
    'ValueChangedColumnName':'CODE_GENDER',
    'ValueChangedFrom':'XNA',
    'ValueChangedTo':'F',
    'AuthorName':'JamesBalcomb',
    'DateTime':'2018-06-15 12:03:00'
}]
list_dict_data_changes


# In[26]:


# Using module time ()
import time
ts = time.time()  # number of seconds since the epoch
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required
ts_human_readable = time.ctime(ts)
print(ts_human_readable)
# Using module datetime
import datetime;
ts = datetime.datetime.now().timestamp()  # number of seconds since the epoch
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required
ts_human_readable = datetime.datetime.fromtimestamp(ts).isoformat()
print(ts_human_readable)
# Using module calendar
import calendar;
import time;
ts = calendar.timegm(time.gmtime())
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required
# using Pandas module
import pandas
ts = pandas.Timestamp.now(tz=None)  # current time local to tz
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required


# In[35]:


#
# Use-Case: Manually Change One Value
# Use-Case: Change All Values In A Column Based On The Existing Value (e.g., np.nan, if "XAN")
def changes_value(DataFileName=None,
                  DataFrameName=None,
                  RecordIdColumnName=None,
                  RecordId=None,
                  ValueChangedColumnName=None,
                  ValueChangedFrom=None,
                  ValueChangedTo=None,
                  AuthorName=None):
    """
    params:
        DataFileName:
        DataFrameName:
        RecordIdColumnName:
        RecordId:
        ValueChangedColumnName:
        ValueChangedFrom:
        ValueChangedTo:
        AuthorName:
        -DateTime
    """
    if not DataFileName:
        DataFileName = data_file_name
    if not DataFrameName:
        DataFileName = df_current
    if not RecordIdColumnName:
        RecordIdColumnName = df_record_id_column_name


def change_values_matching(DataFileName=None,
                           DataFrameName=None,
                           RecordIdColumnName=None,
                           TargetColumnName=None,
                           ValueToChangeFrom=None,
                           ValueToChangeTo=None,
                           AuthorName=None):
    """
    params:
        DataFileName:
        DataFrameName:
        RecordIdColumnName:
        -RecordId:
        ValueChangedColumnName:
        ValueChangedFrom:
        ValueChangedTo:
        AuthorName:
        -DateTime
    """
    if not DataFileName:
        DataFileName = data_file_name
    if not DataFrameName:
        DataFileName = df_current
    if not RecordIdColumnName:
        RecordIdColumnName = df_record_id_column_name
    
    date_time = pandas.Timestamp.now(tz=None)
    
    print('DataFileName: ', DataFileName)
    print('DataFrameName: ', DataFrameName)
    print('RecordIdColumnName: ', RecordIdColumnName)
    print('TargetColumnName: ', TargetColumnName)
    print('ValueToChangeFrom: ', ValueToChangeFrom)
    print('ValueToChangeTo: ', ValueToChangeTo)
    print('AuthorName: ', AuthorName)
    print('DateTime: ', date_time)
    
    dict_data_changes = {
    'DataFileName':DataFileName,
    'DataFrameName':DataFrameName,
    'RecordIdColumnName':RecordIdColumnName,
    'RecordId':None,
    'ValueChangedColumnName':TargetColumnName,
    'ValueChangedFrom':ValueToChangeFrom,
    'ValueChangedTo':ValueToChangeTo,
    'AuthorName':AuthorName,
    'DateTime':ts
    }
    
    #if not list_dict_data_changes:
    #    list_dict_data_changes = dict_data_changes
    #else:
    #    list_dict_data_changes.append(dict_data_changes)
    
    print('list_dict_data_changes: ', list_dict_data_changes)


# In[22]:


changes_value(
    DataFileName='application_train.csv',
    DataFrameName='application_train',
    RecordIdColumnName='SK_ID_CURR',
    RecordId='144669',
    ValueChangedColumnName='CODE_GENDER',
    ValueChangedFrom='XNA',
    ValueChangedTo='F',
    AuthorName='JamesBalcomb'
    #DateTime='2018-06-15 12:03:00'
)


# In[36]:


#pd.Series(np.where(df.FLAG_OWN_CAR.values == 'Y', 1, 0), df.index)
change_values_matching(
    DataFileName=data_file_name,
    DataFrameName=df_current,
    TargetColumnName='FLAG_OWN_CAR',
    ValueToChangeFrom='Y',
    ValueToChangeTo=1,
    AuthorName='James Balcomb'
)

