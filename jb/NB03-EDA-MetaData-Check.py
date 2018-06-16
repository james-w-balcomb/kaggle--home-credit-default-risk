
# coding: utf-8

# In[1]:


#%run NB01-Load.ipynb


# In[2]:


get_ipython().magic('run NB02-EDA-MetaData.ipynb')


# In[3]:


# Columns where ‘Percentage of Unique Values Per Column’ is 100%
for column_name in df_columns_percentage_of_unique_values:
    if df_columns_percentage_of_unique_values[column_name] == 1:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[4]:


# ?Columns where ‘Percentage of Unique Values Per Column’ is GTE 80%?
for column_name in df_columns_percentage_of_unique_values:
    if df_columns_percentage_of_unique_values[column_name] >= 0.80:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[5]:


# Columns where ‘Count of Unique Values Per Column’ is 2
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[6]:


# Columns where ‘Count of Unique Values Per Column’ is 2
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (df[column_name].value_counts(dropna=False))


# In[7]:


# ?Columns where ‘Count of Unique Values Per Column’ is 3?
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[8]:


# ?Columns where ‘Count of Unique Values Per Column’ is 3?
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print (df[column_name].value_counts(dropna=False))


# In[9]:


# ?Columns where ‘Percentage of Missing Values Per Column’ is GTE 60%?
for column_name in df_columns_missing_values_percentage:
    if df_columns_missing_values_percentage[column_name] >= 0.60:
        print (column_name, df_columns_missing_values_percentage[column_name])


# In[10]:


# ?Columns where ‘Percentage of Missing Values Per Column’ is GTE 30%?
for column_name in df_columns_missing_values_percentage:
    if df_columns_missing_values_percentage[column_name] >= 0.30:
        print (column_name, df_columns_missing_values_percentage[column_name])

