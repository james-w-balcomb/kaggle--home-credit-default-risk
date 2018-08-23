
# coding: utf-8

# In[ ]:


#%run NB01-Load.ipynb
get_ipython().run_line_magic('run', 'NB02-EDA-MetaData.ipynb')


# In[ ]:


print("Columns where 'Percentage of Unique Values Per Column' is 100%")
for column_name in df_columns_percentage_of_unique_values:
    if df_columns_percentage_of_unique_values[column_name] == 1:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


print("Columns where 'Percentage of Unique Values Per Column' is GTE 80%?")
for column_name in df_columns_percentage_of_unique_values:
    if df_columns_percentage_of_unique_values[column_name] >= 0.80:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


print("Columns where 'Count of Unique Values Per Column' is 2")
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


print("Columns where 'Count of Unique Values Per Column' is 2")
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (df[column_name].value_counts(dropna=False))


# In[ ]:


print("Columns where 'Count of Unique Values Per Column' is 3")
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print(column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


print("Columns where 'Count of Unique Values Per Column' is 3")
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print(df[column_name].value_counts(dropna=False))


# In[ ]:


print("Columns where 'Percentage of Missing Values Per Column' is GTE 60%")
for column_name in df_columns_missing_values_percentage:
    if df_columns_missing_values_percentage[column_name] >= 0.60:
        print (column_name, df_columns_missing_values_percentage[column_name])


# In[ ]:


print("Columns where 'Percentage of Missing Values Per Column' is GTE 30%")
for column_name in df_columns_missing_values_percentage:
    if df_columns_missing_values_percentage[column_name] >= 0.30:
        print (column_name, df_columns_missing_values_percentage[column_name])


# In[ ]:


print("Categorical Variables with Missing Values")
for column_name in df_categorical_column_names:
    if df_columns_missing_values_percentage[column_name] > 0.00:
        print (column_name, df_columns_missing_values_percentage[column_name])


# In[ ]:


print("Numerical Variables with Missing Values")
for column_name in df_numerical_column_names:
    if df_columns_missing_values_percentage[column_name] > 0.00:
        print (column_name, df_columns_missing_values_percentage[column_name])


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

# In[ ]:


# 1) Always Positive / Never Negative
print("Numerical Variables that are Always Positive / Never Negative")
for column_name in df_columns_always_positive_flag:
    if df_columns_always_positive_flag[column_name] == True:
        print (column_name, df_columns_always_positive_flag[column_name])


# In[ ]:


# 2) Always Negative / Never Positive
print("Numerical Variables that are Always Negative / Never Positive")
for column_name in df_columns_always_negative_flag:
    if df_columns_always_negative_flag[column_name] == True:
        print (column_name, df_columns_always_negative_flag[column_name])


# In[ ]:


# 3) Always Between 1 and 0  
print("Numerical Variables that are Always Between 1 and 0")
for column_name in df_columns_always_between_one_and_zero_flag:
    if df_columns_always_between_one_and_zero_flag[column_name] == True:
        print (column_name, df_columns_always_between_one_and_zero_flag[column_name])


# In[ ]:


# 4) Has Zero  
print("Numerical Variables that are Has Zero")
for column_name in df_columns_has_zero_flag:
    if df_columns_has_zero_flag[column_name] == True:
        print (column_name, df_columns_has_zero_flag[column_name])


# In[ ]:


# 5) Never Zero  
print("Numerical Variables that are Never Zero")
for column_name in df_columns_never_zero_flag:
    if df_columns_never_zero_flag[column_name] == True:
        print (column_name, df_columns_never_zero_flag[column_name])


# In[ ]:


# 6) Has Mean of 0  
print("Numerical Variables that are Has Mean of 0")
for column_name in df_columns_has_mean_of_zero_flag:
    if df_columns_has_mean_of_zero_flag[column_name] == True:
        print (column_name, df_columns_has_mean_of_zero_flag[column_name])


# In[ ]:


# 7) Has Standard Deviation of 1  
print("Numerical Variables that are Has Standard Deviation of 1")
for column_name in df_columns_has_standard_deviation_of_one_flag:
    if df_columns_has_standard_deviation_of_one_flag[column_name] == True:
        print (column_name, df_columns_has_standard_deviation_of_one_flag[column_name])


# In[ ]:


# 8) Is Mean Centered  
print("Numerical Variables that are Is Centered")
for column_name in df_columns_is_mean_centered_flag:
    if df_columns_is_mean_centered_flag[column_name] == True:
        print (column_name, df_columns_is_mean_centered_flag[column_name])


# In[ ]:


# 9) Is Scaled  
print("Numerical Variables that are Is Scaled")
for column_name in df_columns_is_scaled_flag:
    if df_columns_is_scaled_flag[column_name] == True:
        print (column_name, df_columns_is_scaled_flag[column_name])


# In[ ]:


# 10) Is Z-Score (AKA Standardized, Normalized, Centered and Scaled)
print("Numerical Variables that are Is Z-Score")
for column_name in df_columns_is_z_score_flag:
    if df_columns_is_z_score_flag[column_name] == True:
        print (column_name, df_columns_is_z_score_flag[column_name])


# # Always Between 1 and 0 == True, but not listed as "normalized" in the data dictionary
# 
# 

# In[ ]:


df['AMT_ANNUITY'].value_counts()


# In[ ]:


df['AMT_CREDIT'].value_counts()


# In[ ]:


df['AMT_GOODS_PRICE'].value_counts()

