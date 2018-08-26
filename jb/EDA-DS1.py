
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', 'NB03-EDA-MetaData-Check.ipynb')


# In[ ]:


# Columns where ‘Count of Unique Values Per Column’ is 2
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (column_name, 'dtype: ', df[column_name].dtype)


# # Dichotomous? (AKA binary / boolean)
# Columns where ‘Count of Unique Values Per Column’ is 2
# 
# ### Dichotomous:
# 
# FLAG_CONT_MOBILE  
# FLAG_DOCUMENT_2  
# FLAG_DOCUMENT_3  
# FLAG_DOCUMENT_4  
# FLAG_DOCUMENT_5  
# FLAG_DOCUMENT_6  
# FLAG_DOCUMENT_7  
# FLAG_DOCUMENT_8  
# FLAG_DOCUMENT_9  
# FLAG_DOCUMENT_10  
# FLAG_DOCUMENT_11  
# FLAG_DOCUMENT_12  
# FLAG_DOCUMENT_13  
# FLAG_DOCUMENT_14  
# FLAG_DOCUMENT_15  
# FLAG_DOCUMENT_16  
# FLAG_DOCUMENT_17  
# FLAG_DOCUMENT_18  
# FLAG_DOCUMENT_19  
# FLAG_DOCUMENT_20  
# FLAG_DOCUMENT_21  
# FLAG_EMAIL  
# FLAG_EMP_PHONE  
# FLAG_MOBIL  
# FLAG_OWN_CAR  
# FLAG_OWN_REALTY  
# FLAG_PHONE  
# FLAG_WORK_PHONE  
# LIVE_CITY_NOT_WORK_CITY  
# LIVE_REGION_NOT_WORK_REGION  
# NAME_CONTRACT_TYPE  
# REG_CITY_NOT_LIVE_CITY  
# REG_CITY_NOT_WORK_CITY  
# REG_REGION_NOT_LIVE_REGION  
# REG_REGION_NOT_WORK_REGION  
# 
# ### Polychotomous
# EMERGENCYSTATE_MODE  
# 
# ### Non-Model
# TARGET  

# In[ ]:


df_dichotomous_column_names = [
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
'NAME_CONTRACT_TYPE',
'REG_CITY_NOT_LIVE_CITY',
'REG_CITY_NOT_WORK_CITY',
'REG_REGION_NOT_LIVE_REGION',
'REG_REGION_NOT_WORK_REGION'
]


# In[ ]:


for column_name in df_dichotomous_column_names:
    if df_columns_number_of_unique_values[column_name] == 2:
        print(df[column_name].value_counts(dropna=False, sort=True))
        print(df[column_name].value_counts(dropna=False, normalize=True, sort=True))


# # NOT 0/1
# 
# ### FLAG_OWN_CAR
# N    202924 (0.659892)  
# Y    104587 (0.340108)  
# 
# ### FLAG_OWN_REALTY
# Y    213312 (0.693673)  
# N     94199 (0.306327)  
# 
# ### NAME_CONTRACT_TYPE
# Cash loans         278232 (0.904787)  
# Revolving loans     29279 (0.095213)  
# NOTE: This is to be *converted* to a boolean, a la dummy coded, with "Cash loans" as the reference category.

# In[ ]:


df['FLAG_OWN_CAR'] = pd.Series(np.where(df['FLAG_OWN_CAR'].values == 'Y', 1, 0), df.index)
df['FLAG_OWN_CAR'].value_counts(dropna=False, sort=True)


# In[ ]:


df['FLAG_OWN_REALTY'] = pd.Series(np.where(df['FLAG_OWN_REALTY'].values == 'Y', 1, 0), df.index)
df['FLAG_OWN_REALTY'].value_counts(dropna=False, sort=True)


# In[ ]:


#df['NAME_CONTRACT_TYPE__Revolving_loans'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE'].values == 'Revolving loans', 1, 0), df.index)
#df.drop('NAME_CONTRACT_TYPE', axis=1)
df.rename(columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE__Revolving_loans'}, inplace=True)
#df = df.rename(columns={'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE__Revolving_loans'})
df['NAME_CONTRACT_TYPE__Revolving_loans'] = pd.Series(np.where(df['NAME_CONTRACT_TYPE__Revolving_loans'].values == 'Revolving loans', 1, 0), df.index)
df['NAME_CONTRACT_TYPE__Revolving_loans'].value_counts(dropna=False, sort=True)


# In[ ]:


# Columns where ‘Count of Unique Values Per Column’ is 3
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


# Columns where ‘Count of Unique Values Per Column’ is 3
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 3:
        print(df[column_name].value_counts(dropna=False, sort=True))
        print(df[column_name].value_counts(dropna=False, normalize=True, sort=True))


# ### CODE_GENDER
# Looks like we should deal with those four records with a missing value place-holder and convert it to a boolean, with "F" as the reference level.
# 
# ### HOUSETYPE_MODE
# Not sure what is going on with this one. A little EDA provided no indication of pattern to the missingness.  
# It seems likely that this is Missing Not At Random (MNAR) and, therefore, we should dig deeper.
# For now, moving forward with replacing the missing values with "MISSING" and treating it as "Unknown/Not Provided".  
# 
# ### REGION_RATING_CLIENT
# This should be treated as a Categorical - Ordinal variable.
# Check this relationship between this and REGION_RATING_CLIENT_W_CITY.
# 
# ### REGION_RATING_CLIENT_W_CITY
# This should be treated as a Categorical - Ordinal variable.
# Check this relationship between this and REGION_RATING_CLIENT.

# # CODE_GENDER

# In[ ]:


df.loc[df.SK_ID_CURR == 141289, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 319880, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 196708, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 144669, 'CODE_GENDER'] = 'M'


# # HOUSETYPE_MODE

# # REGION_RATING_CLIENT

# # REGION_RATING_CLIENT_W_CITY

# In[ ]:


df[(df['FLAG_OWN_CAR'] == 0) & (df['OWN_CAR_AGE'].notnull())]


# In[ ]:


df[(df['FLAG_OWN_CAR'] == 1) & (df['OWN_CAR_AGE'].isnull())]


# In[ ]:


df[(df['FLAG_OWN_REALTY'] == 0) & (df['HOUSETYPE_MODE'].notnull())]


# In[ ]:


df[(df['FLAG_OWN_REALTY'] == 1) & (df['HOUSETYPE_MODE'].isnull())]


# # Correlation

# In[ ]:


print(df['EXT_SOURCE_1'].corr(df['EXT_SOURCE_2'], method='pearson'))
print(df['EXT_SOURCE_1'].corr(df['EXT_SOURCE_2'], method='spearman'))
print(df['EXT_SOURCE_1'].corr(df['EXT_SOURCE_2'], method='kendall'))


# In[ ]:


print(df['EXT_SOURCE_2'].corr(df['EXT_SOURCE_3'], method='pearson'))
print(df['EXT_SOURCE_2'].corr(df['EXT_SOURCE_3'], method='spearman'))
print(df['EXT_SOURCE_2'].corr(df['EXT_SOURCE_3'], method='kendall'))


# In[ ]:


print(df['EXT_SOURCE_3'].corr(df['EXT_SOURCE_1'], method='pearson'))
print(df['EXT_SOURCE_3'].corr(df['EXT_SOURCE_1'], method='spearman'))
print(df['EXT_SOURCE_3'].corr(df['EXT_SOURCE_1'], method='kendall'))


# In[ ]:


print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_1'], method='pearson'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_1'], method='spearman'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_1'], method='kendall'))


# In[ ]:


print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_2'], method='pearson'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_2'], method='spearman'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_2'], method='kendall'))


# In[ ]:


print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_3'], method='pearson'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_3'], method='spearman'))
print(df['AMT_INCOME_TOTAL'].corr(df['EXT_SOURCE_3'], method='kendall'))


# In[ ]:


tmp_df = df[['TARGET', 'EXT_SOURCE_1']]
tmp_df = tmp_df.dropna()
tmp_df['TARGET'] = pd.to_numeric(tmp_df['TARGET'], errors='coerce')
print(tmp_df['TARGET'].dtype)
tmp_df['EXT_SOURCE_1'] = pd.to_numeric(tmp_df['EXT_SOURCE_1'], errors='coerce')
print(tmp_df['EXT_SOURCE_1'].dtype)
print(pointbiserialr(tmp_df['TARGET'], tmp_df['EXT_SOURCE_1']))


# In[ ]:


tmp_df = df[['TARGET', 'EXT_SOURCE_2']]
tmp_df = tmp_df.dropna()
pointbiserialr(tmp_df['TARGET'], tmp_df['EXT_SOURCE_2'])


# In[ ]:


tmp_df = df[['TARGET', 'EXT_SOURCE_3']]
tmp_df = tmp_df.dropna()
pointbiserialr(tmp_df['TARGET'], tmp_df['EXT_SOURCE_3'])


# In[ ]:


tmp_df = df[['TARGET', 'AMT_INCOME_TOTAL']]
tmp_df = tmp_df.dropna()
pointbiserialr(tmp_df['TARGET'], tmp_df['AMT_INCOME_TOTAL'])


# In[ ]:


sns.boxplot(x=df['AMT_INCOME_TOTAL']);


# In[ ]:


#sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df);


# In[ ]:


#az = sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df);
#az = sns.swarmplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df, color='0.25');


# In[ ]:


sns.violinplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df, scale='count');


# In[ ]:


print(df['AMT_INCOME_TOTAL'].min())
print(df['AMT_INCOME_TOTAL'].max())
print(df['AMT_INCOME_TOTAL'].mean())
print(df['AMT_INCOME_TOTAL'].median())


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'], 10))
print(np.percentile(df['AMT_INCOME_TOTAL'], 20))
print(np.percentile(df['AMT_INCOME_TOTAL'], 30))
print(np.percentile(df['AMT_INCOME_TOTAL'], 40))
print(np.percentile(df['AMT_INCOME_TOTAL'], 50))
print(np.percentile(df['AMT_INCOME_TOTAL'], 60))
print(np.percentile(df['AMT_INCOME_TOTAL'], 70))
print(np.percentile(df['AMT_INCOME_TOTAL'], 80))
print(np.percentile(df['AMT_INCOME_TOTAL'], 90))


# In[ ]:


tmp_percentile_count_boundaries = list()
tmp_percentile_count_boundaries.append(df['AMT_INCOME_TOTAL'].min() - 1)
for iter in range(10,100,10):
    tmp_percentile_count_boundaries.append(np.percentile(df['AMT_INCOME_TOTAL'], iter))
tmp_percentile_count_boundaries.append(df['AMT_INCOME_TOTAL'].max() + 1)
sorted(tmp_percentile_count_boundaries)


# In[ ]:


tmp_percentile_10th = np.percentile(df['AMT_INCOME_TOTAL'], 10)
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= tmp_percentile_10th)].count())
tmp_percentile_10th


# In[ ]:


column_name = 'AMT_INCOME_TOTAL'
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[0]) & (df[column_name] <= tmp_percentile_count_boundaries[1])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[1]) & (df[column_name] <= tmp_percentile_count_boundaries[2])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[2]) & (df[column_name] <= tmp_percentile_count_boundaries[3])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[3]) & (df[column_name] <= tmp_percentile_count_boundaries[4])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[4]) & (df[column_name] <= tmp_percentile_count_boundaries[5])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[5]) & (df[column_name] <= tmp_percentile_count_boundaries[6])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[6]) & (df[column_name] <= tmp_percentile_count_boundaries[7])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[7]) & (df[column_name] <= tmp_percentile_count_boundaries[8])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[8]) & (df[column_name] <= tmp_percentile_count_boundaries[9])].count())
print(df[column_name][(df[column_name] >= tmp_percentile_count_boundaries[9]) & (df[column_name] <= tmp_percentile_count_boundaries[10])].count())


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'],  1))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= np.percentile(df['AMT_INCOME_TOTAL'],  1))].count()[0])
print(np.percentile(df['AMT_INCOME_TOTAL'], 99))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] >= np.percentile(df['AMT_INCOME_TOTAL'], 99))].count()[0])


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'],  0.1))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= np.percentile(df['AMT_INCOME_TOTAL'],  0.1))].count()[0])
print(np.percentile(df['AMT_INCOME_TOTAL'], 99.9))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] >= np.percentile(df['AMT_INCOME_TOTAL'], 99.9))].count()[0])


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'],  0.01))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= np.percentile(df['AMT_INCOME_TOTAL'],  0.01))].count()[0])
print(np.percentile(df['AMT_INCOME_TOTAL'], 99.99))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] >= np.percentile(df['AMT_INCOME_TOTAL'], 99.99))].count()[0])


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'],  0.001))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= np.percentile(df['AMT_INCOME_TOTAL'],  0.001))].count()[0])
print(np.percentile(df['AMT_INCOME_TOTAL'], 99.999))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] >= np.percentile(df['AMT_INCOME_TOTAL'], 99.999))].count()[0])


# In[ ]:


print(np.percentile(df['AMT_INCOME_TOTAL'],  0.0001))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] <= np.percentile(df['AMT_INCOME_TOTAL'],  0.0001))].count()[0])
print(np.percentile(df['AMT_INCOME_TOTAL'], 99.9999))
print(df[['AMT_INCOME_TOTAL']][(df['AMT_INCOME_TOTAL'] >= np.percentile(df['AMT_INCOME_TOTAL'], 99.9999))].count()[0])


# In[ ]:


column_name = 'AMT_INCOME_TOTAL'


# In[ ]:


df[column_name].dtype


# In[ ]:


print(df[column_name].min())
print(df[column_name].max())


# In[ ]:


np.histogram(df[column_name])


# In[ ]:


plt.hist(df[column_name])


# In[ ]:


plt.hist(df[column_name])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


plt.hist(df[column_name],
         bins=[0,10000,100000,1000000,10000000,120000000])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=df[column_name],
                            bins='auto',
                            color='#0504aa',
                            alpha=0.7,
                            rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of ' + column_name)
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=df[column_name],
                            bins=10,
                            color='#0504aa',
                            alpha=0.7,
                            rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of ' + column_name)
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()


# In[ ]:


plt.hist(df[column_name],
         bins=30,
         normed=True,
         alpha=0.5,
         histtype='stepfilled',
         color='steelblue',
         edgecolor='none')
#plt.ticklabel_format(useOffset=False, style='plain', axis='both')
#plt.ticklabel_format(style='plain', axis='both')
#plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


df[column_name].plot.hist(grid=True,
                           bins=20,
                           rwidth=0.9,
                           color='#607c8e')
plt.title('Histogram of ' + column_name)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


sp.stats.skew(df[column_name])


# In[ ]:


sp.stats.kurtosis(df[column_name])


# In[ ]:


df[column_name].max()


# In[ ]:


df[df[column_name] == 117000000.0]
# SK_ID_CURR
# 114967


# In[ ]:


df[[column_name]][df[column_name] != 117000000.0].max()


# In[ ]:


#plt.hist(df[[column_name]][df[column_name] != 117000000.0])
#plt.ticklabel_format(style='plain', axis='x')
#plt.show()
#> ---------------------------------------------------------------------------
#> AttributeError                            Traceback (most recent call last)
#> ~\Anaconda3\lib\site-packages\matplotlib\axes\_base.py in ticklabel_format(self, **kwargs)
#>    2567                 if axis == 'both' or axis == 'x':
#> -> 2568                     self.xaxis.major.formatter.set_scientific(sb)
#>    2569                 if axis == 'both' or axis == 'y':
#> 
#> AttributeError: 'StrCategoryFormatter' object has no attribute 'set_scientific'
#> 
#> During handling of the above exception, another exception occurred:
#> 
#> AttributeError                            Traceback (most recent call last)
#> <ipython-input-42-c0b04d2067b5> in <module>()
#>       1 plt.hist(df[[column_name]][df[column_name] != 117000000.0])
#> ----> 2 plt.ticklabel_format(style='plain', axis='x')
#>       3 plt.show()
#> 
#> ~\Anaconda3\lib\site-packages\matplotlib\pyplot.py in ticklabel_format(**kwargs)
#>    3749 @docstring.copy_dedent(Axes.ticklabel_format)
#>    3750 def ticklabel_format(**kwargs):
#> -> 3751     ret = gca().ticklabel_format(**kwargs)
#>    3752     return ret
#>    3753 
#> 
#> ~\Anaconda3\lib\site-packages\matplotlib\axes\_base.py in ticklabel_format(self, **kwargs)
#>    2591         except AttributeError:
#>    2592             raise AttributeError(
#> -> 2593                 "This method only works with the ScalarFormatter.")
#>    2594 
#>    2595     def locator_params(self, axis='both', tight=None, **kwargs):
#> 
#> AttributeError: This method only works with the ScalarFormatter.


# In[ ]:


df[df['SK_ID_CURR'] == '114967']['AMT_INCOME_TOTAL']
#> 12840    117000000.0
#> Name: AMT_INCOME_TOTAL, dtype: float64


# In[ ]:


df.loc[df.SK_ID_CURR == '114967', 'AMT_INCOME_TOTAL'] = 0


# In[ ]:


df[df['SK_ID_CURR'] == '114967']['AMT_INCOME_TOTAL']
#> 12840    0.0
#> Name: AMT_INCOME_TOTAL, dtype: float64


# In[ ]:


plt.hist(df[column_name])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


df[column_name].max()


# In[ ]:


df[df[column_name] == 18000090.0]['SK_ID_CURR']
#> 203693    336147
#> Name: SK_ID_CURR, dtype: object


# In[ ]:


df[df['SK_ID_CURR'] == '336147']['AMT_INCOME_TOTAL']
#> 203693    18000090.0
#> Name: AMT_INCOME_TOTAL, dtype: float64


# In[ ]:


df.loc[df.SK_ID_CURR == '336147', 'AMT_INCOME_TOTAL'] = 0


# In[ ]:


df[df['SK_ID_CURR'] == '336147']['AMT_INCOME_TOTAL']
#> 203693    0.0
#> Name: AMT_INCOME_TOTAL, dtype: float64


# In[ ]:


plt.hist(df[column_name])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


df[column_name].max()


# In[ ]:


df[column_name].min()


# In[ ]:


df[column_name].median()


# In[ ]:


plt.hist(df[column_name],
         bins=[0,25650,147150,1000000,120000000])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


plt.hist(df[column_name],
         bins=[0,10000,20000,30000,40000,50000,60000,70000,80000,90000,
               100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,
               200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,
               300000,310000,320000,330000,340000,350000,360000,370000,380000,390000,
               400000,410000,420000,430000,440000,450000,460000,470000,480000,490000,
               500000,510000,520000,530000,540000,550000,560000,570000,580000,590000,
               600000,610000,620000,630000,640000,650000,660000,670000,680000,690000,
               700000,710000,720000,730000,740000,750000,760000,770000,780000,790000,
               800000,810000,820000,830000,840000,850000,860000,870000,880000,890000,
               900000,910000,920000,930000,940000,950000,960000,970000,980000,990000,
               1000000])
plt.ticklabel_format(style='plain', axis='x')
plt.show()


# In[ ]:


df['ratio__AMT_CREDIT__ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']


# In[ ]:


df['ratio__AMT_INCOME_TOTAL__AMT_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']


# In[ ]:


# How much of the price of the good was financed?
df['ratio__AMT_CREDIT_AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']


# In[ ]:


DAYS_BIRTH
DAYS_EMPLOYED
DAYS_ID_PUBLISH
DAYS_LAST_PHONE_CHANGE
DAYS_REGISTRATION


# In[ ]:


df['ratio__DAYS_EMPLOYED__DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']


# In[ ]:


df['ratio__DAYS_ID_PUBLISH__DAYS_BIRTH'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']


# In[ ]:


df['ratio__DAYS_LAST_PHONE_CHANGE__DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']


# In[ ]:


df['ratio__DAYS_REGISTRATION__DAYS_BIRTH'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']


# In[ ]:


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


# In[ ]:


df[flag_document_column_names].count(axis=1)


# In[ ]:


df[flag_document_column_names].sum(axis=1)


# In[ ]:


df[flag_document_column_names].dtypes


# In[ ]:


df[flag_document_column_names].astype('int').sum(axis=1)


# In[ ]:


df['FLAG_DOCUMENT__count'] = df[flag_document_column_names].astype('int').sum(axis=1)


# In[ ]:


df['FLAG_DOCUMENT__count']


# In[ ]:


list(sorted(set(df['CNT_CHILDREN'].values)))


# In[ ]:


df['CNT_CHILDREN'].value_counts(ascending=True, dropna=False, sort=True)


# In[ ]:


df['CNT_CHILDREN'].value_counts(ascending=True, dropna=False, normalize=True, sort=True)


# # DAYS...
# DAYS_BIRTH  
# DAYS_EMPLOYED  
# DAYS_ID_PUBLISH  
# DAYS_LAST_PHONE_CHANGE  
# DAYS_REGISTRATION  

# In[ ]:


df['DAYS_BIRTH'].min() / 365.25 * -1


# In[ ]:


df['DAYS_BIRTH'].max() / 365.25 * -1


# In[ ]:


df['DAYS_BIRTH'].value_counts(ascending=True, dropna=False, sort=True)


# In[ ]:


df['DAYS_BIRTH'].value_counts(ascending=True, dropna=False, normalize=True, sort=True)


# In[ ]:


df['DAYS_BIRTH'].hist();


# HOUSETYPE_MODE
# FLAG_OWN_REALTY

# In[ ]:


df['HOUSETYPE_MODE'].value_counts(ascending=True, dropna=False, sort=True)


# In[ ]:


df['FLAG_OWN_REALTY'].value_counts(ascending=True, dropna=False, sort=True)


# In[ ]:


pd.crosstab(df['FLAG_OWN_REALTY'], df['HOUSETYPE_MODE'], dropna=False)


# In[ ]:


pd.crosstab(df['HOUSETYPE_MODE'], df['FLAG_OWN_REALTY'], dropna=False)


# In[ ]:


import scipy.stats as stats


# In[ ]:


df_crosstab = pd.crosstab(index=df['FLAG_OWN_REALTY'], columns=df['HOUSETYPE_MODE'])
print(df_crosstab)
odds_ratio, p_value = stats.fisher_exact(df_crosstab)
print(odds_ratio)
print(p_value)
chi_square_statistic, p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(df_crosstab)
print(chi_square_statistic)
print(p_value)
print(degrees_of_freedom)
print(expected_frequencies)


# In[ ]:


df['HOUSETYPE_MODE'].fillna('MISSING', inplace=True)


# # Missing Values
# # Missing Values Place-Holders
# 
# ### "XNA"
# ### 365243

# In[ ]:


df[df['CODE_GENDER'] == 'XNA']


# In[ ]:


df[df['ORGANIZATION_TYPE'] == 'XNA']


# In[ ]:


df[df['DAYS_EMPLOYED'] == 365243]


# In[ ]:


#df[df['CODE_GENDER'] == 'XNA']
#df[[['SK_ID_CURR','DAYS_EMPLOYED']],df['DAYS_EMPLOYED'] == 365243]
df[['SK_ID_CURR','DAYS_EMPLOYED']df['DAYS_EMPLOYED'] == 365243]


# In[ ]:


#df.loc[df['DAYS_EMPLOYED'] == 365243, 'SK_ID_CURR']
df.loc[df['DAYS_EMPLOYED'] == 365243, ['SK_ID_CURR', 'DAYS_EMPLOYED']]


# FLAG_OWN_CAR  
# OWN_CAR_AGE  
# 
# FLAG_OWN_REALTY  
# 

# In[ ]:


df['FLAG_OWN_REALTY'].value_counts()


# In[ ]:


df['FLAG_OWN_CAR'].value_counts()


# In[ ]:


df['OWN_CAR_AGE'].describe()


# In[ ]:


df['OWN_CAR_AGE'].isnull().sum()


# In[ ]:


sns.violinplot(y='OWN_CAR_AGE', data=df, scale='count');


# In[ ]:


sns.violinplot(x='TARGET', y='OWN_CAR_AGE', data=df, scale='count');


# In[ ]:


sns.boxplot(y=df['OWN_CAR_AGE']);


# In[ ]:


sns.boxplot(x='TARGET', y='OWN_CAR_AGE', data=df);

