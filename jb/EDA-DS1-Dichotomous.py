
# coding: utf-8

# In[ ]:


#%run NB01-Load.ipynb
#%run NB02-EDA-MetaData.ipynb
get_ipython().magic('run NB03-EDA-MetaData-Check.ipynb')


# In[ ]:


import scipy.stats as stats


# # Convert / Transform Dichotomous Variables

# In[ ]:


categorical_dichotomous_column_names = [
'CODE_GENDER',
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


# In[ ]:


for column_name in categorical_dichotomous_column_names:
    if df[column_name].nunique() != 2:
        print(df[column_name].value_counts(dropna=False))


# In[ ]:


for column_name in categorical_dichotomous_column_names:
    if set(df[column_name].unique()) != set(['0','1']):
        print(df[column_name].value_counts(dropna=False))


# ### CODE_GENDER

# In[ ]:


df.loc[df['SK_ID_CURR'] == '141289', 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == '319880', 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == '196708', 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == '144669', 'CODE_GENDER'] = 'M'


# In[ ]:


df['CODE_GENDER'] = df['CODE_GENDER'].replace('M', 1)


# In[ ]:


df['CODE_GENDER'] = df['CODE_GENDER'].replace('F', 0)


# In[ ]:


df['CODE_GENDER'] = df['CODE_GENDER'].astype(str)


# ### FLAG_OWN_CAR

# In[ ]:


df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace('Y', 1)


# In[ ]:


df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace('N', 0)


# In[ ]:


df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].astype(str)


# ### FLAG_OWN_REALTY

# In[ ]:


df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace('Y', 1)


# In[ ]:


df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace('N', 0)


# In[ ]:


df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].astype(str)


# # Two-Way Frequency Tables

# ###### scipy.stats.fisher_exact[https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html]
# Notes  
# The calculated odds ratio is different from the one R uses.  
# This scipy implementation returns the (more common) “unconditional Maximum Likelihood Estimate”, while R uses the “conditional Maximum Likelihood Estimate”.  
# For tables with large numbers, the (inexact) chi-square test implemented in the function chi2_contingency can also be used.  
# 
# 
# Fisher's exact test is a statistical test used to determine if there are nonrandom associations between two categorical variables.  
# Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource. http://mathworld.wolfram.com/FishersExactTest.html
# 
# 
# That p-value is for the null hypothesis of independence between the two categorical variables.  
# We reject the null of independence here.  
# For the odds ratio, if the confidence interval contains one, we fail to reject the null hypothesis of independence.  
# "How to interpret Fisher Test?"[https://stats.stackexchange.com/questions/220044/how-to-interpret-fisher-test]
# 
# ##### scipy.stats.chi2_contingency()[https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html]
# Notes  
# An often quoted guideline for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.  
# 
# 
# ...
# 
# 
# ...calculate the fisher exact test to determine statistical significance...  
# 
# 
# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
# 
# 
# http://www.biostathandbook.com/fishers.html
# 
# 
# http://www.stat.purdue.edu/~tqin/system101/method/method_fisher_sas.htm
# 
# 
# https://www.statsdirect.com/help/exact_tests_on_counts/fisher_exact.htm
# 
# 
# https://stats.stackexchange.com/questions/220044/how-to-interpret-fisher-test
# 
# 
# https://codereview.stackexchange.com/questions/186657/pandas-numpy-statistical-odds-ratio-test
# 

# In[ ]:


for column_name in categorical_dichotomous_column_names:
    if column_name == 'TARGET':
        continue
    df_crosstab = pd.crosstab(index=df['TARGET'], columns=df[column_name])
    print(df_crosstab)
    odds_ratio, p_value = stats.fisher_exact(df_crosstab)
    print(odds_ratio)
    print(p_value)
    chi_square_statistic, p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(df_crosstab)
    print(chi_square_statistic)
    print(p_value)
    print(degrees_of_freedom)
    print(expected_frequencies)

