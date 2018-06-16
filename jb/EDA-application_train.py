
# coding: utf-8

# In[ ]:


#%run NB01-Load.ipynb
#%run NB02-EDA-MetaData.ipynb
get_ipython().magic('run NB03-EDA-MetaData-Check.ipynb')


# # Transform boolean to 1/0, instead of Yes/No/, True/False, etc.
# pd.Series(np.where(df.column_name.values == 'yes', 1, 0), df.index)

# In[2]:


# Columns where ‘Count of Unique Values Per Column’ is 2
for column_name in df_columns_number_of_unique_values:
    if df_columns_number_of_unique_values[column_name] == 2:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[1]:


# List of unique values in the df['column_name'] column
# df.column_name.unique()
df.TARGET.unique()


# In[ ]:


pd.Series(np.where(df.housing.values == 'yes', 1, 0), df.index)


# In[ ]:


FLAG_OWN_CAR


# # Columns where ‘Count of Unique Values Per Column’ is 2 (sans NULL)
# TARGET dtype:  int64
# NAME_CONTRACT_TYPE dtype:  object
# FLAG_OWN_CAR dtype:  object
# FLAG_OWN_REALTY dtype:  object
# FLAG_MOBIL dtype:  int64
# FLAG_EMP_PHONE dtype:  int64
# FLAG_WORK_PHONE dtype:  int64
# FLAG_CONT_MOBILE dtype:  int64
# FLAG_PHONE dtype:  int64
# FLAG_EMAIL dtype:  int64
# REG_REGION_NOT_LIVE_REGION dtype:  int64
# REG_REGION_NOT_WORK_REGION dtype:  int64
# LIVE_REGION_NOT_WORK_REGION dtype:  int64
# REG_CITY_NOT_LIVE_CITY dtype:  int64
# REG_CITY_NOT_WORK_CITY dtype:  int64
# LIVE_CITY_NOT_WORK_CITY dtype:  int64
# EMERGENCYSTATE_MODE dtype:  object
# FLAG_DOCUMENT_2 dtype:  int64
# FLAG_DOCUMENT_3 dtype:  int64
# FLAG_DOCUMENT_4 dtype:  int64
# FLAG_DOCUMENT_5 dtype:  int64
# FLAG_DOCUMENT_6 dtype:  int64
# FLAG_DOCUMENT_7 dtype:  int64
# FLAG_DOCUMENT_8 dtype:  int64
# FLAG_DOCUMENT_9 dtype:  int64
# FLAG_DOCUMENT_10 dtype:  int64
# FLAG_DOCUMENT_11 dtype:  int64
# FLAG_DOCUMENT_12 dtype:  int64
# FLAG_DOCUMENT_13 dtype:  int64
# FLAG_DOCUMENT_14 dtype:  int64
# FLAG_DOCUMENT_15 dtype:  int64
# FLAG_DOCUMENT_16 dtype:  int64
# FLAG_DOCUMENT_17 dtype:  int64
# FLAG_DOCUMENT_18 dtype:  int64
# FLAG_DOCUMENT_19 dtype:  int64
# FLAG_DOCUMENT_20 dtype:  int64
# FLAG_DOCUMENT_21 dtype:  int64
# 
# ### NOT 0/1 - Binary / Dichotomous / Boolean
# 
# Cash loans         278232
# Revolving loans     29279
# Name: NAME_CONTRACT_TYPE, dtype: int64
# 
# N    202924
# Y    104587
# Name: FLAG_OWN_CAR, dtype: int64
# 
# Y    213312
# N     94199
# Name: FLAG_OWN_REALTY, dtype: int64
# 
# No     159428
# NaN    145755
# Yes      2328
# Name: EMERGENCYSTATE_MODE, dtype: int64
# 

# In[ ]:


df['FLAG_MOBIL']value_counts(dropna=False)
df['FLAG_EMP_PHONE']value_counts(dropna=False)
df['FLAG_WORK_PHONE']value_counts(dropna=False)
df['FLAG_CONT_MOBILE']value_counts(dropna=False)
df['FLAG_PHONE']value_counts(dropna=False)
df['FLAG_EMAIL']value_counts(dropna=False)


# In[ ]:


# List of unique values in the df['name'] column
# df.name.unique()
df.TARGET.unique()


# In[ ]:


df['TARGET'].unique()


# In[ ]:


# Number of unique values in the df['name'] column
df.TARGET.nunique()


# In[ ]:


df['TARGET'].nunique()


# In[ ]:


df['TARGET'].value_counts()


# https://stackoverflow.com/questions/45759966/counting-unique-values-in-a-column-in-pandas-dataframe-like-in-qlik
# 
# Count distict values, use nunique:
# df['hID'].nunique()
# Count only non-null values, use count:
# df['hID'].count()
# Count total values including null values, use size attribute:
# df['hID'].size
# To add condition...
# Use boolean indexing:
# df.loc[df['mID']=='A','hID'].agg(['nunique','count','size'])
# Or, using query:
# df.query('mID == "A"')['hID'].agg(['nunique','count','size'])
# 
# New in pandas 0.20.0 pd.DataFrame.agg
# df.agg(['count', 'size', 'nunique'])
# You've always been able to do an agg within a groupby. I used stack at the end because I like the presentation better.
# df.groupby('mID').agg(['count', 'size', 'nunique']).stack()
# 
# 
# https://stackoverflow.com/questions/45125408/how-to-count-the-distinct-values-across-a-column-in-pandas
# 
# df[['Company', 'Date']].drop_duplicates()['Company'].value_counts()
# df.groupby('Company')['Date'].nunique()
# 
# 
# https://stackoverflow.com/questions/48162201/pandas-number-of-unique-values-and-sort-by-the-number-of-unique
# 
# df = df.groupby('A')['B'].nunique().sort_values(ascending=False).reset_index(name='count')
# print (df)
# 
# 
# https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups/38309823
# 
# You can retain the column name like this:
# df = df.groupby(by='domain', as_index=False).agg({'ID': pd.Series.nunique})
# The difference is that 'nunique()' returns a Series and 'agg()' returns a DataFrame.
# 

# In[ ]:


len(df.index)


# In[ ]:


df.shape[0]


# In[ ]:


len(df.columns)


# In[ ]:


df.shape[1]


# In[ ]:


df_row_count, df_column_count = df.shape


# In[ ]:


df_row_count


# In[ ]:


df_column_count


# In[ ]:


dict((column_name,None) for column_name in df_column_names)


# In[ ]:


dict((column_name, df_row_count) for column_name in df_column_names)


# In[ ]:


# https://stackoverflow.com/questions/3869487/how-do-i-create-a-dictionary-with-keys-from-a-list-and-values-defaulting-to-say
# Generator expressions avoid the memory overhead of populating the whole list.
# https://stackoverflow.com/questions/2241891/how-to-initialize-a-dict-with-keys-from-a-list-and-empty-value-in-python
# dict.fromkeys(keys_list)
# Be careful with initializing to something mutable: If you call, e.g., dict.fromkeys([1, 2, 3], []), all of the keys are mapped to the same list, and modifying one will modify them all.
# 
# dict-comprehension solution
# keys = [1,2,3,5,6,7]
# {key: None for key in keys}
#> {1: None, 2: None, 3: None, 5: None, 6: None, 7: None}
# 
# Using a dict-comp also allows the value to be the result of calling a function
#   (which could be passed the key as an argument, if desired)


# In[ ]:


columns_row_count = {column_name:df_row_count for column_name in df_column_names}


# In[ ]:


columns_row_count


# In[ ]:


columns_number_of_unique_values = {column_name:None for column_name in df_column_names}


# In[ ]:


columns_number_of_unique_values


# In[ ]:


for column_name in columns_number_of_unique_values:
    columns_number_of_unique_values[column_name] = df[column_name].nunique()


# In[ ]:


columns_number_of_unique_values


# In[ ]:


columns_cardinality = {column_name:None for column_name in df_column_names}


# In[ ]:


columns_cardinality


# In[ ]:


for column_name in columns_cardinality:
    columns_cardinality[column_name] = columns_number_of_unique_values[column_name]/df_row_count


# In[ ]:


columns_cardinality


# In[ ]:


for column_name in columns_number_of_unique_values:
    if columns_number_of_unique_values[column_name] == 2:
        print (column_name, 'dtype: ', df[column_name].dtype)


# In[ ]:


"""
# http://pbpython.com/pandas-list-dict.html
The “default” manner to create a DataFrame from python is to use a list of dictionaries.
In this case each dictionary key is used for the column headings.
A default index will be created automatically:
sales = [{'account': 'Jones LLC', 'Jan': 150, 'Feb': 200, 'Mar': 140},
         {'account': 'Alpha Co',  'Jan': 200, 'Feb': 210, 'Mar': 215},
         {'account': 'Blue Inc',  'Jan': 50,  'Feb': 90,  'Mar': 95 }]
df = pd.DataFrame(sales)

"""


# In[ ]:


empty_dictionary = {}
empty_dictionary


# In[ ]:


any(empty_dictionary)


# In[ ]:


len(empty_dictionary)


# In[ ]:


bool(empty_dictionary)


# In[ ]:


falsy_dictionary = {0:False}
falsy_dictionary


# In[ ]:


any(falsy_dictionary)


# In[ ]:


len(falsy_dictionary)


# In[ ]:


bool(falsy_dictionary)


# In[ ]:


"""
# https://stackoverflow.com/questions/23177439/python-checking-if-a-dictionary-is-empty-doesnt-seem-to-work
test_dict = {}

# Option 1
if not test_dict:
    print "Dict is Empty"

# Option 2
if not bool(test_dict):
    print "Dict is Empty"

# Option 3
if len(test_dict) == 0:
    print "Dict is Empty"

# The first test in the answer above is true not only if the dict exists and is empty, but also if test_dict is None.
# So use this test only when you know that the dict object exists (or when the difference does not matter).
# The second way also has that behavior.
# Only the third way barks if test_dict is None.
"""


# In[ ]:


# https://stackoverflow.com/questions/18667410/how-can-i-check-if-a-string-only-contains-letters-in-python

if string.isalpha():
    print("It's all letters")

# NOTE: using [A-Za-z ]+ will not match names with non ASCII letterss
# NOTE: using \w includes digits
import re
def only_letters(tested_string):
    #match = re.match("^[ABCDEFGHJKLM]*$", tested_string)
    #match = re.match("^[A-HJ-M]*$", tested_string)
    match = re.match("^[A-Za-z]*$", tested_string)
    return match is not None

import re
def only_letters(string):
    return re.match(r'[a-z\s]+$',string,2) # JWB: What's with the ",2"?

def only_letters(string):
    return all(letter.isalpha() for letter in string)

def only_letters(s):
    for c in s:
        cat = unicodedata.category(c)
        # Lu: Category: Letter, Uppercase (https://codepoints.net/search?gc=Lu)
        # Ll: Category: Letter, Lowercase (https://codepoints.net/search?gc=Ll)
        # Lt: Category: Letter, Titlecase (https://codepoints.net/search?gc=Lt)
        # Lm: Category: Letter, Modifier  (https://codepoints.net/search?gc=Lm)
        # Lo: Category: Letter, Other     (https://codepoints.net/search?gc=Lo)
        # Latin-1: There are no Lm or Lt category codepoints in the Latin-1 subset of Unicode and only 2 Lo characters, ª (U+00AA) and º (U+00BA), Feminine and Masculine Ordinal Indicator).
        if cat not in ('Lu','Ll','Lo'):
            return False
    return True

# https://stackoverflow.com/questions/29460405/checking-if-string-is-only-letters-and-spaces-python
# To require that the string contains only alphas and spaces:
if all(x.isalpha() or x.isspace() for x in string):
    print("Only alphabetical letters and spaces: yes")
else:
    print("Only alphabetical letters and spaces: no")
# To require that the string contains at least one alpha and at least one space:
if any(x.isalpha() for x in string) and any(x.isspace() for x in string):
# To require that the string contains at least one alpha, at least one space, and only alphas and spaces:
if (any(x.isalpha() for x in string)
    and any(x.isspace() for x in string)
    and all(x.isalpha() or x.isspace() for x in string)):


# In[ ]:


#locale.getlocale()
#> NameError: name 'locale' is not defined


# In[ ]:


'dog'.isalpha()


# In[ ]:


'äöå'.isalpha()


# In[ ]:


'привіт'.isalpha()


# In[ ]:


u'привіт'.isalpha()


# In[ ]:


repr('äöå')


# In[ ]:


ascii('äöå')


# In[ ]:


# https://stackoverflow.com/questions/4286637/python-isalpha-and-scandics
# http://en.wikipedia.org/wiki/Windows-1252
s = '\xe4\xf6\xe5'
import unicodedata
for c in s:
    u = c.decode('1252')
    print (ascii(c), ascii(u), unicodedata.name(u, '<no name>'))
#'\xe4' u'\xe4' LATIN SMALL LETTER A WITH DIAERESIS
#'\xf6' u'\xf6' LATIN SMALL LETTER O WITH DIAERESIS
#'\xe5' u'\xe5' LATIN SMALL LETTER A WITH RING ABOVE
s.isalpha()


# In[ ]:


# The isalpha() methods returns “True” if all characters in the string are alphabets, Otherwise, It returns “False”.
# This function is used to check if the argument contains any alphabets characters such as:
#     ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# Both uppercase and lowercase alphabets return “True”
# Space is not considered to be alphabet, therefore it returns “False”

def is_plain_text(string):
    if string != 0:
            # require that the string contains only alphas
            #if all(character.isalpha() for character in string)
            # require that the string contains only alphas and spaces
            if all(character.isalpha() or character.isspace() for character in string):
                return True
            else:
                return False
    else:
            return None


# In[ ]:


is_plain_text(df_as_objects['SK_ID_CURR'])
#TypeError: 'numpy.bool_' object is not iterable


# In[ ]:


df_as_objects['SK_ID_CURR'].dtype


# In[ ]:


df_as_objects['SK_ID_CURR'].str.isdigit().any()


# In[ ]:


df_as_objects['SK_ID_CURR'].str.isdigit().all()


# In[ ]:


df_as_objects['SK_ID_CURR'].str.isalpha().any()


# In[ ]:


df_as_objects['SK_ID_CURR'].str.isalpha().all()


# In[ ]:


df[pd.to_numeric(df.A, errors='coerce').notnull()]


# In[ ]:


df['SK_ID_CURR'].count()


# In[ ]:


df['SK_ID_CURR'].value_counts()


# In[ ]:


# checking for uniqueness:
print(len(df['SK_ID_CURR'].unique()))


# In[ ]:


# How many classes
df['TARGET'].nunique()


# In[ ]:


# Distribution of those classes
df['TARGET'].value_counts(dropna=False)


# In[ ]:


dtypes = df.dtypes
dtypes = dtypes[dtypes != 'object']
features = list(set(dtypes.index) - set(['TARGET']))
len(features)


# In[ ]:


X = df[features]
y = df['TARGET']


# In[ ]:


id_column = ['SK_ID_CURR']


# In[ ]:


target_column = ['TARGET']


# In[ ]:


boolean_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']


# In[ ]:


categorical_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']


# In[ ]:


object_columns = []


# In[ ]:


numerical_columns = []


# In[ ]:


int_columns = []


# In[ ]:


float_columns = []


# In[ ]:


datetime_columns = []


# In[ ]:


date_columns = []


# In[ ]:


nondata_columns = ['SK_ID_CURR']


# In[ ]:


target_distribution = df['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(10, 10),
                             title='Target Distribution',
                             fontsize=15, 
                             legend=True,
                             autopct=lambda v: "{:0.1f}%".format(v))


# In[ ]:


total_nans = df.isnull().sum()
total_nans


# In[ ]:


nan_precents = (df.isnull().sum()/df.isnull().count()*100)
feature_overview_df  = pd.concat([total_nans, nan_precents], axis=1, keys=['NaN Count', 'NaN Pencent'])
feature_overview_df['Type'] = [application_train[c].dtype for c in feature_overview_df.index]
pd.set_option('display.max_rows', None)
display(feature_overview_df)
pd.set_option('display.max_rows', 20)


# In[ ]:


all_application_is_nan_df = pd.DataFrame()
for column in df.columns:
    if application_train[column].isnull().sum() == 0:
        continue
    all_application_is_nan_df['is_nan_' + column] = df[column].isnull()
    all_application_is_nan_df['is_nan_' + column] = all_application_is_nan_df['is_nan_' + column].map(lambda v: 1 if v else 0)
all_application_is_nan_df['target'] = df['TARGET']
all_application_is_nan_df = all_application_is_nan_df[pd.notnull(all_application_is_nan_df['target'])]


# In[ ]:


display(all_application_is_nan_df)


# In[ ]:


Y = all_application_is_nan_df.pop('target')
X = all_application_is_nan_df


# In[ ]:


train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=2018)


# In[ ]:


clf = LGBMClassifier(n_estimators=200, learning_rate=0.01)


# In[ ]:


clf.fit(
        train_X,
        train_Y,
        eval_set=[(train_X, train_Y), (valid_X, valid_Y)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=False
       )


# In[ ]:


plot_importance(clf, figsize=(10,10))


# In[ ]:


#print("only showing the distribution for the first few columns, edit the counter to show all distribution")
#show_feature_count = 10
#for column in all_application_df.columns:
#   if show_feature_count == 0:
#        break
#    show_feature_count -= 1
#    draw_feature_distribution(all_application_df, column)


# In[ ]:


draw_feature_distribution(df, 'TARGET')


# In[ ]:


draw_feature_distribution(df, 'DAYS_EMPLOYED')
# ToDo(JamesBalcomb): fix "ValueError: max() arg is an empty sequence" - add check for 'class_t_values'


# In[ ]:


#EXT_SOURCE_1
#EXT_SOURCE_2
#EXT_SOURCE_3


# In[ ]:


draw_feature_distribution(df, 'EXT_SOURCE_1')


# In[ ]:


draw_feature_distribution(df, 'EXT_SOURCE_2')


# In[ ]:


draw_feature_distribution(df, 'EXT_SOURCE_3')


# # EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
# 
# Q: Is there a relationship between any of these three continuous variables and the binary classification target variable?
# A: Yes, but EXT_SOURCE_2 is oddly shapen.

# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_1', data=df)


# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_2', data=df)


# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_3', data=df)


# In[ ]:


dfapplication_train.hist(column='EXT_SOURCE_1', # Column to plot
              figsize=(8,8),                  # Plot size
              color="blue"                    # Plot color
              )


# In[ ]:


df.hist(column='EXT_SOURCE_2', # Column to plot
              figsize=(8,8),                  # Plot size
              color="blue"                    # Plot color
              )


# In[ ]:


df.hist(column='EXT_SOURCE_3', # Column to plot
              figsize=(8,8),                  # Plot size
              color="blue"                    # Plot color
              )


# In[ ]:


# https://gist.github.com/ltfschoen/4c5d2cf26b8be5355043273493a6b8b9#file-proportions_of_missing_data_in_dataframe_columns-py
def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num / den, 2)


# In[ ]:


get_percentage_missing(df['EXT_SOURCE_1'])


# In[ ]:


get_percentage_missing(df['EXT_SOURCE_2'])


# In[ ]:


get_percentage_missing(df['EXT_SOURCE_3'])


# In[ ]:


# # https://datascience.stackexchange.com/questions/12645/how-to-count-the-number-of-missing-values-in-each-row-in-pandas-dataframe
# # Count of Missing Values per Column
# df.isnull().sum(axis=0)
# # Count of Missing Values per Row
# df.isnull().sum(axis=1)


# In[ ]:


# https://towardsdatascience.com/the-tale-of-missing-values-in-python-c96beb0e8a9d
# If the missing value isn’t identified as NaN , then we have to first convert or replace such non NaN entry with a NaN.
data_name[‘column_name’].replace(0, np.nan, inplace= True)


# In[ ]:


# https://gist.github.com/ltfschoen/4c5d2cf26b8be5355043273493a6b8b9#file-proportions_of_missing_data_in_dataframe_columns-py
df = application_train
# Only include columns that contain any NaN values
df_with_any_null_values = df[df.columns[df.isnull().any()].tolist()]

get_percentage_missing(df_with_any_null_values)

# Iterate over columns in DataFrame and delete those with where >30% of the values are null/NaN
for name, values in df_with_any_null_values.iteritems():
    # print("%r: %r" % (name, values))
    if get_percentage_missing(df_with_any_null_values[name]) > 0.30:
        print("Deleting Column %r: " % (name))
        # df_with_any_null_values.drop(name, axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_1")
ax = sns.distplot(application_traindf["EXT_SOURCE_1"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_2")
ax = sns.distplot(df["EXT_SOURCE_2"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_3")
ax = sns.distplot(df["EXT_SOURCE_3"].dropna())


# In[ ]:


df['EXT_SOURCE_AVG'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_AVG")
ax = sns.distplot(df["EXT_SOURCE_AVG"].dropna())


# In[ ]:


# https://stackoverflow.com/questions/35277075/python-pandas-counting-the-occurrences-of-a-specific-value
#df.loc[df.education == '9th', 'education'].count()
#(df.education == '9th').sum()
#df.query('education == "9th"').education.count()


# In[ ]:


df.loc[df.EXT_SOURCE_1 == 0.0, 'EXT_SOURCE_1'].count()


# In[ ]:


df.loc[df.EXT_SOURCE_2 == 0.0, 'EXT_SOURCE_2'].count()


# In[ ]:


df.loc[df.EXT_SOURCE_3 == 0.0, 'EXT_SOURCE_3'].count()


# In[ ]:


df['EXT_SOURCE_1'].value_counts()


# In[ ]:


df['EXT_SOURCE_2'].value_counts()


# In[ ]:


df['EXT_SOURCE_3'].value_counts()
# df['EXT_SOURCE_3'].value_counts()[:20]
# # ValueError: index must be monotonic increasing or decreasing


# In[ ]:


df['EXT_SOURCE_1'].describe()


# In[ ]:


df['EXT_SOURCE_2'].describe()


# In[ ]:


df['EXT_SOURCE_3'].describe()


# In[ ]:


df['EXT_SOURCE_1'].plot(kind='hist', bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


# In[ ]:


df['EXT_SOURCE_1'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


df['EXT_SOURCE_2'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


df['EXT_SOURCE_3'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


# https://community.modeanalytics.com/python/tutorial/python-histograms-boxplots-and-distributions/
bin_values = np.arange(start=0, stop=1, step=0.2)
us_mq_airlines_index = df['TARGET'].isin(['US','MQ']) # create index
us_mq_airlines = df[us_mq_airlines_index] # select rows
group_carriers = us_mq_airlines.groupby('TARGET')['EXT_SOURCE_1'] # group values
group_carriers.plot(kind='hist', bins=bin_values, figsize=[12,6], alpha=.4, legend=True) # alpha for transparency


# In[ ]:


df['EXT_SOURCE_1'].plot(kind='box', figsize=[16,8])


# In[ ]:


df['EXT_SOURCE_2'].plot(kind='box', figsize=[16,8])


# In[ ]:


df['EXT_SOURCE_3'].plot(kind='box', figsize=[16,8])


# In[ ]:


# Define a function to create the scatterplot. This makes it easy to
# reuse code within and across notebooks
def scatterplot(x_data, y_data, x_label, y_label, title):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha) of the points
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# In[ ]:


## Call the function to create plot
#scatterplot(x_data = daily_data['temp']
#            , y_data = daily_data['cnt']
#            , x_label = 'Normalized temperature (C)'
#            , y_label = 'Check outs'
#            , title = 'Number of Check Outs vs Temperature')


# In[ ]:


# Call the function to create plot
scatterplot(x_data = df['EXT_SOURCE_1']
            , y_data = df['EXT_SOURCE_2']
            , x_label = 'EXT_SOURCE_1'
            , y_label = 'EXT_SOURCE_2'
            , title = 'EXT_SOURCE_1 vs. EXT_SOURCE_2'
           )


# In[ ]:


# Call the function to create plot
scatterplot(x_data = df['EXT_SOURCE_1']
            , y_data = df['EXT_SOURCE_3']
            , x_label = 'EXT_SOURCE_1'
            , y_label = 'EXT_SOURCE_2'
            , title = 'EXT_SOURCE_1 vs. EXT_SOURCE_3'
           )


# In[ ]:


# Call the function to create plot
scatterplot(x_data = df['EXT_SOURCE_2']
            , y_data = df['EXT_SOURCE_3']
            , x_label = 'EXT_SOURCE_2'
            , y_label = 'EXT_SOURCE_3'
            , title = 'EXT_SOURCE_2 vs. EXT_SOURCE_3'
           )


# In[ ]:


df_ext_source_1__dropna = df.loc[:,['EXT_SOURCE_1','TARGET']]
df_ext_source_1__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(df_ext_source_1__dropna['EXT_SOURCE_1'], df_ext_source_1__dropna['TARGET'])


# In[ ]:


df_ext_source_2__dropna = application_train.loc[:,['EXT_SOURCE_2','TARGET']]
df_ext_source_2__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(df_ext_source_2__dropna['EXT_SOURCE_2'], df_ext_source_2__dropna['TARGET'])


# In[ ]:


df_ext_source_3__dropna = df.loc[:,['EXT_SOURCE_3','TARGET']]
df_ext_source_3__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(df_ext_source_3__dropna['EXT_SOURCE_3'], df_ext_source_3__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(df_ext_source_1__dropna['EXT_SOURCE_1'], df_ext_source_1__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(df_ext_source_2__dropna['EXT_SOURCE_2'], df_ext_source_2__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(df_ext_source_3__dropna['EXT_SOURCE_3'], df_ext_source_3__dropna['TARGET'])


# In[ ]:


np.corrcoef(df_ext_source_1__dropna['EXT_SOURCE_1'], df_ext_source_1__dropna['TARGET'])


# In[ ]:


np.corrcoef(df_ext_source_2__dropna['EXT_SOURCE_2'], df_ext_source_2__dropna['TARGET'])


# In[ ]:


np.corrcoef(df_ext_source_3__dropna['EXT_SOURCE_3'], df_ext_source_3__dropna['TARGET'])


# In[ ]:


temp = previous_application["NAME_CONTRACT_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Contract product type of previous application",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Contract product type",
                "x": 0.12,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# # Pearson Correlation of features

# In[ ]:


data = [
    go.Heatmap(
        z=df.corr().values,
        x=df.columns.values,
        y=df.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
margin=dict(
    l=240,
),)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# # Multicollinearity
# https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels/44012251#44012251
# https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels/25833792#25833792
# https://onlinecourses.science.psu.edu/stat501/node/347/
# 

# In[ ]:


## https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels/44012251#44012251
##...looking for a single number that captured the collinearity
##...options include the determinant and condition number of the correlation matrix
##...determinant of the correlation matrix will "range from 0 (Perfect Collinearity) to 1 (No Collinearity)"


# In[ ]:


# Compute correlation matrices
pearson_product_moment_correlation_coefficients = np.corrcoef(df, rowvar=0)
## https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
## Return Pearson product-moment correlation coefficients.


# In[ ]:


# Compare the determinants
print np.linalg.det(pearson_product_moment_correlation_coefficients)


# In[ ]:


# the condition number of the covariance matrix will approach infinity with perfect linear dependence
print np.linalg.cond(pearson_product_moment_correlation_coefficients)


# In[ ]:


df.info()


# In[ ]:


df.info(memory_usage='deep')


# In[ ]:


df.memory_usage()


# In[ ]:


df.memory_usage(index=False)


# In[ ]:


# https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
# Group loans by client id and calculate mean, max, min of loans
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']

# Merge with the clients dataframe
stats = clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left')

stats.head(10)


# In[ ]:


#F      202448
#M      105059
#XNA         4
#Name: CODE_GENDER, dtype: int64
202448 / (202448 + 105059), 105059 / (202448 + 105059)


# In[ ]:


#F      202448
#M      105059
#XNA         4
#Name: CODE_GENDER, dtype: int64
(4 * 0.658352492788782), (4 * 0.34164750721121795)


# In[ ]:


# Female: 202448 (0.6584) Male: 105059 (0.3417)


# In[ ]:


# application_train[application_train['CODE_GENDER'] == "XNA"]
#        SK_ID_CURR
#  35657     141289
#  38566     144669
#  83382     196708
# 189640     319880
temporary_list = df[application_train['CODE_GENDER'] == "XNA"]['SK_ID_CURR'].tolist()
temporary_list
#> [141289, 144669, 196708, 319880]


# In[ ]:


"""
# https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch02s09.html
# But the winner is the version that appears to be the simplest:
def best(  ):
    random.shuffle(data)
    for elem in data: process(elem)
# Or, if you need to preserve the data list's original ordering:
def best_preserve(  ):
    aux = list(data)
    random.shuffle(aux)
    for elem in aux: process(elem)
"""


# In[ ]:


"""
# https://www.pythoncentral.io/select-random-item-list-tuple-data-structure-python/
# One of the most common tasks that requires random action is selecting one item from a group, be it a character from a string, unicode, or buffer, a byte from a bytearray, or an item from a list, tuple, set, or xrange.
# It's also common to want a sample of more than one item.
# The pythonic way to select a single item from a Python sequence type — that's any of str, unicode, list, tuple, bytearray, buffer, xrange — is to use random.choice.
# For example, the last line of our single-item selection would be:
rand_item = random.choice(items)
# Much simpler, isn't it? There's an equally simple way to select n items from the sequence:
rand_items = random.sample(items, n)
"""


# In[ ]:


#set seed = 1234567890
random_item = random.choice(temporary_list)
random_item
#> 196708


# In[ ]:


random_items = random.sample(temporary_list, 3)
random_items
#> [141289, 319880, 196708]


# In[ ]:


# Python 3.6+
from random import choices
random_items = choices(temporary_list, k=3)
random_items
#> [141289, 144669, 141289]


# In[ ]:


# CODE_GENDER: XNA ==> CODE_GENDER: F
#application_train[application_train['SK_ID_CURR'] == 141289]
df[application_train['SK_ID_CURR'] == 141289]['CODE_GENDER']


# In[ ]:


df.loc[df.SK_ID_CURR == 141289, 'CODE_GENDER'] = 'F'


# In[ ]:


df[df['SK_ID_CURR'] == 141289]['CODE_GENDER']


# In[ ]:


# CODE_GENDER: XNA ==> CODE_GENDER: F
df[df['SK_ID_CURR'] == 319880]['CODE_GENDER']


# In[ ]:


df.loc[df.SK_ID_CURR == 319880, 'CODE_GENDER'] = 'F'


# In[ ]:


df[df['SK_ID_CURR'] == 319880]['CODE_GENDER']


# In[ ]:


# CODE_GENDER: XNA ==> CODE_GENDER: F
df[df['SK_ID_CURR'] == 196708]['CODE_GENDER']


# In[ ]:


df.loc[df.SK_ID_CURR == 196708, 'CODE_GENDER'] = 'F'


# In[ ]:


df[df['SK_ID_CURR'] == 196708]['CODE_GENDER']


# In[ ]:


# CODE_GENDER: XNA ==> CODE_GENDER: M
df[df['SK_ID_CURR'] == 144669]['CODE_GENDER']


# In[ ]:


df.loc[df.SK_ID_CURR == 144669, 'CODE_GENDER'] = 'M'


# In[ ]:


df[df['SK_ID_CURR'] == 144669]['CODE_GENDER']


# In[ ]:


df.loc[df.SK_ID_CURR == 141289, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 319880, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 196708, 'CODE_GENDER'] = 'F'
df.loc[df.SK_ID_CURR == 144669, 'CODE_GENDER'] = 'M'


# In[ ]:


# Using module time ()
import time
ts = time.time()  # number of seconds since the epoch
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required
ts_human_readable = time.ctime(1529007180)
print(ts_human_readable)
# Using module datetime
import datetime;
ts = datetime.datetime.now().timestamp()  # number of seconds since the epoch
print(ts)
#print(time.strftime("%Y-%m-%d %H:%M:%S", ts))  # TypeError: Tuple or struct_time argument required
ts_human_readable = datetime.datetime.fromtimestamp(1529007180).isoformat()
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


# In[ ]:


df['ORGANIZATION_TYPE'].value_counts()


# In[ ]:


df['OrganizationTypeGroup'] = ''


# In[ ]:


df.loc[df.ORGANIZATION_TYPE == 'Advertising', 'OrganizationTypeGroup'] = 'Advertising'


# In[ ]:


df['OrganizationTypeGroup'].value_counts()


# In[ ]:


df.drop('OrganizationTypeGroup', axis=1, inplace=True)


# In[ ]:


dict_organization_type_groups = {
'Advertising':'Advertising',
'Agriculture':'Agriculture',
'Bank':'Bank',
'Business_Entity':'Business Entity Type 1',
'Business_Entity':'Business Entity Type 2',
'Business_Entity':'Business Entity Type 3',
'Cleaning':'Cleaning',
'Construction':'Construction',
'Culture':'Culture',
'Electriciy':'Electricity',
'Emergency':'Emergency',
'Government':'Government',
'Hotel':'Hotel',
'Housing':'Housing',
'Industry':'Industry: type 1',
'Industry':'Industry: type 2',
'Industry':'Industry: type 3',
'Industry':'Industry: type 4',
'Industry':'Industry: type 5',
'Industry':'Industry: type 6',
'Industry':'Industry: type 7',
'Industry':'Industry: type 8',
'Industry':'Industry: type 9',
'Industry':'Industry: type 10',
'Industry':'Industry: type 11',
'Industry':'Industry: type 12',
'Industry':'Industry: type 13',
'Insurance':'Insurance',
'Kindergarten':'Kindergarten',
'Legal_Services':'Legal Services',
'Medicine':'Medicine',
'Military':'Military',
'Mobile':'Mobile',
'Other':'Other',
'Police':'Police',
'Postal':'Postal',
'Realtor':'Realtor',
'Religion':'Religion',
'Restaurant':'Restaurant',
'School':'School',
'Security':'Security',
'Security':'Security Ministries',
'Self_Employed':'Self-employed',
'Services':'Services',
'Telecom':'Telecom',
'Trade':'Trade: type 1',
'Trade':'Trade: type 2',
'Trade':'Trade: type 3',
'Trade':'Trade: type 4',
'Trade':'Trade: type 5',
'Trade':'Trade: type 6',
'Trade':'Trade: type 7',
'Transport':'Transport: type 1',
'Transport':'Transport: type 2',
'Transport':'Transport: type 3',
'Transport':'Transport: type 4',
'University':'University',
'XNA':'XNA'
}
dict_organization_type_groups


# In[ ]:


# df_organization_type_groups = pd.DataFrame(dict_organization_type_groups)
#> ValueError: If using all scalar values, you must pass an index
#df_organization_type_groups = pd.Series(dict_organization_type_groups, name='OrganizationType')
#df_organization_type_groups.index.name = 'OrganizationTypeGroup'
#df_organization_type_groups.reset_index()
#df_organization_type_groups
df_organization_type_groups = pd.DataFrame(list(dict_organization_type_groups.items()), columns=['OrganizationTypeGroup', 'ORGANIZATION_TYPE'])  # Python 3
df_organization_type_groups


# In[ ]:


#application_train['OrganizationTypeGroup'] = application_train['OrganizationTypeGroup'].applymap(organization_type_groups.get)
# pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
#         left_index=False, right_index=False, sort=True,
#         suffixes=('_x', '_y'), copy=True, indicator=False,
#         validate=None)
df = pd.merge(application_train,
                             df_organization_type_groups,
                             how='right',
                             left_on='ORGANIZATION_TYPE',
                             right_on='ORGANIZATION_TYPE')
df['OrganizationTypeGroup'].value_counts()


# # FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL  
# FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL  
# FLAG_MOBIL        Did client provide mobile phone (1=YES, 0=NO)  
# FLAG_EMP_PHONE    Did client provide work phone (1=YES, 0=NO)  
# FLAG_WORK_PHONE   Did client provide home phone (1=YES, 0=NO)  
# FLAG_CONT_MOBILE  Was mobile phone reachable (1=YES, 0=NO)  
# FLAG_PHONE        Did client provide home phone (1=YES, 0=NO)  
# FLAG_EMAIL        Did client provide email (1=YES, 0=NO)  

# In[ ]:


df['FLAG_MOBIL'].value_counts()


# In[ ]:


df['FLAG_EMP_PHONE'].value_counts()


# In[ ]:


df['FLAG_WORK_PHONE'].value_counts()


# In[ ]:


df['FLAG_CONT_MOBILE'].value_counts()


# In[ ]:


df['FLAG_PHONE'].value_counts()


# In[ ]:


df['FLAG_EMAIL'].value_counts()


# # DAYS_LAST_PHONE_CHANGE

# In[ ]:


df['DAYS_LAST_PHONE_CHANGE'].value_counts()


# # FLAG_OWN_REALTY, FLAG_OWN_CAR, OWN_CAR_AGE, NAME_HOUSING_TYPE

# In[ ]:


df['FLAG_OWN_REALTY'].value_counts()


# In[ ]:


df['FLAG_OWN_CAR'].value_counts()


# In[ ]:


df['OWN_CAR_AGE'].value_counts()


# In[ ]:


df['NAME_HOUSING_TYPE'].value_counts()


# In[ ]:


df['NAME_CONTRACT_TYPE'].value_counts()


# In[ ]:


# len(application_train.index)
# application_train.shape[0]
# len(application_train.columns)
# application_train.shape[1]
df_row_count, df_column_count = df.shape
df_row_count, df_column_count


# In[ ]:


df_column_names = df.columns.tolist()
df_column_names


# In[ ]:


df_columns_row_count = {column_name:df_row_count for column_name in df_column_names}
df_columns_row_count


# In[ ]:


# Cash loans
278232 / df_row_count
#> 0.9047871458256778


# In[ ]:


# Revolving loans
29279 / df_row_count
#> 0.09521285417432222

