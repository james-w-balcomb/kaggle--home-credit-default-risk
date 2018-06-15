
# coding: utf-8

# In[ ]:


#%run NB01-Load.ipynb


# In[1]:


import collections
import gc
import IPython
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import random 
#import seaborn as sb
import seaborn as sns
#import seaborn as snss
import scipy
import scipy as sp
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import warnings

from datetime import date
from IPython.display import HTML
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from pandas import DataFrame
from pandas import Series
from patsy import dmatrices
from plotly import tools
from plotly.offline import init_notebook_mode
from plotly.offline import iplot
from pylab import rcParams
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from wordcloud import WordCloud

init_notebook_mode(connected=True)
plt.rcParams.update({'figure.max_open_warning': 200})
# Suppress warnings
warnings.filterwarnings("ignore")

# In a notebook environment, display the plots inline
get_ipython().magic('matplotlib inline')
# Set some parameters to apply to all plots. These can be overridden in each plot if desired
# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

np.set_printoptions(threshold=np.nan)

#import C:/Development/kaggle--home-credit-default-risk/rand_jitter
#import C:/Development/kaggle--home-credit-default-risk/draw_feature_distribution
import sys
# sys.path.insert(0, 'C:/Development/kaggle--home-credit-default-risk/') # ~= sys.path.prepend
sys.path.append('C:/Development/kaggle--home-credit-default-risk/')
# import rand_jitter
# import draw_feature_distribution
##from rand_jitter import * # NOTE: added directly to draw_feature_distribution_v2
# from draw_feature_distribution import *
# from draw_feature_distribution_v1 import *
from draw_feature_distribution_v2 import *

# C:\Users\jbalcomb\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning:
# The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.


# In[2]:


print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))
print("SciPy version: {}".format(sp.__version__))
print("scikit-learn version: {}".format(sklearn.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("IPython version: {}".format(IPython.__version__))


# In[3]:


path = 'C:/Development/kaggle--home-credit-default-risk/data/'


# In[11]:


application_train = pd.read_table(path + 'application_train.csv', sep=',')


# In[ ]:


#application_train = application_train.fillna(np.nan)


# In[ ]:


#df = application_train


# In[9]:


application_train.shape


# In[5]:


application_train.head()


# In[12]:


application_train.describe()


# In[13]:


application_train_column_names = application_train.columns.tolist()
application_train_column_names


# In[8]:


application_train.columns.values


# In[14]:


application_train__dtypes = application_train.dtypes


# In[15]:


application_train__dtypes.to_csv('application_train__dtypes.csv')


# In[16]:


application_train__dtypes


# In[17]:


application_train.columns.to_series().groupby(application_train.dtypes).groups


# In[18]:


application_train.select_dtypes(include=['bool'])


# In[19]:


list(application_train.select_dtypes(include=['bool']).columns)


# In[20]:


list(application_train.select_dtypes(include=['int64']).columns)


# In[21]:


list(application_train.select_dtypes(include=['float64']).columns)


# In[22]:


list(application_train.select_dtypes(include=['O']).columns)


# In[42]:


application_train_as_objects = pd.read_table(path + 'application_train.csv', sep=',', dtype=object)


# In[ ]:


application_train_column_names


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


# In[25]:


#locale.getlocale()
#> NameError: name 'locale' is not defined


# In[26]:


'dog'.isalpha()


# In[27]:


'äöå'.isalpha()


# In[28]:


'привіт'.isalpha()


# In[29]:


u'привіт'.isalpha()


# In[30]:


repr('äöå')


# In[32]:


ascii('äöå')


# In[35]:


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


# In[37]:


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


# In[40]:


is_plain_text(application_train_as_objects['SK_ID_CURR'])
#TypeError: 'numpy.bool_' object is not iterable


# In[46]:


application_train_as_objects['SK_ID_CURR'].dtype


# In[49]:


application_train_as_objects['SK_ID_CURR'].str.isdigit().any()


# In[50]:


application_train_as_objects['SK_ID_CURR'].str.isdigit().all()


# In[51]:


application_train_as_objects['SK_ID_CURR'].str.isalpha().any()


# In[52]:


application_train_as_objects['SK_ID_CURR'].str.isalpha().all()


# In[ ]:


df[pd.to_numeric(df.A, errors='coerce').notnull()]


# In[ ]:


application_train['SK_ID_CURR'].count()


# In[ ]:


application_train['SK_ID_CURR'].value_counts()


# In[ ]:


# checking for uniqueness:
print(len(application_train['SK_ID_CURR'].unique()))


# In[ ]:


# How many classes
application_train['TARGET'].nunique()


# In[ ]:


# Distribution of those classes
application_train['TARGET'].value_counts(dropna=False)


# In[ ]:


dtypes = application_train.dtypes
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


target_distribution = application_train['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(10, 10),
                             title='Target Distribution',
                             fontsize=15, 
                             legend=True, 
                             autopct=lambda v: "{:0.1f}%".format(v))


# In[ ]:


total_nans = application_train.isnull().sum()
nan_precents = (application_train.isnull().sum()/application_train.isnull().count()*100)
feature_overview_df  = pd.concat([total_nans, nan_precents], axis=1, keys=['NaN Count', 'NaN Pencent'])
feature_overview_df['Type'] = [application_train[c].dtype for c in feature_overview_df.index]
pd.set_option('display.max_rows', None)
display(feature_overview_df)
pd.set_option('display.max_rows', 20)


# In[ ]:


all_application_is_nan_df = pd.DataFrame()
for column in application_train.columns:
    if application_train[column].isnull().sum() == 0:
        continue
    all_application_is_nan_df['is_nan_' + column] = application_train[column].isnull()
    all_application_is_nan_df['is_nan_' + column] = all_application_is_nan_df['is_nan_' + column].map(lambda v: 1 if v else 0)
all_application_is_nan_df['target'] = application_train['TARGET']
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


draw_feature_distribution(application_train, 'TARGET')


# In[ ]:


draw_feature_distribution(application_train, 'DAYS_EMPLOYED')
# ToDo(JamesBalcomb): fix "ValueError: max() arg is an empty sequence" - add check for 'class_t_values'


# In[ ]:


#EXT_SOURCE_1
#EXT_SOURCE_2
#EXT_SOURCE_3


# In[ ]:


draw_feature_distribution(application_train, 'EXT_SOURCE_1')


# In[ ]:


draw_feature_distribution(application_train, 'EXT_SOURCE_2')


# In[ ]:


draw_feature_distribution(application_train, 'EXT_SOURCE_3')


# # EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
# 
# Q: Is there a relationship between any of these three continuous variables and the binary classification target variable?
# A: Yes, but EXT_SOURCE_2 is oddly shapen.

# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_1', data=application_train)


# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_2', data=application_train)


# In[ ]:


# Seaborn Violin Plot - correlation; distribution and density
sns.violinplot(x='TARGET', y='EXT_SOURCE_3', data=application_train)


# In[ ]:


application_train.hist(column='EXT_SOURCE_1', # Column to plot
              figsize=(8,8),                  # Plot size
              color="blue"                    # Plot color
              )


# In[ ]:


application_train.hist(column='EXT_SOURCE_2', # Column to plot
              figsize=(8,8),                  # Plot size
              color="blue"                    # Plot color
              )


# In[ ]:


application_train.hist(column='EXT_SOURCE_3', # Column to plot
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


get_percentage_missing(application_train['EXT_SOURCE_1'])


# In[ ]:


get_percentage_missing(application_train['EXT_SOURCE_2'])


# In[ ]:


get_percentage_missing(application_train['EXT_SOURCE_3'])


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
ax = sns.distplot(application_train["EXT_SOURCE_1"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_2")
ax = sns.distplot(application_train["EXT_SOURCE_2"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_3")
ax = sns.distplot(application_train["EXT_SOURCE_3"].dropna())


# In[ ]:


application_train['EXT_SOURCE_AVG'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of EXT_SOURCE_AVG")
ax = sns.distplot(application_train["EXT_SOURCE_AVG"].dropna())


# In[ ]:


# https://stackoverflow.com/questions/35277075/python-pandas-counting-the-occurrences-of-a-specific-value
#df.loc[df.education == '9th', 'education'].count()
#(df.education == '9th').sum()
#df.query('education == "9th"').education.count()


# In[ ]:


application_train.loc[application_train.EXT_SOURCE_1 == 0.0, 'EXT_SOURCE_1'].count()


# In[ ]:


application_train.loc[application_train.EXT_SOURCE_2 == 0.0, 'EXT_SOURCE_2'].count()


# In[ ]:


application_train.loc[application_train.EXT_SOURCE_3 == 0.0, 'EXT_SOURCE_3'].count()


# In[ ]:


application_train['EXT_SOURCE_1'].value_counts()


# In[ ]:


application_train['EXT_SOURCE_2'].value_counts()


# In[ ]:


application_train['EXT_SOURCE_3'].value_counts()
# application_train['EXT_SOURCE_3'].value_counts()[:20]
# # ValueError: index must be monotonic increasing or decreasing


# In[ ]:


application_train['EXT_SOURCE_1'].describe()


# In[ ]:


application_train['EXT_SOURCE_2'].describe()


# In[ ]:


application_train['EXT_SOURCE_3'].describe()


# In[ ]:


application_train['EXT_SOURCE_1'].plot(kind='hist', bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


# In[ ]:


application_train['EXT_SOURCE_1'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


application_train['EXT_SOURCE_2'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


application_train['EXT_SOURCE_3'].plot(kind='hist', bins=[0.0,0.2,0.4,0.6,0.8,1.0])


# In[ ]:


# https://community.modeanalytics.com/python/tutorial/python-histograms-boxplots-and-distributions/
bin_values = np.arange(start=0, stop=1, step=0.2)
us_mq_airlines_index = application_train['TARGET'].isin(['US','MQ']) # create index
us_mq_airlines = application_train[us_mq_airlines_index] # select rows
group_carriers = us_mq_airlines.groupby('TARGET')['EXT_SOURCE_1'] # group values
group_carriers.plot(kind='hist', bins=bin_values, figsize=[12,6], alpha=.4, legend=True) # alpha for transparency


# In[ ]:


application_train['EXT_SOURCE_1'].plot(kind='box', figsize=[16,8])


# In[ ]:


application_train['EXT_SOURCE_2'].plot(kind='box', figsize=[16,8])


# In[ ]:


application_train['EXT_SOURCE_3'].plot(kind='box', figsize=[16,8])


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
scatterplot(x_data = application_train['EXT_SOURCE_1']
            , y_data = application_train['EXT_SOURCE_2']
            , x_label = 'EXT_SOURCE_1'
            , y_label = 'EXT_SOURCE_2'
            , title = 'EXT_SOURCE_1 vs. EXT_SOURCE_2'
           )


# In[ ]:


# Call the function to create plot
scatterplot(x_data = application_train['EXT_SOURCE_1']
            , y_data = application_train['EXT_SOURCE_3']
            , x_label = 'EXT_SOURCE_1'
            , y_label = 'EXT_SOURCE_2'
            , title = 'EXT_SOURCE_1 vs. EXT_SOURCE_3'
           )


# In[ ]:


# Call the function to create plot
scatterplot(x_data = application_train['EXT_SOURCE_2']
            , y_data = application_train['EXT_SOURCE_3']
            , x_label = 'EXT_SOURCE_2'
            , y_label = 'EXT_SOURCE_3'
            , title = 'EXT_SOURCE_2 vs. EXT_SOURCE_3'
           )


# In[ ]:


application_train__ext_source_1__dropna = application_train.loc[:,['EXT_SOURCE_1','TARGET']]
application_train__ext_source_1__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(application_train__ext_source_1__dropna['EXT_SOURCE_1'], application_train__ext_source_1__dropna['TARGET'])


# In[ ]:


application_train__ext_source_2__dropna = application_train.loc[:,['EXT_SOURCE_2','TARGET']]
application_train__ext_source_2__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(application_train__ext_source_2__dropna['EXT_SOURCE_2'], application_train__ext_source_2__dropna['TARGET'])


# In[ ]:


application_train__ext_source_3__dropna = application_train.loc[:,['EXT_SOURCE_3','TARGET']]
application_train__ext_source_3__dropna.dropna(inplace = True)
scipy.stats.pointbiserialr(application_train__ext_source_3__dropna['EXT_SOURCE_3'], application_train__ext_source_3__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(application_train__ext_source_1__dropna['EXT_SOURCE_1'], application_train__ext_source_1__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(application_train__ext_source_2__dropna['EXT_SOURCE_2'], application_train__ext_source_2__dropna['TARGET'])


# In[ ]:


scipy.stats.pearsonr(application_train__ext_source_3__dropna['EXT_SOURCE_3'], application_train__ext_source_3__dropna['TARGET'])


# In[ ]:


np.corrcoef(application_train__ext_source_1__dropna['EXT_SOURCE_1'], application_train__ext_source_1__dropna['TARGET'])


# In[ ]:


np.corrcoef(application_train__ext_source_2__dropna['EXT_SOURCE_2'], application_train__ext_source_2__dropna['TARGET'])


# In[ ]:


np.corrcoef(application_train__ext_source_3__dropna['EXT_SOURCE_3'], application_train__ext_source_3__dropna['TARGET'])


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
        z=application_train.corr().values,
        x=application_train.columns.values,
        y=application_train.columns.values,
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
pearson_product_moment_correlation_coefficients = np.corrcoef(application_train, rowvar=0)
## https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
## Return Pearson product-moment correlation coefficients.


# In[ ]:


# Compare the determinants
print np.linalg.det(pearson_product_moment_correlation_coefficients)


# In[ ]:


# the condition number of the covariance matrix will approach infinity with perfect linear dependence
print np.linalg.cond(pearson_product_moment_correlation_coefficients)


# In[ ]:


application_train.info()


# In[ ]:


application_train.info(memory_usage='deep')


# In[ ]:


application_train.memory_usage()


# In[ ]:


application_train.memory_usage(index=False)


# In[ ]:


# https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219
# Group loans by client id and calculate mean, max, min of loans
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']

# Merge with the clients dataframe
stats = clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left')

stats.head(10)

