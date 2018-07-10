
# coding: utf-8

# In[1]:


import collections
import gc
import IPython
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import plotly.graph_objs as go
import plotly.plotly as py
import random 
#import seaborn as sb
import seaborn as sns
#import seaborn as snss
#import scipy
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
from random import choice
from random import choices # Python 3.6+
from random import sample
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
# Set some parameters to apply to all plots. These can be overridden in each plot if desired.
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
# emulate the aesthetics of ggplot (a popular plotting package for R)
matplotlib.style.use('ggplot')

np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

#import C:/Development/kaggle--home-credit-default-risk/rand_jitter
#import C:/Development/kaggle--home-credit-default-risk/draw_feature_distribution

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


random_seed = 1234567890
random.seed(random_seed)
np.random.seed(random_seed)


# In[4]:


# Set configuration based on environment variables
if os.getenv('HCDR_DATA_FILE_PATH'):
    print('Using Environment Variable for data_file_path')
    data_file_path = os.getenv('HCDR_DATA_FILE_PATH')
    data_file_path = os.path.join(data_file_path, '')
#TODO(JamesBalcomb): add code to fall back on .config file
#else:
#    'kaggle--home-credit-default-risk.config'
else:
    print('Using Hard-Coded Configuration for data_file_path')
    data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
    data_file_path = os.path.join(data_file_path, '')
print('data_file_path: ', data_file_path)


# In[5]:


# Set configuration based on environment variables
if os.getenv('HCDR_WORKING_DIRECTORY'):
    print('Using Environment Variable for working_directory')
    working_directory = os.getenv('HCDR_WORKING_DIRECTORY')
    working_directory = os.path.join(working_directory, '')
#TODO(JamesBalcomb): add code to fall back on .config file
#else:
#    'kaggle--home-credit-default-risk.config'
else:
    print('Using Hard-Coded Configuration for working_directory')
    working_directory = 'C:/Development/kaggle--home-credit-default-risk/'
    working_directory = os.path.join(working_directory, '')
print('working_directory: ', working_directory)


# In[6]:


data_file_names = [
    'application_test.csv',
    'application_train.csv',
    'bureau.csv',
    'bureau_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'POS_CASH_balance.csv',
    'previous_application.csv'
]
data_file_names


# In[7]:


df_dataframe_names_and_files = [
    ['df1','application_train.csv'],
    ['df2','application_test.csv'],
    ['df3','bureau.csv'],
    ['df4','bureau_balance.csv'],
    ['df5','credit_card_balance.csv'],
    ['df6','installments_payments.csv'],
    ['df7','POS_CASH_balance.csv'],
    ['df8','previous_application.csv']
]
df_dataframe_names_and_files


# In[8]:


df_dataset_names_and_files = [
    ['ds1','application_train.csv'],
    ['ds2','application_test.csv'],
    ['ds3','bureau.csv'],
    ['ds4','bureau_balance.csv'],
    ['ds5','credit_card_balance.csv'],
    ['ds6','installments_payments.csv'],
    ['ds7','POS_CASH_balance.csv'],
    ['ds8','previous_application.csv']
]
df_dataset_names_and_files


# In[9]:


# TODO(JamesBalcomb): add a function that handles specifying multiple files
data_file_name = 'application_train.csv'
print('data_file_name: ', data_file_name)


# In[10]:


df = pd.read_table(
        data_file_path + data_file_name,
        sep=',',
        dtype={
                'SK_ID_CURR':'object',
                'TARGET':'object',
                'NAME_CONTRACT_TYPE':'object',
                'CODE_GENDER':'object',
                'FLAG_OWN_CAR':'object',
                'FLAG_OWN_REALTY':'object',
                'CNT_CHILDREN':'float64',
                'AMT_INCOME_TOTAL':'float64',
                'AMT_CREDIT':'float64',
                'AMT_ANNUITY':'float64',
                'AMT_GOODS_PRICE':'float64',
                'NAME_TYPE_SUITE':'object',
                'NAME_INCOME_TYPE':'object',
                'NAME_EDUCATION_TYPE':'object',
                'NAME_FAMILY_STATUS':'object',
                'NAME_HOUSING_TYPE':'object',
                'REGION_POPULATION_RELATIVE':'float64',
                'DAYS_BIRTH':'float64',
                'DAYS_EMPLOYED':'float64',
                'DAYS_REGISTRATION':'float64',
                'DAYS_ID_PUBLISH':'float64',
                'OWN_CAR_AGE':'float64',
                'FLAG_MOBIL':'object',
                'FLAG_EMP_PHONE':'object',
                'FLAG_WORK_PHONE':'object',
                'FLAG_CONT_MOBILE':'object',
                'FLAG_PHONE':'object',
                'FLAG_EMAIL':'object',
                'OCCUPATION_TYPE':'object',
                'CNT_FAM_MEMBERS':'float64',
                'REGION_RATING_CLIENT':'object',
                'REGION_RATING_CLIENT_W_CITY':'object',
                'WEEKDAY_APPR_PROCESS_START':'object',
                'HOUR_APPR_PROCESS_START':'float64',
                'REG_REGION_NOT_LIVE_REGION':'object',
                'REG_REGION_NOT_WORK_REGION':'object',
                'LIVE_REGION_NOT_WORK_REGION':'object',
                'REG_CITY_NOT_LIVE_CITY':'object',
                'REG_CITY_NOT_WORK_CITY':'object',
                'LIVE_CITY_NOT_WORK_CITY':'object',
                'ORGANIZATION_TYPE':'object',
                'EXT_SOURCE_1':'float64',
                'EXT_SOURCE_2':'float64',
                'EXT_SOURCE_3':'float64',
                'APARTMENTS_AVG':'float64',
                'BASEMENTAREA_AVG':'float64',
                'YEARS_BEGINEXPLUATATION_AVG':'float64',
                'YEARS_BUILD_AVG':'float64',
                'COMMONAREA_AVG':'float64',
                'ELEVATORS_AVG':'float64',
                'ENTRANCES_AVG':'float64',
                'FLOORSMAX_AVG':'float64',
                'FLOORSMIN_AVG':'float64',
                'LANDAREA_AVG':'float64',
                'LIVINGAPARTMENTS_AVG':'float64',
                'LIVINGAREA_AVG':'float64',
                'NONLIVINGAPARTMENTS_AVG':'float64',
                'NONLIVINGAREA_AVG':'float64',
                'APARTMENTS_MODE':'float64',
                'BASEMENTAREA_MODE':'float64',
                'YEARS_BEGINEXPLUATATION_MODE':'float64',
                'YEARS_BUILD_MODE':'float64',
                'COMMONAREA_MODE':'float64',
                'ELEVATORS_MODE':'float64',
                'ENTRANCES_MODE':'float64',
                'FLOORSMAX_MODE':'float64',
                'FLOORSMIN_MODE':'float64',
                'LANDAREA_MODE':'float64',
                'LIVINGAPARTMENTS_MODE':'float64',
                'LIVINGAREA_MODE':'float64',
                'NONLIVINGAPARTMENTS_MODE':'float64',
                'NONLIVINGAREA_MODE':'float64',
                'APARTMENTS_MEDI':'float64',
                'BASEMENTAREA_MEDI':'float64',
                'YEARS_BEGINEXPLUATATION_MEDI':'float64',
                'YEARS_BUILD_MEDI':'float64',
                'COMMONAREA_MEDI':'float64',
                'ELEVATORS_MEDI':'float64',
                'ENTRANCES_MEDI':'float64',
                'FLOORSMAX_MEDI':'float64',
                'FLOORSMIN_MEDI':'float64',
                'LANDAREA_MEDI':'float64',
                'LIVINGAPARTMENTS_MEDI':'float64',
                'LIVINGAREA_MEDI':'float64',
                'NONLIVINGAPARTMENTS_MEDI':'float64',
                'NONLIVINGAREA_MEDI':'float64',
                'FONDKAPREMONT_MODE':'object',
                'HOUSETYPE_MODE':'object',
                'TOTALAREA_MODE':'float64',
                'WALLSMATERIAL_MODE':'object',
                'EMERGENCYSTATE_MODE':'object',
                'OBS_30_CNT_SOCIAL_CIRCLE':'float64',
                'DEF_30_CNT_SOCIAL_CIRCLE':'float64',
                'OBS_60_CNT_SOCIAL_CIRCLE':'float64',
                'DEF_60_CNT_SOCIAL_CIRCLE':'float64',
                'DAYS_LAST_PHONE_CHANGE':'float64',
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
                'AMT_REQ_CREDIT_BUREAU_HOUR':'float64',
                'AMT_REQ_CREDIT_BUREAU_DAY':'float64',
                'AMT_REQ_CREDIT_BUREAU_WEEK':'float64',
                'AMT_REQ_CREDIT_BUREAU_MON':'float64',
                'AMT_REQ_CREDIT_BUREAU_QRT':'float64',
                'AMT_REQ_CREDIT_BUREAU_YEAR':'float64'
                }
        )


# In[11]:


#application_test = pd.read_table(data_file_path + 'application_test.csv', sep=',')


# In[12]:


#application_train = pd.read_table(data_file_path + 'application_train.csv', sep=',')


# In[13]:


#bureau = pd.read_table(data_file_path + 'bureau.csv', sep=',')


# In[14]:


#bureau_balance = pd.read_table(data_file_path + 'bureau_balance.csv', sep=',')


# In[15]:


#credit_card_balance = pd.read_table(data_file_path + 'credit_card_balance.csv', sep=',')


# In[16]:


#installments_payments = pd.read_table(data_file_path + 'installments_payments.csv', sep=',')


# In[17]:


#POS_CASH_balance = pd.read_table(data_file_path + 'POS_CASH_balance.csv', sep=',')


# In[18]:


#previous_application = pd.read_table(data_file_path + 'previous_application.csv', sep=',')

