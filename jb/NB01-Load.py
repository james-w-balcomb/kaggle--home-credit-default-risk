
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


# In[6]:


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


# In[7]:


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


# In[8]:


# TODO(JamesBalcomb): add a function that handles specifying multiple files
data_file_name = 'application_train.csv'
print('data_file_name: ', data_file_name)


# In[9]:


df = pd.read_table(data_file_path + data_file_name, sep=',')


# In[10]:


#application_test = pd.read_table(data_file_path + 'application_test.csv', sep=',')


# In[11]:


#application_train = pd.read_table(data_file_path + 'application_train.csv', sep=',')


# In[12]:


#bureau = pd.read_table(data_file_path + 'bureau.csv', sep=',')


# In[13]:


#bureau_balance = pd.read_table(data_file_path + 'bureau_balance.csv', sep=',')


# In[14]:


#credit_card_balance = pd.read_table(data_file_path + 'credit_card_balance.csv', sep=',')


# In[15]:


#installments_payments = pd.read_table(data_file_path + 'installments_payments.csv', sep=',')


# In[16]:


#POS_CASH_balance = pd.read_table(data_file_path + 'POS_CASH_balance.csv', sep=',')


# In[17]:


#previous_application = pd.read_table(data_file_path + 'previous_application.csv', sep=',')

