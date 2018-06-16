
# coding: utf-8

# In[ ]:


# https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/


# In[ ]:


#%run NB01-Load.ipynb


# In[ ]:


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


# In[ ]:


print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))
print("SciPy version: {}".format(sp.__version__))
print("scikit-learn version: {}".format(sklearn.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("IPython version: {}".format(IPython.__version__))


# In[ ]:


random.seed(1234567890)
numpy.random.seed(1234567890)


# In[ ]:


path = "C:/Development/kaggle--home-credit-default-risk/data/"


# In[ ]:


application_train = pd.read_table(path + 'application_train.csv', sep=',', dtype=object)


# In[ ]:


application_train__ext_source = application_train.loc[:, ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]


# In[ ]:


# Remove observations with missing values
#application_train__ext_source.dropna(inplace = True)
application_train__ext_source__dropna = application_train__ext_source.dropna()


# In[ ]:


#application_train['EXT_SOURCE_AVG'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)


# # 1. Classification Accuracy

# In[ ]:


# Cross Validation Classification Accuracy


# In[ ]:


#dataframe = application_train__ext_source
dataframe = application_train__ext_source__dropna


# In[ ]:


array = dataframe.values


# In[ ]:


X = array[:,0:2]


# In[ ]:


Y = array[:,3]


# In[ ]:


seed = 1234567890


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=seed)


# In[ ]:


model = LogisticRegression()


# In[ ]:


scoring = 'accuracy'


# In[ ]:


results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# In[ ]:


print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:


#dataframe = application_train__ext_source
dataframe = application_train__ext_source__dropna


# In[ ]:


array = dataframe.values


# In[ ]:


X = array[:,0:2]


# In[ ]:


Y = array[:,3]


# In[ ]:


seed = 1234567890


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=seed)


# In[ ]:


model = LogisticRegression(class_weight='balanced')


# In[ ]:


scoring = 'accuracy'


# In[ ]:


results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# In[ ]:


print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# # 4. Confusion Matrix
# https://en.wikipedia.org/wiki/Confusion_matrix

# ## Unbalanced Logistic Regression

# In[ ]:


# Cross Validation Classification Confusion Matrix
dataframe = application_train__ext_source__dropna
array = dataframe.values
X = array[:,0:2]
Y = array[:,3]
test_size = 0.33
seed = 1234567890
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
# TP FP
# FN TN


# In[ ]:


# True-Positive (TP)
TP = matrix[0,0]
TP


# In[ ]:


# False-Positive (FP)
FP = matrix[0,1]
FP


# In[ ]:


# False-Negative (FN)
FN = matrix[1,0]
FN


# In[ ]:


# True-Negative (TN)
TN = matrix[1,1]
TN


# In[ ]:


# accuracy (ACC)
# ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
ACC = (TP+TN)/(TP+TN+FP+FN)
ACC


# In[ ]:


# sensitivity, recall, hit rate, or true positive rate (TPR)
# TPR = TP/P = TP/(TP+FN)
TPR = TP/(TP+FN)
TPR


# In[ ]:


# specificity or true negative rate (TNR)
# TNR = TN/N = TN/(TN+FP)
TNR = TN/(TN+FP)
TNR


# In[ ]:


# precision or positive predictive value (PPV)
# PPV = TP/(TP+FP)
PPV = TP/(TP+FP)
PPV


# In[ ]:


# negative predictive value (NPV)
# NPV = TN/(TN+FN)
NPV = TN/(TN+FN)
NPV


# In[ ]:


# miss rate or false negative rate (FNR)
# FNR = FN/P = FN/(FN+TP) = 1-TPR
FNR = FN/(FN+TP)
FNR


# In[ ]:


# fall-out or false positive rate (FPR)
# FPR = FP/N = FP/(FP+TN) = 1-TNR
FPR = FP/(FP+TN)
FPR


# In[ ]:


# false discovery rate (FDR)
# FDR = FP/(FP+TP) = 1-PPV
FDR = FP/(FP+TP)
FDR


# In[ ]:


# false omission rate (FOR)
# FOR = FN/(FN+TN) = 1-NPV
FOR = FN/(FN+TN)
FOR


# In[ ]:


# F1 score (...is the harmonic mean of precision and sensitivity)
# F1 = 2*((PPV*TPR)/(PPV+TPR)) = 2*TP/(2*TP+FP+FN)
F1 = 2*TP/(2*TP+FP+FN)
F1


# In[ ]:


# Matthews correlation coefficient (MCC)
# MCC = (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
MCC = (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
MCC


# In[ ]:


# Informedness or Bookmaker Informedness (BM)
# BM = TPR+TNR-1
BM = TPR+TNR-1
BM


# In[ ]:


# Markedness (MK)
# MK = PPV+NPV-1
MK = PPV+NPV-1
MK


# In[ ]:


report = classification_report(Y_test, predicted)
print(report)


# ## Balanced Logistic Regression

# In[ ]:


# Cross Validation Classification Confusion Matrix
dataframe = application_train__ext_source__dropna
array = dataframe.values
X = array[:,0:2]
Y = array[:,3]
test_size = 0.33
seed = 1234567890
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)
# TP FP
# FN TN


# In[ ]:


# True-Positive (TP)
TP = matrix[0,0]
TP


# In[ ]:


# False-Positive (FP)
FP = matrix[0,1]
FP


# In[ ]:


# False-Negative (FN)
FN = matrix[1,0]
FN


# In[ ]:


# True-Negative (TN)
TN = matrix[1,1]
TN


# In[ ]:


# accuracy (ACC)
# ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
ACC = (TP+TN)/(TP+TN+FP+FN)
ACC


# In[ ]:


# sensitivity, recall, hit rate, or true positive rate (TPR)
# TPR = TP/P = TP/(TP+FN)
TPR = TP/(TP+FN)
TPR


# In[ ]:


# specificity or true negative rate (TNR)
# TNR = TN/N = TN/(TN+FP)
TNR = TN/(TN+FP)
TNR


# In[ ]:


# precision or positive predictive value (PPV)
# PPV = TP/(TP+FP)
PPV = TP/(TP+FP)
PPV


# In[ ]:


# negative predictive value (NPV)
# NPV = TN/(TN+FN)
NPV = TN/(TN+FN)
NPV


# In[ ]:


# miss rate or false negative rate (FNR)
# FNR = FN/P = FN/(FN+TP) = 1-TPR
FNR = FN/(FN+TP)
FNR


# In[ ]:


# fall-out or false positive rate (FPR)
# FPR = FP/N = FP/(FP+TN) = 1-TNR
FPR = FP/(FP+TN)
FPR


# In[ ]:


# false discovery rate (FDR)
# FDR = FP/(FP+TP) = 1-PPV
FDR = FP/(FP+TP)
FDR


# In[ ]:


# false omission rate (FOR)
# FOR = FN/(FN+TN) = 1-NPV
FOR = FN/(FN+TN)
FOR


# In[ ]:


# F1 score (...is the harmonic mean of precision and sensitivity)
# F1 = 2*((PPV*TPR)/(PPV+TPR)) = 2*TP/(2*TP+FP+FN)
F1 = 2*TP/(2*TP+FP+FN)
F1


# In[ ]:


# Matthews correlation coefficient (MCC)
# MCC = (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
MCC = (TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
MCC


# In[ ]:


# Informedness or Bookmaker Informedness (BM)
# BM = TPR+TNR-1
BM = TPR+TNR-1
BM


# In[ ]:


# Markedness (MK)
# MK = PPV+NPV-1
MK = PPV+NPV-1
MK


# In[ ]:


report = classification_report(Y_test, predicted)
print(report)

