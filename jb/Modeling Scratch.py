
# coding: utf-8

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


data_file_path = 'C:/Development/kaggle--home-credit-default-risk/data/'
data_file_path


# In[ ]:


#application_train = pd.read_table(path + 'application_train.csv', sep=',', dtype=object)
application_train = pd.read_table(data_file_path + 'application_train.csv', sep=',')


# In[ ]:


application_train.head()


# In[ ]:


#application_train[other_columns] = ds13[other_columns].fillna(value = "MISSING")


# In[ ]:


application_train__ext_source = application_train.loc[:, ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]


# In[ ]:


# Remove observations with missing values
#application_train__ext_source.dropna(inplace = True)
application_train__ext_source__dropna = application_train__ext_source.dropna()


# In[ ]:


#Create train and validation set
X_train, X_test, y_train, y_test = train_test_split(application_train__ext_source, application_train__ext_source['TARGET'], random_state=0)
#train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)


# In[ ]:


print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[ ]:


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


# Train model
model = logreg.fit(X_train, y_train)


# In[ ]:


# Predict class
model.predict(X_test)


# In[ ]:


# View predicted probabilities
model.predict_proba(X_test)


# In[ ]:


logisticRegr = LogisticRegression()


# In[ ]:


logisticRegr.fit(ds_train, convert_train.values.ravel())


# In[ ]:


#print("Score=%.3f" % clf.score(X, grades["Letter"]))


# In[ ]:


#print(logisticRegr.score(ds_test, convert_test))


# In[ ]:


predictions = logisticRegr.predict(ds_test)


# In[ ]:


print(logisticRegr.score(ds_test, convert_test))


# In[ ]:


#cm = confusion_matrix(predictions, convert_test)


# In[ ]:


#print(pd.DataFrame(cm, columns=labels, index=labels))
#print(pd.DataFrame(cm)


# In[ ]:


seed = 1234567890


# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=seed)


# In[ ]:


model = LogisticRegression(class_weight='balanced')


# In[ ]:


scoring = 'accuracy'


# In[ ]:


results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)


# In[ ]:


print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())


# In[ ]:


print(results.mean())


# In[ ]:


print(results.std())


# In[ ]:


logit = sm.Logit(y_train, X_train)


# In[ ]:


result = logit.fit(maxiter=999)


# In[ ]:


print result.summary()


# In[ ]:


#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)


# In[ ]:


# https://medium.com/@sunwoopark/kaggle-%EB%8F%84%EC%A0%84%EA%B8%B0-home-credit-default-risk-part-1-735030d40ee0
### SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

### ROC_AUC_SCORE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train) 
y_pred = logreg.predict_proba(X_test_set)[:,1]
roc_auc_score(y_test_set, y_pred)


# In[ ]:


# https://medium.com/@faizanahemad/participating-in-kaggle-data-science-competitions-part-1-step-by-step-guide-and-baseline-model-5b0c6973022a


# In[ ]:


df = application_train


# In[ ]:


# How many classes
df["TARGET"].nunique()


# In[ ]:


# Distribution of those classes
df["TARGET"].value_counts(dropna=False)


# In[ ]:


dtypes = df.dtypes
dtypes = dtypes[dtypes != 'object']
features = list(set(dtypes.index) - set(['TARGET']))
len(features)


# In[ ]:


X = df[features]
y = df['TARGET']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


model = XGBClassifier(max_depth=6,
                      learning_rate=0.1,
                      n_estimators=100,
                      n_jobs=16,
                      scale_pos_weight=4,
                      missing=np.nan,
                      gamma=16,
                      eval_metric='auc',
                      reg_lambda=40,reg_alpha=40
                     )
model.fit(X_train,y_train)


# ### Step 10: Scoring on Train and Test set
# 
# We will predict probabilities of our TARGET=1,
#   P(TARGET=1|X) and use it for finding AUC_ROC metric.

# In[ ]:


from sklearn.metrics import roc_auc_score
y_train_predicted = model.predict_proba(X_train)[:,1]
y_test_predicted = model.predict_proba(X_test)[:,1]

print('Train AUC %.4f' % roc_auc_score(y_train,y_train_predicted))
print('Test AUC %.4f' % roc_auc_score(y_test,y_test_predicted))


# In[ ]:


def generate_results(model,df_test,features,id_col,target,file):
    dft = df_test[features]
    results = df_test[[id_col]]
    results[target] = model.predict_proba(dft)[:,1]
    results.to_csv(file,index=False,columns=results.columns
)
    
generate_results(model,df_test,features,"SK_ID_CURR","TARGET","results/results.csv")


# [Jupyter Notebook - Typesetting Equations](http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html)
# The Markdown parser included in the Jupyter Notebook is MathJax-aware. This means that you can freely mix in mathematical expressions using the [MathJax subset of Tex and LaTeX](http://docs.mathjax.org/en/latest/tex.html#tex-support). Some examples from the MathJax site are reproduced below, as well as the Markdown+TeX source.
# [MathJax basic tutorial and quick reference](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)
# 
# \begin{align}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{align}
# 
# \begin{equation*}
# P(E)   = {n \choose k} p^k (1-p)^{ n-k}
# \end{equation*}
# 
# This expression $\sqrt{3x-1}+(1+x)^2$ is an example of a TeX inline equation in a [Markdown-formatted](https://daringfireball.net/projects/markdown/) sentence.
# 
# ###### Other Syntax
# You will notice in other places on the web that $$ are needed explicitly to begin and end MathJax typesetting. This is not required if you will be using TeX environments, but the Jupyter notebook will accept this syntax on legacy notebooks.
# 
# $$\begin{eqnarray}
# x' &=& &x \sin\phi &+& z \cos\phi \\
# z' &=& - &x \cos\phi &+& z \sin\phi \\
# \end{eqnarray}$$

# \begin{align}
# Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4
# \end{align}
# \begin{align}
# R^2
# \end{align}

# # Multicollinearity Detection
# https://statinfer.com/204-1-9-issue-of-multicollinearity-in-python/
#     Y = β0 + β1X1 + β2X2 + β3X3 + β4X4
#     Build a model X1 vs X2 X3 X4 find R^2, say R1
#     Build a model X2 vs X1 X3 X4 find R^2, say R2
#     Build a model X3 vs X1 X2 X4 find R^2, say R3
#     Build a model X4 vs X1 X2 X3 find R^2, say R4
#     For example if R3 is 95% then we don’t really need X3 in the model.
#     Since it can be explained as liner combination of other three.
#     For each variable we find individual R2.
#     1/(1 − R^2) is called VIF.

# In[ ]:


application_train__ext_source = application_train.loc[:, ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]


# In[ ]:


# Remove observations with missing values
application_train__ext_source__dropna = application_train__ext_source.dropna()


# In[ ]:


"""
There is no function for calculating VIF values.
None of the pre-built libraries have this VIF calculation function
We may have to write our own function to calculate VIF values for each variable 
"""

# Code for VIF Calculation
# Writing a function to calculate the VIF values
#def variable_inflation_factor_calculations(input_data, dependent_col):
def vif_cal(input_data, dependent_col):
    x_vars = input_data.drop([dependent_col], axis=1)
    xvar_names = x_vars.columns
    for i in range(0, xvar_names.shape[0]):
        y = x_vars[xvar_names[i]] 
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = smf.ols(formula='y ~ x', data = x_vars).fit().rsquared  
        vif = round(1 / (1 - rsq), 2)
        print (xvar_names[i], 'VIF = ' , vif)


# In[ ]:


# EXT_SOURCE_1 ~ EXT_SOURCE_2 + EXT_SOURCE_3
linreg1 = LinearRegression()
linreg1.fit(application_train__ext_source__dropna[['EXT_SOURCE_2'] + ['EXT_SOURCE_3']], application_train__ext_source__dropna[['EXT_SOURCE_1']])
linreg1_predicted = linreg1.predict(application_train__ext_source__dropna[['EXT_SOURCE_2'] + ['EXT_SOURCE_3']])
linreg1_model = smf.ols(formula='EXT_SOURCE_1 ~ EXT_SOURCE_2 + EXT_SOURCE_3', data=application_train__ext_source__dropna)
linreg1_fitted = linreg1_model.fit()


# In[ ]:


linreg1_fitted.summary()


# In[ ]:


linreg1_fitted.summary2()


# In[ ]:


# Calculating VIF values using that function
vif_cal(input_data = application_train__ext_source__dropna, dependent_col = 'EXT_SOURCE_1')


# In[ ]:


# EXT_SOURCE_2 ~ EXT_SOURCE_3 + EXT_SOURCE_1
linreg2 = LinearRegression()
linreg2.fit(application_train__ext_source__dropna[['EXT_SOURCE_3'] + ['EXT_SOURCE_1']], application_train__ext_source__dropna[['EXT_SOURCE_2']])
linreg2_predicted = linreg1.predict(application_train__ext_source__dropna[['EXT_SOURCE_3'] + ['EXT_SOURCE_1']])
linreg2_model = smf.ols(formula='EXT_SOURCE_2 ~ EXT_SOURCE_3 + EXT_SOURCE_1', data=application_train__ext_source__dropna)
linreg2_fitted = linreg1_model.fit()


# In[ ]:


linreg2_fitted.summary()


# In[ ]:


linreg2_fitted.summary2()


# In[ ]:


# Calculating VIF values using that function
vif_cal(input_data = application_train__ext_source__dropna, dependent_col = 'EXT_SOURCE_2')


# In[ ]:


# EXT_SOURCE_3 ~ EXT_SOURCE_1 + EXT_SOURCE_2
linreg3 = LinearRegression()
linreg3.fit(application_train__ext_source__dropna[['EXT_SOURCE_1'] + ['EXT_SOURCE_2']], application_train__ext_source__dropna[['EXT_SOURCE_3']])
linreg3_predicted = linreg1.predict(application_train__ext_source__dropna[['EXT_SOURCE_1'] + ['EXT_SOURCE_2']])
linreg3_model = smf.ols(formula='EXT_SOURCE_3 ~ EXT_SOURCE_1 + EXT_SOURCE_2', data=application_train__ext_source__dropna)
linreg3_fitted = linreg1_model.fit()


# In[ ]:


linreg3_fitted.summary()


# In[ ]:


linreg3_fitted.summary2()


# In[ ]:


# Calculating VIF values using that function
vif_cal(input_data = application_train__ext_source__dropna, dependent_col = 'EXT_SOURCE_3')


# In[ ]:


# https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
# df.columns.to_series().groupby(df.dtypes).groups
application_train.columns.to_series().groupby(application_train.dtypes).groups

