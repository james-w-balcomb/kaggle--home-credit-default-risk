
# coding: utf-8

# This is the note book for Kaggle Competition - Give Me Some Credit

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz
from sklearn import preprocessing,model_selection
import itertools



# In[2]:


fileaddress = 'E:\\Kaggle\\Give Me Some Credit\\data'
train_df = pd.read_csv(fileaddress+'\\cs-training.csv')
test_df = pd.read_csv(fileaddress+'\\cs-test.csv')
print ("training dataset shape is {}".format(train_df.shape))
print ("testing dataset shape is {}".format(test_df.shape))


# In[3]:


col_names = train_df.columns.values
col_names[0] = 'ID' ## rename first column to ID
train_df.columns = col_names ## assign new column name to training dataset
test_df.columns = col_names ## assign new column name to testing dataset


# In[4]:


print ("Take a peek at training dataset")
train_df.head()


# In[5]:


print ("Take a peek at testing dataset")
test_df.head()


# Check column type

# In[6]:


print(train_df.dtypes)


# In[7]:


print(test_df.dtypes)


# Check distribution of each features to see outlier
# 
# * "MonthlyIncome" and "NumberOfDependents" are removed here as they have nan values

# In[8]:


# remove ID, target variable Dlqin2yrs and variables with missing values
feature_list=list(train_df.columns.values)
remove_list = ['ID','SeriousDlqin2yrs','MonthlyIncome','NumberOfDependents']
for each in remove_list:
    feature_list.remove(each)

for each in feature_list:
    sns.distplot(train_df[each])
    plt.show()


# Distribution of following features are highly skewed.
# 
# * RevolvingUtilizationOfUnsecuredLines
# * NumberOfTime30-59DaysPastDueNotWorse
# * DebtRatio
# * NumberOfTimes30DaysLate
# * NumberRealEstateLoansOrLines
# * NumberOfTime60-89DaysPastDueNotWorse
# 
# Take a log transformation to see if distribution can be less skewed.

# In[9]:


print (train_df.columns.values)


# In[10]:



log_trans_list = train_df.columns.values[[2,4,5,8,9,10]]
log_trans_list
for each in log_trans_list:
    train_df[each] = np.log(1+train_df[each].values)


# Distribution after log transformation

# In[11]:


for each in feature_list:
    sns.distplot(train_df[each])
    plt.show()


# The distribution after transformation is much less skewed. We may able to put them into machine learning algorithm later.

# Remove nan values in "MonthlyIncome" and "NumberOfDependents" to check their distribution

# In[12]:


partial_train_df = train_df[['MonthlyIncome','NumberOfDependents']]
#partial_train_df.dropna(how='any')
partial_train_df = partial_train_df.dropna(how='any')

sns.distplot(partial_train_df['MonthlyIncome'])
plt.show()
sns.distplot(partial_train_df['NumberOfDependents'])
plt.show()


# monthlyIncome is highly skewed. let us take log transformation on both then check their distribution again

# In[13]:


partial_train_df['MonthlyIncome'] = np.log(1+partial_train_df['MonthlyIncome'].values)
partial_train_df['NumberOfDependents'] = np.log(1+partial_train_df['NumberOfDependents'].values)
sns.distplot(partial_train_df['MonthlyIncome'])
plt.show()
sns.distplot(partial_train_df['NumberOfDependents'])
plt.show()


# Post transformation looks better than before. I will keep log transformation on both at this time.

# In[14]:


train_df['MonthlyIncome'] = np.log(1+train_df['MonthlyIncome'].values)
train_df['NumberOfDependents'] = np.log(1+train_df['NumberOfDependents'].values)


# check nan values in training set

# In[15]:


print (pd.isnull(train_df).sum(axis=0))


# Check nan values in testing set

# In[18]:


print (pd.isnull(test_df).sum(axis=0))


# Since the feature "Monthly Income" and "Number of Dependents" have many nan values in it. We create new features to see if observations with nan values in these two features are indictive for serious dlinquency rate.

# In[16]:


train_df['MonthlyIncome_Null'] = pd.isnull(train_df['MonthlyIncome'])
grouped_df = train_df.groupby('MonthlyIncome_Null')
Dlqin = grouped_df['SeriousDlqin2yrs'].aggregate(np.mean).reset_index()
Dlqin


# In[17]:


train_df['NoD_Null'] = pd.isnull(train_df['NumberOfDependents'])
grouped_df = train_df.groupby('NoD_Null')
Dlqin = grouped_df['SeriousDlqin2yrs'].aggregate(np.mean).reset_index()
Dlqin


# It seems observations with nan values in "Monthly Income" or/and "Number of Dpendents" have lower deliquency rate than those with valid values.

# In[18]:


print(train_df.shape,type(train_df))
train_df.dropna(axis=0,how='any',subset=['NumberOfDependents'],inplace=True)
train_df.reset_index()
print(train_df.shape)
pd.isnull(train_df).sum(axis=0)


# Simple way to get some new features

# In[19]:


#print(set(train_df['NumberOfDependents']))
#print(set(train_df['NumberOfDependents']+1))

train_df['IncomePerPerson'] = train_df['MonthlyIncome']/(train_df['NumberOfDependents']+1)
test_df['IncomePerPerson'] = test_df['MonthlyIncome']/(test_df['NumberOfDependents']+1)
train_df['NumOfPastDue'] = train_df['NumberOfTimes90DaysLate']+train_df['NumberOfTime60-89DaysPastDueNotWorse'] +train_df['NumberOfTime30-59DaysPastDueNotWorse']
test_df['NumOfPastDue'] = test_df['NumberOfTimes90DaysLate']+test_df['NumberOfTime60-89DaysPastDueNotWorse'] +test_df['NumberOfTime30-59DaysPastDueNotWorse']
train_df['MonthlyDebt'] = train_df['DebtRatio']*train_df['MonthlyIncome']
test_df['MonthlyDebt'] = test_df['DebtRatio']*test_df['MonthlyIncome']
train_df['NumOfOpenCreditLines'] = train_df['NumberOfOpenCreditLinesAndLoans']-train_df['NumberRealEstateLoansOrLines']
test_df['NumOfOpenCreditLines'] = test_df['NumberOfOpenCreditLinesAndLoans']-test_df['NumberRealEstateLoansOrLines']
train_df['MonthlyBalance'] = train_df['MonthlyIncome']-train_df['MonthlyDebt']
test_df['MonthlyBalance'] = test_df['MonthlyIncome']-test_df['MonthlyDebt']


# Group the data by age to see if there is any pattern in deliquency rate with respect to age

# In[20]:


grouped_df = train_df.groupby('age')
dlinq_age = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()
print(dlinq_age)
dlinq_age.columns =['age','DlqinFreq','count']
sns.regplot(x='age',y='DlqinFreq',data=dlinq_age)
plt.show()


# From the plot above, we can see:
# * DlinFreq is negatively associated with age in general
# * age of 0,99 and 101 looks like outliers
# * DlinFreq looks like a quardratic function of age. Put a higher order of age maybe helpful

# Remove outlier in age and create new feature $age^2$

# In[21]:


## remove outlier
train_df = train_df[train_df['age'] != 0]
train_df = train_df[train_df['age'] !=99]
train_df = train_df[train_df['age'] !=101]
grouped_df = train_df.groupby('age')
dlinq_age = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()
dlinq_age.columns =['age','DlqinFreq','count']
sns.regplot(x='age',y='DlqinFreq',data=dlinq_age)
plt.show()

## create new features
train_df['age_sqr'] = train_df['age'].values^2 
## apply the same operation on testing set
test_df['age_sqr'] = test_df['age'].values^2


# Split training data into 5 fold. Run xgboost and plot feature importance

# In[22]:


train_y = train_df['SeriousDlqin2yrs']
#'RevolvingUtilizationOfUnsecuredLines'
train_X = train_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False)
test_X = test_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False)
print(type(train_y))
skf = model_selection.StratifiedKFold(n_splits=5,random_state=100)
xgb_params = {
'eta':0.03,
'max_depth':4,
'sub_sample':0.9,
'colsample_bytree':0.5,
'objective':'binary:logistic',
'eval_metric':'auc',
'silent':0
}

print(train_X.shape)
print(train_X.columns)
print(test_X.shape)


# In[26]:


best_iteration =[]
best_score= []
training_score = []
for train_ind,val_ind in skf.split(train_X,train_y):
    #print (set(train_y))
    #print (type(train_y))
    X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,]
    y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind]
    #print (set(train_y))
    #print (max(train_ind),min(train_ind),max(val_ind),min(val_ind))
    #print (train_ind,val_ind)
    #print(set(y_train))
    dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns)
    dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns)
    model = xgb.train(xgb_params,dtrain,num_boost_round=1000,
                      evals=[(dtrain,'train'),(dval,'val')],verbose_eval=True,early_stopping_rounds=30)
    best_iteration.append(model.attributes()['best_iteration'])
    best_score.append(model.attributes()['best_score'])
    # training_score.append(model.attributes()['best_msg'].split()[1][-8:])
    xgb.plot_importance(model)
    plt.show()


# From the feature importance plot, we can see that the newly created features: "MonthlyIncome_Null" and "NoD_Null" is not as predictive as I thought. I am going to drop them off at this point.

# In[36]:


try:
    train_X.drop(['MonthlyIncome_Null','NoD_Null'],axis=1,inplace=True)
    train_df.drop(['MonthlyIncome_Null','NoD_Null'],axis=1,inplace=True)
except ValueError:
    print ("These features have been dropped")


# Define a function to perform cross validation in order to tune key parameters: 'eta','max_depth','sub_sample','colsample_bytree'

# In[47]:


def xgbCV(eta=[0.05],max_depth=[6],sub_sample=[0.9],colsample_bytree=[0.9]):
    train_y = train_df['SeriousDlqin2yrs'] # label for training data
    train_X = train_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for training data
    test_X = test_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for testing data
    skf = model_selection.StratifiedKFold(n_splits=5,random_state=100) # stratified sampling
    train_performance ={} 
    val_performance={}
    for each_param in itertools.product(eta,max_depth,sub_sample,colsample_bytree): # iterative over each combination in parameter space
        xgb_params = {
                    'eta':each_param[0],
                    'max_depth':each_param[1],
                    'sub_sample':each_param[2],
                    'colsample_bytree':each_param[3],
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }
        best_iteration =[]
        best_score=[]
        training_score=[]
        for train_ind,val_ind in skf.split(train_X,train_y): # five fold stratified cross validation
            X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,] # train X and train y
            y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind] # validation X and validation y
            dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns) # convert into DMatrix (xgb library data structure)
            dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns) # convert into DMatrix (xgb library data structure)
            model = xgb.train(xgb_params,dtrain,num_boost_round=1000, 
                              evals=[(dtrain,'train'),(dval,'val')],verbose_eval=False,early_stopping_rounds=30) # train the model
            best_iteration.append(model.attributes()['best_iteration']) # best iteration regarding AUC in valid set
            best_score.append(model.attributes()['best_score']) # best score regarding AUC in valid set
            training_score.append(model.attributes()['best_msg'].split()[1][10:]) # best score regarding AUC in training set
        valid_mean = (np.asarray(best_score).astype(np.float).mean()) # mean AUC in valid set
        train_mean = (np.asarray(training_score).astype(np.float).mean()) # mean AUC in training set
        val_performance[each_param] =  train_mean
        train_performance[each_param] =  valid_mean
        print ("Parameters are {}. Training performance is {:.4f}. Validation performance is {:.4f}".format(each_param,train_mean,valid_mean))
    return (train_performance,val_performance)
#xgbCV(eta=[0.01,0.02,0.03,0.04,0.05],max_depth=[4,6,8,10],colsample_bytree=[0.3,0.5,0.7,0.9]) 
xgbCV(eta=[0.04],max_depth=[4],colsample_bytree=[0.5])


# In[39]:


print(train_X.columns)
any(train_X.columns == test_X.columns)


# In[44]:


train = xgb.DMatrix(train_X,train_y,feature_names=train_X.columns)
test = xgb.DMatrix(test_X,feature_names=test_X.columns)
xgb_params = {
                    'eta':0.03,
                    'max_depth':4,
                    'sub_sample':0.9,
                    'colsample_bytree':0.5,
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }

final_model = xgb.train(xgb_params,train,num_boost_round=500)
ypred = final_model.predict(test)


# In[45]:


xgb.plot_importance(final_model)
plt.show()


# Plot a tree

# In[42]:


#xgb.to_graphviz(final_model,num_trees=0,size='20,20')
#plt.show()


# In[46]:


yout = pd.DataFrame({'Id':test_df.ID.values,'Probability':ypred})
yout.to_csv('E:\\Kaggle\\Give Me Some Credit\\result\\xgboost_result.csv',index=False)

