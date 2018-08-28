#%%

import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


#%%

RANDOM_SEED = 1234567890
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


#%%
# training data

df_application_train = pd.read_csv('../data/application_train.csv')


#%%
# testing data

df_application_test = pd.read_csv('../data/application_test.csv')


#%%
# replace categorical missing values

for column_name in df_application_train.columns.tolist():
    if df_application_train[column_name].dtype == 'object':
        if df_application_train[column_name].isnull().any():
            df_application_train[column_name] = df_application_train[column_name].fillna('MISSING')
for column_name in df_application_train.columns.tolist():
    if df_application_train[column_name].dtype == 'object':
        if df_application_train[column_name].isnull().any():
            df_application_test[column_name] = df_application_test[column_name].fillna('MISSING')


#%%
# Convert categorical dichotomous (binary) variables to {0,1}

label_encoder = LabelEncoder()

for column_name in df_application_train:
    if df_application_train[column_name].dtype == 'object':
        if df_application_train[column_name].nunique() == 2:
            # Train the LabelEncoder on the training data
            label_encoder.fit(df_application_train[column_name])
            # Transform both the training and testing data
            df_application_train[column_name] = label_encoder.transform(df_application_train[column_name])
            df_application_test[column_name] = label_encoder.transform(df_application_test[column_name])


#%%
# dummy encode the categorical variables
            
df_application_train = pd.get_dummies(df_application_train, drop_first=True)
df_application_test = pd.get_dummies(df_application_test, drop_first=True)


#%%
# replace numerical missing values

for column_name in df_application_train.columns.tolist():
    if df_application_train[column_name].dtype == 'float':
        column_mean = df_application_train[column_name].mean()
        if df_application_train[column_name].isnull().any():
            df_application_train[column_name] = df_application_train[column_name].fillna(column_mean)
        if df_application_test[column_name].isnull().any():
            df_application_test[column_name] = df_application_test[column_name].fillna(column_mean)


#%%

mismatched_columns_in_train = sorted(list(set(df_application_train.columns.tolist()) - set(df_application_test.columns.tolist())))
for column_name in mismatched_columns_in_train:
    if column_name == 'TARGET':
        continue
    df_application_test[column_name] = 0


#%%

X_train = df_application_train.drop(columns = ['SK_ID_CURR', 'TARGET'])

y_train = df_application_train['TARGET']

X_test = df_application_test.drop(columns = ['SK_ID_CURR'])


#%%
print("Scale each feature to mean=0, stddev=1 (\"z-score\")")

for column_name in X_train:
    if column_name == 'TARGET':
        continue
    if X_train[column_name].nunique() == 2:
        continue
    column_mean = X_train[column_name].mean()
    column_stddev = X_train[column_name].std(ddof=0)
    X_train[column_name] = (X_train[column_name] - column_mean) / column_stddev
    X_test[column_name] = (X_test[column_name] - column_mean) / column_stddev


#%%
print("Instantiate an instance of the LogisticRegression class")

logreg_sk = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    #class_weight=None,
    class_weight='balanced',
    #random_state=None,
    random_state=RANDOM_SEED,
    solver='liblinear',
    max_iter=100,
    multi_class='ovr',
    verbose=0,
    warm_start=False,
    n_jobs=1
    )


#%%
print("Fit the Logistic Regression model")
logreg_sk.fit(X_train, y_train)


#%%
print("Produce the predictions from the fitted Logistic Regression model")
logreg_sk_predict_proba = logreg_sk.predict_proba(X_test)[:,1]


#%%
print("Save predictions to a file, for submission")
df_submission = pd.DataFrame(df_application_test.loc[:,'SK_ID_CURR'])
df_submission['TARGET'] = logreg_sk_predict_proba
df_submission.to_csv('LogReg_MVP_Sk.csv', index= False)
