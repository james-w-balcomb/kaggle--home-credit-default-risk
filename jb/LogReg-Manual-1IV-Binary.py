#%%

import numpy

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import hcdr_functions


#%%

RANDOM_SEED = 1234567890


#%%

df = hcdr_functions.load_data_file__application_train_csv()


#%%

df['CODE_GENDER'] = df['CODE_GENDER'].astype('object')

df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', numpy.nan)

df.loc[df['SK_ID_CURR'] == 141289, 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == 319880, 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == 196708, 'CODE_GENDER'] = 'F'
df.loc[df['SK_ID_CURR'] == 144669, 'CODE_GENDER'] = 'M'

#df['CODE_GENDER__Male'] = pandas.Series(numpy.where(df['CODE_GENDER'].values == 'M', 1, 0), df.index)
df['CODE_GENDER__Male'] = 0
df.loc[df['CODE_GENDER'] == 'M', 'CODE_GENDER__Male'] = 1
#df.loc[df['CODE_GENDER'] == 'F', 'CODE_GENDER__Male'] = 0

df['CODE_GENDER'] = df['CODE_GENDER'].astype('category')
#df['CODE_GENDER__Male'] = df['CODE_GENDER'].astype('category')


#%%

dependent_column_name = 'TARGET'
independent_column_name = 'CODE_GENDER__Male'
y_train = df[dependent_column_name].astype(int)
X_train = df[independent_column_name].values.reshape(-1,1)


#%%

kfold = model_selection.KFold(n_splits=10,
                              random_state=RANDOM_SEED)

model = LogisticRegression(C=1.0,
                           #class_weight=None,
                           class_weight='balanced',
                           dual=False,
                           fit_intercept=True,
                           intercept_scaling=1,
                           max_iter=100,
                           multi_class='ovr',
                           n_jobs=1,
                           penalty='l2',
                           #random_state=None,
                           random_state=RANDOM_SEED,
                           solver='liblinear',
                           tol=0.0001,
                           verbose=0,
                           warm_start=False)


#%%

# Cross Validation Classification ROC AUC
scoring = 'roc_auc'
results = model_selection.cross_val_score(model,
                                          X_train,
                                          y_train,
                                          cv=kfold,
                                          scoring=scoring)
#print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
print(results.mean())
print(results.std())

#%%

model.fit(X_train, y_train)


#%%

print(model.intercept_)
print(model.coef_)
