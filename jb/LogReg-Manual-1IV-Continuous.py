#%%

import pandas

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import hcdr_functions


#%%

RANDOM_SEED = 1234567890


#%%

df = hcdr_functions.load_data_file__application_train_csv()


#%%

column_name = 'AMT_INCOME_TOTAL'
Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)
IQR = Q3 - Q1
IQR150 = IQR * 1.50
Q1IQR150 = Q1 - IQR150
Q3IQR150 = Q3 + IQR150
df[column_name + '__capped_iqr'] = df[column_name]
df.loc[df[column_name + '__capped_iqr'] < Q1IQR150, column_name + '__capped_iqr'] = Q1IQR150 - 1
df.loc[df[column_name + '__capped_iqr'] > Q3IQR150, column_name + '__capped_iqr'] = Q3IQR150 + 1


#%%

column_name = 'AMT_INCOME_TOTAL'
df[column_name + '__zscore'] = (df[column_name] - df[column_name].mean()) / df[column_name].std(ddof=0)

column_name = 'AMT_INCOME_TOTAL__capped_iqr'
df[column_name + '__zscore'] = (df[column_name] - df[column_name].mean()) / df[column_name].std(ddof=0)


#%%

pandas.options.display.float_format = '{:.2f}'.format
print()
print(df['AMT_INCOME_TOTAL'].describe())
print()
print(df['AMT_INCOME_TOTAL__capped_iqr'].describe())
print()
print(df['AMT_INCOME_TOTAL__zscore'].describe())
print()
print(df['AMT_INCOME_TOTAL__capped_iqr__zscore'].describe())
print()


#%%

dependent_column_name = 'TARGET'
#independent_column_name = 'AMT_INCOME_TOTAL'
#independent_column_name = 'AMT_INCOME_TOTAL__capped_iqr'
#independent_column_name = 'AMT_INCOME_TOTAL__zscore'
independent_column_name = 'AMT_INCOME_TOTAL__capped_iqr__zscore'
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


#%%

### independent_column_name = 'AMT_INCOME_TOTAL'
#print(results.mean())
#0.5191501236347784
#print(results.std())
#0.0057890585700015
#print(model.intercept_)
#[1.87636461e-14]
#print(model.coef_)
#[[-1.28017337e-08]]

### independent_column_name = 'AMT_INCOME_TOTAL__capped_iqr'
#print(results.mean())
#0.5191433133926201
#print(results.std())
#0.005797374873864827
#print(model.intercept_)
#[1.02754668e-12]
#print(model.coef_)
#[[-1.96881488e-07]]

### independent_column_name = 'AMT_INCOME_TOTAL__zscore'
#print(results.mean())
#0.5191501236347784
#print(results.std())
#0.0057890585700015
#print(model.intercept_)
#[-2.87413424e-05]
#print(model.coef_)
#[[-0.0035589]]

### independent_column_name = 'AMT_INCOME_TOTAL__capped_iqr__zscore'
#print(results.mean())
#0.5191433133926201
#print(results.std())
#0.005797374873864827
#print(model.intercept_)
#[-0.00306471]
#print(model.coef_)
#[[-0.08743046]]
