# Base-Line Model
# Intercept Only Model
# Null Model
# 0IV Model

#%%
import matplotlib
import numpy
#import pandas
import sklearn
#import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor

#import ats_functions
import hcdr_functions


#%%

RANDOM_SEED = 1234567890


#%%

df = hcdr_functions.load_prepared_data_set__application_train_csv()


#%%

nullmodel_sm = smf.logit(formula = 'TARGET ~ 1', data = df)

nullmodel_sm_results = nullmodel_sm.fit()

print(nullmodel_sm_results.summary())

#print(nullmodel_sm_results.conf_int())


#%%

#X = ?
#y = df['TARGET']

#logit = sm.Logit(y, X)

#x = sm.add_constant(x, prepend=True)
#np.ones((X.shape[0],1))
#The resulting model have the expected mean as interception_ and coef_ is array([0.]).

#logit_model = sm.Logit(y,X)
#result = logit_model.fit()

#%%
df.shape[0]
#%%
numpy.ones((df.shape[0],1))
#%%
numpy.ones((df.shape[0],1)).shape
#%%
X = numpy.ones((df.shape[0],1))
#%%
y = df['TARGET']
#%%
nullmodel_sk = LogisticRegression(
        C=1.0,
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
        warm_start=False
        )
#%%
nullmodel_sk.fit(X, y)
#%%
print(nullmodel_sk.intercept_)
#%%
print(nullmodel_sk.coef_)
#%%
# Cross Validation Classification ROC AUC
nullmodel_sk_kfold = model_selection.KFold(n_splits=10, random_state=RANDOM_SEED)
nullmodel_sk_scoring = 'roc_auc'
nullmodel_sk_results = model_selection.cross_val_score(nullmodel_sk,
                                                       X,
                                                       y,
                                                       cv=nullmodel_sk_kfold,
                                                       scoring=nullmodel_sk_scoring)
#%%
#print("AUC: %.3f (%.3f)") % (results.mean(), results.std())
print(nullmodel_sk_results.mean())
print(nullmodel_sk_results.std())


#%%
nullmodel_sk__coefficients = nullmodel_sk.coef_
nullmodel_sk__intercept = nullmodel_sk.intercept_
nullmodel_sk__number_of_iterations = nullmodel_sk.n_iter_
nullmodel_sk__params = nullmodel_sk.get_params()
nullmodel_sk__X_predicted_class_labels = nullmodel_sk.predict(X)
nullmodel_sk__X_predicted_log_probability_estimates = nullmodel_sk.predict_log_proba(X)
nullmodel_sk__X_predicted_probability_estimates = nullmodel_sk.predict_proba(X)
nullmodel_sk__X_predicted_confidence_scores = nullmodel_sk.decision_function(X)
nullmodel_sk__score = nullmodel_sk.score(X, y, sample_weight=None)
nullmodel_sk__confusion_matrix = sklearn.metrics.confusion_matrix(y, nullmodel_sk__X_predicted_class_labels)
nullmodel_sk__classification_report = sklearn.metrics.classification_report(y, nullmodel_sk__X_predicted_class_labels)
nullmodel_sk__logit_roc_auc = sklearn.metrics.roc_auc_score(y, nullmodel_sk__X_predicted_class_labels)
nullmodel_sk__fpr, nullmodel_sk__tpr, nullmodel_sk__thresholds = sklearn.metrics.roc_curve(y, nullmodel_sk__X_predicted_probability_estimates[:,1])
nullmodel_sk__true_negative_count, nullmodel_sk__false_positive_count, nullmodel_sk__false_negative_count, nullmodel_sk__true_positive_count = nullmodel_sk__confusion_matrix.ravel()

print()
print('Displaying the model outputs...')
print()

print(nullmodel_sk__params)
print(nullmodel_sk__number_of_iterations)
print(nullmodel_sk__intercept)

print()
print('Displaying the model results...')
print()

print('Mean Accuracy on training data-set: {:.5f}'.format(nullmodel_sk__score))
#print(logistic_regression_models[i].confusion_matrix)
print('True Positves:   {:>6,}'.format(nullmodel_sk__true_positive_count))
print('False Positves:  {:>6,}'.format(nullmodel_sk__false_positive_count))
print('False Negatives: {:>6,}'.format(nullmodel_sk__false_negative_count))
print('True Negatives:  {:>6,}'.format(nullmodel_sk__true_negative_count))
print('Classification Report:', '\n', nullmodel_sk__classification_report)
print('Area Under the ROC Curve (AUC): {:.5f}'.format(nullmodel_sk__logit_roc_auc))

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(nullmodel_sk__fpr,
                       nullmodel_sk__tpr,
                       label='Logistic Regression (AUC = %0.5f)' % nullmodel_sk__logit_roc_auc)
matplotlib.pyplot.plot([0, 1], [0, 1],'r--')
matplotlib.pyplot.xlim([0.0, 1.0])
matplotlib.pyplot.ylim([0.0, 1.05])
matplotlib.pyplot.xlabel('False Positive Rate')
matplotlib.pyplot.ylabel('True Positive Rate')
matplotlib.pyplot.title('Receiver Operating Characteristic')
matplotlib.pyplot.legend(loc="lower right")
#matplotlib.pyplot.savefig('Log_ROC')
matplotlib.pyplot.show()
