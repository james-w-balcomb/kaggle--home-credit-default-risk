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