import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

train = pd.read_csv("data/application_train.csv")
test = pd.read_csv("data/application_test.csv")


# common fuction
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))


def log_transform(frame, feature):
    frame[feature] = np.log1p(frame[feature].values)


def quadratic(frame, feature):
    frame[feature + '2'] = frame[feature] ** 2


# customize function
def func_NAME_EDUCATION_TYPE(x):
    if x in ('Higher education', 'Academic degree'):
        return 1
    else:
        return 0


def func_NAME_HOUSING_TYPE(x):
    if x in ('Maternity leave', 'Unemployede'):
        return 1
    else:
        return 0


def feature_processing(frame):
    _FLAG_DOCUMENT_SUM = frame[[col for col in frame.columns if 'FLAG_DOCUMENT_' in col]]
    frame['FLAG_DOCUMENT_SUM'] = _FLAG_DOCUMENT_SUM.sum(axis=1)

    _FLAG_PHONE_SUM = frame[[
        'FLAG_MOBIL',
        'FLAG_EMP_PHONE',
        'FLAG_WORK_PHONE',
        'FLAG_CONT_MOBILE',
        'FLAG_PHONE']]
    frame['PHONE_SUM'] = _FLAG_PHONE_SUM.sum(axis=1)

    frame['YEARS_BIRTH'] = frame['DAYS_BIRTH'] * (-1) / 365
    frame['YEARS_EMPLOYED'] = frame['DAYS_EMPLOYED'] * (-1) / 365
    frame['YEARS_REGISTRATION'] = frame['DAYS_REGISTRATION'] * (-1) / 365
    frame['YEARS_ID_PUBLISH'] = frame['DAYS_ID_PUBLISH'] * (-1) / 365
    frame['YEARS_LAST_PHONE_CHANGE'] = frame['DAYS_LAST_PHONE_CHANGE'] * (-1) / 365

    frame['AMT_INCOME_TOTAL_PER_FAM_MEMBERS'] = frame['AMT_INCOME_TOTAL'] / frame['CNT_FAM_MEMBERS']

    frame['NAME_CONTRACT_TYPE'] = frame['NAME_CONTRACT_TYPE'].apply(lambda x: 1 if x == 'Cash loans' else 0)
    frame['FLAG_OWN_CAR'] = frame['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'y' else 0)
    frame['AMT_INCOME_TOTAL'] = frame['AMT_INCOME_TOTAL'].apply(lambda x: 1 if x > 13.3 else 0)

    frame['NAME_EDUCATION_TYPE'] = frame['NAME_EDUCATION_TYPE'].apply(func_NAME_EDUCATION_TYPE)
    frame['NAME_HOUSING_TYPE'] = frame['NAME_HOUSING_TYPE'].apply(func_NAME_HOUSING_TYPE)

    frame['REGION_POPULATION_RELATIVE'] = frame['REGION_POPULATION_RELATIVE'].apply(lambda x: 1 if x >= 0.02 else 0)
    frame['OWN_CAR_AGE'] = frame['OWN_CAR_AGE'].apply(lambda x: 1 if x <= 10 else 0)


def drop_columns(frame):
    frame = frame.drop(columns=['APARTMENTS_MEDI',
                                'BASEMENTAREA_MEDI',
                                'YEARS_BEGINEXPLUATATION_MEDI',
                                'YEARS_BUILD_MEDI',
                                'COMMONAREA_MEDI',
                                'ELEVATORS_MEDI',
                                'ENTRANCES_MEDI',
                                'FLOORSMAX_MEDI',
                                'FLOORSMIN_MEDI',
                                'LANDAREA_MEDI',
                                'LIVINGAPARTMENTS_MEDI',
                                'LIVINGAREA_MEDI',
                                'NONLIVINGAPARTMENTS_MEDI',
                                'NONLIVINGAREA_MEDI',
                                'APARTMENTS_MODE',
                                'BASEMENTAREA_MODE',
                                'YEARS_BEGINEXPLUATATION_MODE',
                                'YEARS_BUILD_MODE',
                                'COMMONAREA_MODE',
                                'ELEVATORS_MODE',
                                'ENTRANCES_MODE',
                                'FLOORSMAX_MODE',
                                'FLOORSMIN_MODE',
                                'LANDAREA_MODE',
                                'LIVINGAPARTMENTS_MODE',
                                'LIVINGAREA_MODE',
                                'NONLIVINGAPARTMENTS_MODE',
                                'NONLIVINGAREA_MODE',
                                'FONDKAPREMONT_MODE',
                                'HOUSETYPE_MODE',
                                'TOTALAREA_MODE',
                                'WALLSMATERIAL_MODE',
                                'EMERGENCYSTATE_MODE',
                                'APARTMENTS_AVG',
                                'BASEMENTAREA_AVG',
                                'YEARS_BEGINEXPLUATATION_AVG',
                                'YEARS_BUILD_AVG',
                                'COMMONAREA_AVG',
                                'ELEVATORS_AVG',
                                'ENTRANCES_AVG',
                                'FLOORSMAX_AVG',
                                'FLOORSMIN_AVG',
                                'LANDAREA_AVG',
                                'LIVINGAPARTMENTS_AVG',
                                'LIVINGAREA_AVG',
                                'NONLIVINGAPARTMENTS_AVG',
                                'NONLIVINGAREA_AVG',
                                'FLAG_DOCUMENT_2',
                                'FLAG_DOCUMENT_3',
                                'FLAG_DOCUMENT_4',
                                'FLAG_DOCUMENT_5',
                                'FLAG_DOCUMENT_6',
                                'FLAG_DOCUMENT_7',
                                'FLAG_DOCUMENT_8',
                                'FLAG_DOCUMENT_9',
                                'FLAG_DOCUMENT_10',
                                'FLAG_DOCUMENT_11',
                                'FLAG_DOCUMENT_12',
                                'FLAG_DOCUMENT_13',
                                'FLAG_DOCUMENT_14',
                                'FLAG_DOCUMENT_15',
                                'FLAG_DOCUMENT_16',
                                'FLAG_DOCUMENT_17',
                                'FLAG_DOCUMENT_18',
                                'FLAG_DOCUMENT_19',
                                'FLAG_DOCUMENT_20',
                                'FLAG_DOCUMENT_21',
                                'FLAG_MOBIL',
                                'FLAG_EMP_PHONE',
                                'FLAG_WORK_PHONE',
                                'FLAG_CONT_MOBILE',
                                'FLAG_PHONE',
                                'SK_ID_CURR',
                                'DAYS_BIRTH',
                                'DAYS_EMPLOYED',
                                'DAYS_REGISTRATION',
                                'DAYS_ID_PUBLISH',
                                'DAYS_LAST_PHONE_CHANGE'
                                ])


def categorical_processing(frame):
    for c in categorical:
        frame[c] = frame[c].astype('category')
        if frame[c].isnull().any():
            frame[c] = frame[c].cat.add_categories(['MISSING'])
            frame[c] = frame[c].fillna('MISSING')


def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['ordering'].to_dict()

    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature + '_E'] = o


feature_processing(train)
feature_processing(test)

numerical = [f for f in train.columns if train.dtypes[f] != 'object']
numerical.remove('TARGET')
categorical = [f for f in train.columns if train.dtypes[f] == 'object']

log_transform(train, 'AMT_CREDIT')
log_transform(train, 'AMT_ANNUITY')
log_transform(train, 'AMT_GOODS_PRICE')
log_transform(train, 'AMT_INCOME_TOTAL')

log_transform(test, 'AMT_CREDIT')
log_transform(test, 'AMT_ANNUITY')
log_transform(test, 'AMT_GOODS_PRICE')
log_transform(test, 'AMT_INCOME_TOTAL')

drop_columns(train)
drop_columns(test)

cate_encoded = []
for q in categorical:
    encode(train, q)
    encode(test, q)
    cate_encoded.append(q + '_E')

features = numerical + cate_encoded

split = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
for train_index, test_index in split.split(train, train["NAME_INCOME_TYPE"]):
    train_set = train.loc[train_index]
    test_set = train.loc[test_index]

X_train = train_set[features].fillna(0.).values
y_train = train_set['TARGET'].values
X_test_set = test_set[features].fillna(0.).values
y_test_set = test_set['TARGET'].values

### SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

### ROC_AUC_SCORE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
# logreg.fit(X_resampled, y_resampled)  # 0.6201171140754779
logreg.fit(X_train, y_train)            # 0.6373158605096496
y_pred = logreg.predict_proba(X_test_set)[:,1]
roc_auc_score(y_test_set, y_pred)
print("LogisticRegression :", roc_auc_score(y_test_set, y_pred))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)                # 0.7190124215159207
# rf.fit(X_resampled, y_resampled)      # 0.6841750535878548
y_pred_rf = rf.predict_proba(X_test_set)[:, 1]
roc_auc_score(y_test_set, y_pred_rf)
print("RandomForestClassifier :", roc_auc_score(y_test_set, y_pred_rf))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
lda.fit(X_train, y_train)               # 0.7211824305608072
# lda.fit(X_resampled, y_resampled)     # 0.7209399233541647
y_pred_lda = lda.predict_proba(X_test_set)[:,1]
roc_auc_score(y_test_set, y_pred_lda)
print("LinearDiscriminantAnalysis :", roc_auc_score(y_test_set, y_pred_lda))

my_submission = pd.DataFrame({'SK_ID_CURR': test.SK_ID_CURR, 'TARGET': y_pred})
my_submission.to_csv('submission_logreg.csv', index=False)