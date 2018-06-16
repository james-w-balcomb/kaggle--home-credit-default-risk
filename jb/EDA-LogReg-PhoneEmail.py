
# coding: utf-8

# In[1]:


#%run NB01-Load.ipynb
#%run NB02-EDA-MetaData.ipynb
get_ipython().magic('run NB03-EDA-MetaData-Check.ipynb')


# In[19]:


df['FLAG_CONT_MOBILE'].value_counts(dropna=False, sort=True)


# In[20]:


df['FLAG_EMAIL'].value_counts(dropna=False, sort=True)


# In[21]:


df['FLAG_EMP_PHONE'].value_counts(dropna=False, sort=True)


# In[22]:


df['FLAG_MOBIL'].value_counts(dropna=False, sort=True)


# In[23]:


df['FLAG_PHONE'].value_counts(dropna=False, sort=True)


# In[24]:


df['FLAG_WORK_PHONE'].value_counts(dropna=False, sort=True)


# In[2]:


y = df['TARGET']


# In[18]:


X = df.loc[:, ['FLAG_CONT_MOBILE',
               'FLAG_EMAIL',
               'FLAG_EMP_PHONE',
               'FLAG_MOBIL',
               'FLAG_PHONE',
               'FLAG_WORK_PHONE']]


# In[4]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[5]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
logreg = LogisticRegression(class_weight='balanced', random_state=seed)
logreg.fit(X_train, y_train)


# In[7]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[8]:


kfold = model_selection.KFold(n_splits=10, random_state=seed)
modelCV = LogisticRegression(class_weight='balanced')
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[9]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# TP, FP
# FN, TN


# In[10]:


print(classification_report(y_test, y_pred))


# In[11]:


logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
#logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Multicollinearity
# 'FLAG_CONT_MOBILE',
# 'FLAG_EMAIL'
# 'FLAG_EMP_PHONE',
# 'FLAG_MOBIL',
# 'FLAG_PHONE',
# 'FLAG_WORK_PHONE',
# 
# #1 FLAG_CONT_MOBILE ~ 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'
# #2 FLAG_EMAIL ~ 'FLAG_CONT_MOBILE', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'
# #3 FLAG_EMP_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'
# #4 FLAG_MOBIL ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_PHONE', 'FLAG_WORK_PHONE'
# #5 FLAG_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE'
# #6 FLAG_WORK_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE'
# 

# #### 1 FLAG_CONT_MOBILE ~ 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'

# In[25]:


y = df['FLAG_CONT_MOBILE']


# In[26]:


X = df.loc[:, ['FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE']]


# In[27]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[28]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# #### 2 FLAG_EMAIL ~ 'FLAG_CONT_MOBILE', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'

# In[29]:


y = df['FLAG_EMAIL']


# In[30]:


X = df.loc[:, ['FLAG_CONT_MOBILE', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE']]


# In[31]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[32]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# #### 3 FLAG_EMP_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE'

# In[33]:


y = df['FLAG_EMP_PHONE']


# In[34]:


X = df.loc[:, ['FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE']]


# In[35]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[36]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# #### 4 FLAG_MOBIL ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_PHONE', 'FLAG_WORK_PHONE'

# In[37]:


y = df['FLAG_MOBIL']


# In[38]:


X = df.loc[:, ['FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_PHONE', 'FLAG_WORK_PHONE']]


# In[39]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[40]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# #### 5 FLAG_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE'

# In[41]:


y = df['FLAG_PHONE']


# In[42]:


X = df.loc[:, ['FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE']]


# In[43]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[44]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


# #### 6 FLAG_WORK_PHONE ~ 'FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE'

# In[45]:


y = df['FLAG_WORK_PHONE']


# In[46]:


X = df.loc[:, ['FLAG_CONT_MOBILE', 'FLAG_EMAIL' 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE']]


# In[47]:


logreg = LogisticRegression(class_weight='balanced', random_state=seed)


# In[48]:


logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())

