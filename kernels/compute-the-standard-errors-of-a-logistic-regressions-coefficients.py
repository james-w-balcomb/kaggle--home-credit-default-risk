
# https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients

# The standard errors of the model coefficients are the square roots of the diagonal entries of the covariance matrix.
# Consider the following:
# Design matrix:

# X = ⎡1 x_1,1 ... x_1,p⎤
#     ⎢1 x_2,1 ... x_2,p⎥
#     ⎢. .     .   .    ⎥
#     ⎢. .      .  .    ⎥
#     ⎢. .       . .    ⎥
#     ⎣1 x_n,1 ... x_n,p⎦
# where x_i,j is the value of the jth predictor for the ith observations.
# (NOTE: This assumes a model with an intercept.)

# V = ⎡pihat_1(1 - pihat_1)            0           ...            0         ⎤
#     ⎢          0           pihat_2(1 - pihat_2)  ...            0         ⎥
#     ⎢          .                     .           .              .         ⎥
#     ⎢          .                     .            .             .         ⎥
#     ⎢          .                     .             .            .         ⎥
#     ⎣          0                     0           ...  pihat_n(1 - pihat_n)⎦
# where pihat_i represents the predicted probability of class membership for observation i.

# The covariance matrix can be written as:
# ((X^T)(V)(X))^−1

# This can be implemented with the following code:

import numpy
import sklearn

# Initiate logistic regression object
logit = sklearn.linear_model.LogisticRegression()

# Fit model. Let X_train = matrix of predictors, y_train = matrix of variable.
# NOTE: Do not include a column for the intercept when fitting the model.
resLogit = logit.fit(X_train, y_train)

# Calculate matrix of predicted class probabilities. 
# Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
predProbs = numpy.matrix(resLogit.predict_proba(X_train))

# Design matrix -- add column of 1's at the beginning of your X_train matrix
X_design = numpy.hstack((numpy.ones(shape = (X_train.shape[0],1)), X))

# Initiate Matrix of 0's, fill diagonal with each predicted observation's variance
V = numpy.matrix(numpy.zeros(shape = (X_design.shape[0], X_design.shape[0])))
numpy.fill_diagonal(V, numpy.multiply(predProbs[:,0], predProbs[:,1]).A1)

# Covariance matrix
covLogit = numpy.linalg.inv(X_design.T * V * X_design)
print('Covariance matrix: ', covLogit)

# Standard errors
print('Standard errors: ', numpy.sqrt(np.diag(covLogit)))

# Wald statistic (coefficient / s.e.) ^ 2
logitParams = numpy.insert(resLogit.coef_, 0, resLogit.intercept_)
print('Wald statistics: ', (logitParams / numpy.sqrt(numpy.diag(covLogit))) ** 2)
