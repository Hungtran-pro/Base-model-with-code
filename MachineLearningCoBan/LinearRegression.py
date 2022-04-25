from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

## LINEAR REGRESSION WITHOU SKLEARN
# #height
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# #weight
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# #visualize data
# plt.plot(X, y, 'ro', markerSize = 4)
# plt.axis([140, 190, 40, 70])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

# # Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# # Calculating weights of the fitting line 
# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, y)
# w = np.dot(np.linalg.pinv(A), b)
# print('w = ', w)
# # Preparing the fitting line 
# w_0 = w[0][0]
# w_1 = w[1][0]
# x0 = np.linspace(145, 185, num=100) #num ~ number of samples generates
# print(x0)
# y0 = w_0 + w_1*x0
# # print(np.linspace(1, 10, 1))
# plt.plot(X.T, y.T, 'ro')    #data 
# plt.plot(x0, y0, '-g')            #the fitting line
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

## USING SKLEARN
from sklearn import datasets, linear_model

#fit the model by LR.
regr = linear_model.LinearRegression(fit_intercept=False) #fit_intercept = False -> calculate bias
regr.fit(Xbar, y)
# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )