# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle
import pandas as pd


diabetes = pd.read_pickle('F:/ML/Project 1/basecode/diabetes.pickle')


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    w=np.multiply(inv(np.multiply(X.transpose(),X)),np.multiply(X.transpose(),y))
    #w = np.zeros((X.shape[0],1))
    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    a=np.multiply(w.transpose(),Xtest)
    b=np.subtract(ytest,a)
    c=np.multiply(b,b)
    d=np.sum(c)
    e=np.divide(d,len(Xtest))
    rmse = np.sqrt(e)
    return rmse


# problem 1
# create variables with respect to the dataset
train_x = np.asarray(diabetes[0])
train_y = np.asarray(diabetes[1])
test_x = np.asarray(diabetes[2])
test_y = np.asarray(diabetes[3])


#add intercept
x1 = np.ones((len(train_x),1))
x2 = np.ones((len(test_x),1))
Xtrain_i = np.concatenate((np.ones((train_x.shape[0],1)), train_x), axis=1)
Xtest_i = np.concatenate((np.ones((test_x.shape[0],1)), test_x), axis=1)

# transpose test data so that the colume of test data can match up with weight vector
xtrain_t=np.transpose(Xtrain_i)
xtest_t = np.transpose(Xtest_i)