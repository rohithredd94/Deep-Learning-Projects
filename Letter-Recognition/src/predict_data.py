from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

#Problem 6, Running Logistic Regression

train_size = 200000
valid_size = 10000
test_size = 10000

with open('../notMNIST.pickle','rb') as f:
    final_dataset = pickle.load(f)

train_X = final_dataset['train_dataset'] #Logistic regression on all the samples
train_X = train_X.reshape(train_size, np.shape(train_X)[1]*np.shape(train_X)[2])
train_Y = final_dataset['train_labels']
print(train_X,train_Y)

test_X = final_dataset['test_dataset']
test_X = test_X.reshape(test_size, np.shape(test_X)[1]*np.shape(test_X)[2])
test_Y = final_dataset['test_labels']
print(test_X,test_Y)

#Off the shelf classifier is Logistic Regression
logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver='lbfgs')

logreg.fit(train_X, train_Y)

logreg.get_params()
logreg.coef_
logreg.score(test_X,test_Y)

print('Number of images in test set: %d' % test_size)
print(
  'Number of correctly predicted images in test set: %d'
  % sum(np.equal(logreg.predict(test_X), test_Y)))
print(
  '%.2f%% of images in test set correctly predicted' 
  % round(np.float32(sum(np.equal(logreg.predict(test_X), test_Y)))/test_size, 2)
)
