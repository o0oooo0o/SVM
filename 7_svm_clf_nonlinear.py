# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:22:59 2019

@author: 321
"""

from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

#data
x,y = make_moons(n_samples=100, noise=0.15, random_state=42)

n=1000
x1 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), n)
x2 = np.linspace(np.min(x[:,1]), np.max(x[:,0]), n)
X1, X2 = np.meshgrid(x1, x2)
XX = np.hstack((np.reshape(X1, (n*n,1)), np.reshape(X2, (n*n,1))))

#training
svm1 = SVC(kernel='poly', degree=3, C=5, coef0=1).fit(x,y)
predict1 = svm1.predict(XX)
predict1 = np.reshape(predict1, (n,n))

svm2 = SVC(kernel='rbf', degree=3, C=100, gamma=0.1).fit(x,y)
predict2 = svm2.predict(XX)
predict2 = np.reshape(predict2, (n,n))

plt.subplots(1,2, figsize=(12,4))
plt.subplot(121)
plt.title("sklearn kernelSVM - poly")
plt.scatter(x[y==1][:,0], x[y==1][:,1], color='c')
plt.scatter(x[y==0][:,0], x[y==0][:,1], color='m')
plt.ylim(-1, 1.5)
plt.contour(X1, X2, predict1, 0, colors='k', linewidth=.1)

plt.subplot(122)
plt.title("sklearn kernelSVM - rbf")
plt.scatter(x[y==1][:,0], x[y==1][:,1], color='c')
plt.scatter(x[y==0][:,0], x[y==0][:,1], color='m')
plt.ylim(-1, 1.5)
plt.contour(X1, X2, predict2, 0, colors='k', linewidth=.1)
plt.show()