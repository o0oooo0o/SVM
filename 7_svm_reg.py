# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:51:40 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR, SVR
n = 30
x=np.linspace(-3,3,n)
x=np.reshape(x, (-1,1))
s = 3*x + 3
s=s.reshape(n,1)
num =2*np.random.randn(n,1)
y =s+num

x_new = np.linspace(-3,3,n)
x_new=np.reshape(x_new, (-1,1))

svm1 = LinearSVR(epsilon=1.5).fit(x,y)
predict1 = svm1.predict(x_new)


ss = 3*(x**2) - 5
ss = ss.reshape(n,1)
yy = ss+num

svm2 = SVR(kernel='poly', degree=2, C=100, epsilon=0.1).fit(x,yy)
predict2 = svm2.predict(x_new)

svm3 = SVR(kernel='rbf', degree=2, C=10, gamma=0.1).fit(x,yy)
predict3 = svm3.predict(x_new)

plt.subplots(1,3, figsize=(15,4))
plt.subplot(131)
plt.title("linear SVM reg")
plt.scatter(x,y, color ='c')
plt.ylim(-9, 16)
plt.plot(x_new, predict1, color='k')

plt.subplot(132)
plt.title("kernel(poly) SVM reg")
plt.scatter(x,yy, color = 'c')
plt.ylim(-9, 16)
plt.plot(x_new, predict2, color='k')

plt.subplot(133)
plt.title("kernel(rbf) SVM reg")
plt.scatter(x,yy, color = 'c')
plt.ylim(-9, 16)
plt.plot(x_new, predict3, color='k')
plt.show()