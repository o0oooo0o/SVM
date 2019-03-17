# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:56:45 2019

@author: 321
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import LinearSVC

#data
iris = datasets.load_iris()
x = iris["data"][:100,(2,3)]
y = (iris["target"][:100] == 1).astype(np.float64)

clf = LinearSVC(C = 10).fit(x,y)

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
x = np.hstack([x, -np.ones((len(x),1))])

n=1000
x1 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), n)
x2 = np.linspace(np.min(x[:,1]), np.max(x[:,0]), n)
X1, X2 = np.meshgrid(x1, x2)
XX = np.hstack((np.reshape(X1, (n*n,1)), np.reshape(X2, (n*n,1))))

predict = clf.predict(XX)
predict = np.reshape(predict, (n,n))


XX = np.hstack((XX, -np.ones((n*n,1))))
l_rate = 0.001
def svm_function(x,y):
    
    w = np.zeros(len(x[0]))
    epoch = 100000
    out = []
    
    #training svm
    for e in range(epoch):
        for i, val in enumerate(x):
            val1 = np.dot(x[i], w)
            if (y[i]*val1 < 1):
                w = w + l_rate * ((y[i]*x[i]) - (2*(1/epoch)*w))
            else:
                w = w + l_rate * (-2*(1/epoch)*w)
    
    for i, val in enumerate(XX):
        out.append(np.dot(XX[i], w))
    
    out = np.array(out).reshape(n,n)
    return w, out

w, out = svm_function(x, y)

        
plt.subplots(1,2, figsize=(12,4))
plt.subplot(121)
plt.title("SVM sklearn")
plt.scatter(x[:50,0], x[:50,1], color='c')
plt.scatter(x[50:,0], x[50:,1], color='m')
plt.contour(X1, X2, predict, 0, colors='k', linewidth=.1)
plt.ylim(0, 2)

plt.subplot(122)
for val, inp in enumerate(x):
    if y[val] == -1:
        plt.scatter(inp[0], inp[1], color='c', marker='.', linewidths=5)
    else:
        plt.scatter(inp[0], inp[1], color='m', marker='.', linewidths=5)
        
plt.title("SVM numpy learning_rate={}".format(l_rate))
plt.contour(X1,X2, out, 0, colors='k')
plt.ylim(0, 2)
plt.show()
