# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:37:29 2019

@author: 321
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

x_train[x_train > 0] = 1
x_test[x_test > 0] = 1
#pixel = np.array(x_train[0,:]).reshape(28,28)
#plt.imshow(pixel, cmap='gray')

y_train=np.where(y_train == 1)[1]
y_test=np.where(y_test == 1)[1]

classifier = svm.SVC(gamma=0.001)
classifier.fit(x_train, y_train)

#predicted = classifier.predict(x_test)
scores=classifier.score(x_test,y_test)
