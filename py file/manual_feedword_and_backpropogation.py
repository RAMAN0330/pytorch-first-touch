import numpy as np
import torch

"""Defining Activation function"""

def sigmoid_scaler(x):
  return 1/(1+np.exp(-x))
sigmoid  = np.vectorize(sigmoid_scaler)    # np.vectorize() --->  to convert matrix into array

def softmax(x):
  return np.exp(x)/np.exp(x).sum()

def log_loss(y,y_hat):
  return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def l2_norm(y,y_hat):
  return np.sum((y-y_hat)**2)

weight1 = np.array([[.1,.3,.8,-.4],[-.3,-.2,.5,.5],[-.3,0,.5,.4],[.2,.5,-.9,.7]])
X = np.array([2,5,3,3])
bias1 = np.zeros(4)
a1 = np.matmul(weight1,X) + bias1
h1 = sigmoid(a1)
print(h1.round(2))

"""Feed-forword hidden layer"""
weight2 = np.array([[.5,.8,.2,.4],[.5,.2,.3,-.5]])
bias2 = np.zeros(2)
a2 =  np.matmul(weight2,h1) + bias2
h2 = softmax(a2)
print(h2.round(2))

"""Calculating Loss using LogLoss Function"""
y = np.array([1,0])
loss_logloss = log_loss(y,h2)
loss_l2norm = l2_norm(y,h2)
print("Log Loss ",loss_logloss.round(2))
print("L2 norm  ",loss_l2norm.round(2))

"""Backpropogation for Hidden layer"""
d_l2norm = 2*(h2-y)
d_softmax = np.diag(h2) - np.outer(h2,h2)
d_a2 = np.matmul(d_softmax,d_l2norm)
d_w2 = np.outer(d_a2,h1)
d_b2 = d_a2
d_h1 = np.matmul(weight2.T,d_a2)

"""Backpropogation for Input layer"""
d_sigmoid = h1*(1-h1)
d_a1 = d_h1*d_sigmoid
d_w1 = np.outer(d_a1,X)
d_b1 = d_a1

"""Updating Weights and Biases"""
learning_rate = 0.1
weight1 = weight1 - learning_rate*d_w1
weight2 = weight2 - learning_rate*d_w2
bias1 = bias1 - learning_rate*d_b1
bias2 = bias2 - learning_rate*d_b2

print("Updated Weights for Input Layer:\n", weight1.round(2))
print("Updated Weights for Hidden Layer:\n", weight2.round(2))
print("Updated Biases for Input Layer:\n", bias1.round(2))
print("Updated Biases for Hidden Layer:\n", bias2.round(2))

