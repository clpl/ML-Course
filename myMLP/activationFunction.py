# activation Function of nerual
import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

def tanh(x, mode = 'F'):
	return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))

def d_tanh(x):
	return (1.0 - x * x)

def relu(x):
	return np.maximum(x, 0)

def d_relu(x):
	x[x <= 0] = 0
	x[x > 0] = 1
	return x

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))

def d_softmax(x):
	return softmax(x) * (1 - softmax(x))

def make_one_hot(data1):
	return (np.arange(10)==data1[:,None]).astype(np.float)