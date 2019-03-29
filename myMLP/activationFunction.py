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

def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.float)