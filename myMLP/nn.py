#coding:utf-8

import numpy as np
from activationFunction import *
from lossFunction import *

#load dataset
mndata = MNIST('../python-mnist/data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

class learning_rate:
	def __init__(self, learning_rate_start = 0.1, decay = 0.5):
		self.learning_rate_start = learning_rate_start
		self.learning_rate = self.learning_rate_start
		self.decay = decay
	def step(self):
		self.learning_rate = self.learning_rate_start * 1.0 / (1.0 + self.decay * self.learning_rate)
	def __mul__(self, scalar):
		return self.learning_rate * scalar
	def __rmul__(self, scalar):
		return self.__mul__(scalar)

class mlp:
	def __init__(self, learning_rate = 0.1):
		
		self.learning_rate = learning_rate


		np.random.seed(2020)

		self.w1 = 2.0*np.random.random((784, 300))-1.0
		self.b1 = 2.0*np.random.random((300,))-1.0
		self.w2 = 2.0*np.random.random((300, 10))-1.0
		self.b2 = 2.0*np.random.random((10,))-1.0

	def forward(self, X, actF = sigmoid):
		z1 = np.dot(X, self.w1) + self.b1
		
		a1 = sigmoid(z1)

		z2 = np.dot(a1, self.w2) + self.b2
		a2 = sigmoid(z2)

		return a2

	def fit(self, x_batch, y_batch):
		loss = 0.0
		for i, item in enumerate(x_batch):
			X = np.array(x_batch[i])/256.0
			Y = y_batch[i]
			#Y_P = self.forward(X)

			#forward
			z1 = np.dot(X, self.w1) + self.b1
			a1 = sigmoid(z1)
			z2 = np.dot(a1, self.w2) + self.b2
			a2 = sigmoid(z2)
			Y_P = a2

			#cal loss
			Y_vec = one_hot(10, Y)
			loss_t = MSE_loss(Y_P, Y_vec)
			loss += loss_t
			#print('\n\nepoch:',i, '\nloss:', loss_t,'\nz1:', z1[:5])
			
			#backward
			# 10 - 10 -> 10
			delta_C = a2 - Y_vec
			#print('C:',delta_C)
			
			# 10 * 10 -> 10
			delta_L = delta_C * sigmoid(z2, 'B')
						
			# 10 * 10x100 -> 10x100
			tmp = np.dot(delta_L, self.w2.T)

			delta_l1 = sigmoid(z1, 'B') * tmp
			
			self.b2 = self.b2 - self.learning_rate*delta_L
			self.b1 = self.b1 - self.learning_rate*delta_l1

			self.w2 = self.w2 - self.learning_rate*np.dot(a1.reshape(a1.shape[0],1), delta_L.reshape(1,delta_L.shape[0]))
			self.w1 = self.w1 - self.learning_rate*np.dot(X.reshape(X.shape[0],1), delta_l1.reshape(1,delta_l1.shape[0]))
		print(loss)
		
			

def gen_batch(batch_size):
	start_index = 0
	while(True):
		end_index = start_index + batch_size
		if end_index > len(train_images):
			start_index = 0
		X = train_images[start_index:end_index]
		Y = train_labels[start_index:end_index]
		yield (X, Y)
	
mlp = mlp()
def main():
	max_epoch = 5000
	batch_size = 50

	for epoch in range(max_epoch):
		X, Y = next(gen_batch(batch_size))
		mlp.fit(X,Y)
		#break



if __name__ == '__main__':
	main()