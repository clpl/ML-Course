#coding:utf-8

import numpy as np
from activationFunction import *
from lossFunction import *
from dataset import mnist_dataset

#load dataset
mndata = mnist_dataset('../python-mnist/data')

class mlp:
	def __init__(self, layer_size = [784,100,10], learning_rate = 0.1):
		
		self.learning_rate = learning_rate
		self.layer_size = layer_size
		np.random.seed(2020)

		self.w, self.b = self._layer_init()



	def _layer_init(self):
		w_list = []
		b_list = []
		bound = 1.0

		for i in range(len(layer_size) - 1):
			layer_shape = tuple(layer_size[i:i+1])
			layer_w = self._random_init(layer_shape, bound)
			w_list.append(layer_w)
			layer_b = self._random_init(layer_size[1], bound)
			b_list.append(layer_b)

		return w_list, b_list

	def _random_init(self, shape, bound):
		return numpy.random.uniform(-bound,bound,size=shape)

	def forward(self, input_data, actF = sigmoid):
		z_list = []
		a_list = [input_data]
		for i in len(self.w):
			z = np.dot(self.w[i], a_list[i]) + self.b[i]
			a = actF(z)

			z_list.append(z)
			a_list.append(a)

		# z, a, output_layer
		return z_list, a_list, a_list[-1]

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
			
			delta_C = a2 - Y_vec
			delta_L = delta_C * sigmoid(z2, 'B')
						
			tmp = np.dot(delta_L, self.w2.T)

			delta_l1 = sigmoid(z1, 'B') * tmp
			
			self.b2 = self.b2 - self.learning_rate*delta_L
			self.b1 = self.b1 - self.learning_rate*delta_l1

			self.w2 = self.w2 - self.learning_rate*np.dot(a1.reshape(a1.shape[0],1), delta_L.reshape(1,delta_L.shape[0]))
			self.w1 = self.w1 - self.learning_rate*np.dot(X.reshape(X.shape[0],1), delta_l1.reshape(1,delta_l1.shape[0]))
		print(loss)
	
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