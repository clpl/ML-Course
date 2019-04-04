#coding:utf-8

import numpy as np
import argparse
from activationFunction import *
from lossFunction import *
from dataset import mnist_dataset




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

		for i in range(len(self.layer_size) - 1):
			layer_shape = tuple(self.layer_size[i:i+2])
			layer_w = self._random_init(layer_shape, bound)
			w_list.append(layer_w)
			layer_b = self._random_init(self.layer_size[i+1], bound)
			b_list.append(layer_b)

		return w_list, b_list

	def _random_init(self, shape, bound):
		return np.random.uniform(-bound,bound,size=shape)

	def forward(self, input_data, actF = relu):
		z_list = []
		a_list = [input_data]
		for i in range(len(self.layer_size) - 1):
			pre_a = a_list[i]
			z = np.dot(pre_a, self.w[i]) + np.tile(self.b[i],(pre_a.shape[0], 1))
			a = actF(z)

			#store 'z' and 'a'
			z_list.append(z)
			a_list.append(a)

		output_layer = softmax(a_list[-1])

		# z, a, output_layer
		return z_list, a_list, output_layer

	def backward(self, Y, z_list, a_list, d_act = d_relu):
		batch_size = Y.shape[0]
		#L2 loss
		delta_C = (a_list[-1] - Y)
		delta_L = delta_C * d_act(z_list[-1])

		delta_list = [delta_L]
		delta_b_list = []
		delta_w_list = []
		for i in range(len(self.layer_size) - 2, -1, -1):
			delta_l_plus_one = delta_list[-1]
			delta_b_list.append(delta_l_plus_one)

			delta_w = np.empty((batch_size, a_list[i].shape[1], delta_l_plus_one.shape[1]))

			for index in range(batch_size):
				delta_w[index,:,:] = np.outer(a_list[i][index,:], delta_l_plus_one[index,:])
				
			delta_w_list.append(delta_w)
			
			if i == 0:
				break
			delta_l = np.multiply(np.dot(self.w[i], delta_l_plus_one.T), d_act(z_list[i-1].T))
			delta_list.append(delta_l.T)
			
			
		delta_b_list = list(reversed(delta_b_list))
		delta_w_list = list(reversed(delta_w_list))
		for i in range(0, len(self.layer_size) - 1):
			self.w[i] -= self.learning_rate * delta_w_list[i].mean(axis = 0)
			self.b[i] -= self.learning_rate * delta_b_list[i].mean(axis = 0)


	def predict(self, x):
		x = np.array(x)
		_, _, y_p = self.forward(x)
		
		y_p = np.argmax(y_p, axis = 1)
		return y_p
		

	def fit(self, x_batch, y_batch):
		loss = 0.0
		
		X = np.array(x_batch)/256.0
		Y = np.array(y_batch)
		Y = make_one_hot(Y)
	
		z_list, a_list, output_layer = self.forward(X)
		# # MSE loss
		# loss, C = MSE_loss(output_layer, Y)
		loss = CrossEntropyLoss(Y, output_layer)

		self.backward(Y, z_list, a_list)

		return loss.sum()

			

def eval_model(model, x, y):
	y_p = model.predict(x)
	y = np.array(y)
	acc = np.array([y == y_p]).sum()
	
	return acc/y.shape[0] * 100
	

#load dataset
mndata = mnist_dataset('../python-mnist/data')		




def read_arg():
	parser = argparse.ArgumentParser()
	parser.add_argument("--max_epoch", type=int, default = 10000)
	parser.add_argument("--batch_size", type=int, default = 550)
	parser.add_argument("--score_per_epoch", type=int, default = 50)
	parser.add_argument("--loss_per_epoch", type=int, default = 25)
	parser.add_argument("--layer_szie", type=str, default = '784,300,10')
	parser.add_argument("--out_put_prefix", type=str, default = 'file')

	args = parser.parse_args()
	return args


args = read_arg()

max_epoch = args.max_epoch
batch_size = args.batch_size
score_per_epoch = args.score_per_epoch
loss_per_epoch = args.loss_per_epoch
layer_size = args.layer_szie.split(',')
layer_size = list(map(eval, layer_size))
file_prefix = args.out_put_prefix
mlp = mlp(layer_size = layer_size)

def main():

	file_loss = open(file_prefix + "_loss", 'w')
	file_score = open(file_prefix + "_score", "w")

	for epoch in range(max_epoch):
		X, Y = next(mndata.get_batch(batch_size))
		loss = mlp.fit(X,Y)
		if epoch % loss_per_epoch == 0:
			print('epoch:', epoch, 'loss:', loss)
			print(epoch, loss, file = file_loss)
		if epoch % score_per_epoch == 0:
			test_image, test_label = mndata.get_test()
			score = eval_model(mlp, test_image, test_label)
			print("\nscore:", score, '%\n')
			print(epoch, score, file = file_score)

	file_loss.close()
	file_score.close()



if __name__ == '__main__':
	main()