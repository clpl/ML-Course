from sklearn.datasets import load_boston
import numpy as np

pattern_size = 13

def softmax(x):
	return 1 / (1 + np.exp(-x))

def d_softmax(x):
	return softmax(x) * (1 - softmax(x))

def loss(a, y):
    return (-y * np.log(a) - (1 - y) * np.log(1 - a)).mean()

def LR_init():
	W = np.random.uniform(0,0.1,size=[13])
	#W = np.ones(13)
	return W

def LR_learn(X, Y, w):
	learning_rate = 0.000001
	epoch_num = 500

	w = np.ones(13)
	b = 0.0
	for epoch in range(epoch_num):
		x = X
		y = Y
		a = np.dot(x, w) + b
		
		update = (a - y).mean()
		b -= learning_rate * update
		print(update)
		# print('update:',update)
		# print('a:',a)
		# print('y:',y)
		# print(x.T.mean(axis = 1).shape)
		# exit()
		w -= learning_rate * np.multiply(x.T.mean(axis = 1),update)
		# print('w:',w)
	for index in range(10):
		x = X[index]
		y = Y[index]
		a = np.dot(x, w) + b
		print(a, y)

		
	
# y = w1x1 + w2x2 + w3x3

# x1 x2 x3


def LRs_predict(x, w):

	return y



def main():
	x, y = load_boston(return_X_y = True)
	#print(x[0])
	W = LR_init()
	W = LR_learn(x, y, W)

if __name__ == '__main__':
	main()