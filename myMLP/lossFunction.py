import numpy as np


# L2 loss
def MSE_loss(Y_P, Y):

	delta_C = np.square(Y_P - Y)
	loss_sum = delta_C.sum()
	
	return loss_sum, delta_C

def d_MSE_loss(a, y):
    return (a - y)

def CrossEntropyLoss(yi, yi_hat):
    return -np.sum(yi * np.log(yi_hat), axis = 1)

def d_CrossEntropyLoss(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))