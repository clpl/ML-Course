import numpy as np


# L2 loss
def MSE_loss(Y_P, Y):

	delta_C = np.square(Y_P - Y)
	loss_sum = delta_C.sum()
	
	return loss_sum, delta_C


def CrossEntropyLoss(yi, yi_hat):
    '''
    compute Cross Entropy Loss for output yi
    type yi: predicted output
    type label_batch: ground truth
    '''
    return -np.sum(yi * np.log(yi_hat), axis = 1)