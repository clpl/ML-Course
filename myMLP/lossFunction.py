
# L1 loss
def MSE_loss(Y_P, Y):

	loss_sum = np.square(Y_P - Y).sum()
	
	return loss_sum / (2 * len(Y))