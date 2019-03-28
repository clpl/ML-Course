

def sigmoid(x, mode = 'F'):
	if mode != 'F':
		return sigmoid(x) * (1.0 - sigmoid(x))
	return 1.0 / (1.0 + np.exp(-x))

def tanh(x, mode = 'F'):
	if mode != 'F':
		return (1.0 - x * x)
	return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))