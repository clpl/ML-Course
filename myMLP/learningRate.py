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