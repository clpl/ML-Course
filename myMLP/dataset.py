from mnist import MNIST


class mnist_dataset:
	def __init__(self, filePath = '../python-mnist/data'):
		#load dataset
		self.mndata = MNIST(filePath)
		self.train_images, self.train_labels = mndata.load_training()
		self.test_images, self.test_labels = mndata.load_testing()
	def get_batch(self, batch_size):

		return self.mndata.load_training_in_batches(batch_size).next()

		# # write by me
		# start_index = 0
		# while(True):
		# 	end_index = start_index + batch_size
		# 	if end_index > len(train_images):
		# 		start_index = 0
		# 	X = train_images[start_index:end_index]
		# 	Y = train_labels[start_index:end_index]
		# 	yield (X, Y)
