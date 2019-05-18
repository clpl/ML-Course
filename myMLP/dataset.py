from mnist import MNIST
import random

class mnist_dataset:
	def __init__(self, filePath = '../python-mnist/data'):
		#load dataset
		self.mndata = MNIST(filePath)
		self.train_images, self.train_labels = self.mndata.load_training()
		self.test_images, self.test_labels = self.mndata.load_testing()
	def get_batch(self, batch_size):

		#return next(self.mndata.load_training_in_batches(batch_size))

		# write by me
		start_index = 0
		while(True):
			end_index = start_index + batch_size
			if end_index > len(self.train_images):
				start_index = 0
				random.shuffle(self.train_images)
			X = self.train_images[start_index:end_index]
			Y = self.train_labels[start_index:end_index]
			yield (X, Y)

	def get_test(self):
		return self.test_images, self.test_labels
