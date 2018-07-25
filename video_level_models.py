

import tensorflow as tf


class Logistic(features, classes):

	def __init__():
		self.features = features
		self.classes = classes

		model = tf.contrib.layers.fully_connected(inputs = self.features, 
			num_outputs = self.classes, 
			activation_fn = tf.nn.softmax)

		return model








