

import tensorflow as tf


def get_func_by_name(class_name, function):

	return getattr(class_name, function, None)


class optimizer(learning_rate):
	
	def __init__():
		self.learning_rate = learning_rate

	def adam(self):
		return tf.train.AdamOptimizer(self.learning_rate)

	def gradientdescent(self):
		return tf.train.GradientDescentOptimizer(self.learning_rate)

	def rmsprop(self):
		return tf.train.RMSPropOptimizer(self.learning_rate)


class loss_fn():

	def sfmax_cross_entropy(self, logits, labels):

		#Ensure both logits and labels are of same dtypes
		if labels.dtype == logits.dtype:
		else:
			labels = tf.cast(labels, logits.dtype)

		return tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(
				logits=logits, labels=labels))


class modelParams():

	def get_loss_fn(self, loss_fns, loss_fn):
		return get_func_by_name(loss_fns,loss_fn)

	def get_optimizer_fn(self, optimizers, optimizer_fn):
		return get_func_by_name(optimizers, optimizer_fn)

	def get_model(self, models, training_model):
		return get_func_by_name(models, training_model)

