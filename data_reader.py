
import tensorflow as tf

class VideoLevelReader(classes):

	def __init__():
		self.classes = classes


	def parser(self, serialized_examples):
	    # set the mapping from the fields to data types in the proto

	    feature_map = {"id": tf.FixedLenFeature([], tf.string),
	                   "labels": tf.VarLenFeature(tf.int64),
	                   "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
	                   "mean_audio": tf.FixedLenFeature([128], tf.float32)}

	    features = tf.parse_single_example(serialized_examples, features=feature_map)
	    labels = tf.sparse_to_indicator(features["labels"], self.classes)
	    labels.set_shape([self.classes, ])
	    concatenated_features = tf.concat([features["mean_rgb"], features["mean_audio"]], axis=0)

	    return concatenated_features, labels

	