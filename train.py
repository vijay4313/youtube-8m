



from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf # pylint: disable=protected-access 

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

from . import train_utils
from . import data_reader
from . import video_level_models as models

if __name__ == '__main__':

	# Connection Params
	flags.DEFINE_string(
	    'tpu', default=None,
	    help='The Cloud TPU to use for training. This should be either the name '
	    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
	
	flags.DEFINE_string(
	    "gcp_project", default=None,
	    help="Project name for the Cloud TPU-enabled project. If not specified, we "
	    "will attempt to automatically detect the GCE project from metadata.")
	
	flags.DEFINE_string(
	    "tpu_zone", default=None,
	    help="GCE zone where the Cloud TPU is located in. If not specified, we "
	    "will attempt to automatically detect the GCE project from metadata.")

	flags.DEFINE_integer("num_shards", 16, "Number of shards (TPU chips).")

	flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
	
	flags.DEFINE_string("model_dir", "gs://memory-box-1", "Estimator model_dir")

	flags.DEFINE_string("train_file", "gs://youtube8m-ml-us-east1/2/video/train/train*.tfrecord", "Path to cifar10 training data.")
	

	# Training Params/Hyperparams

	flags.DEFINE_string("training_model", "Logistic", "Model to be used for training")

	flags.DEFINE_integer("num_classes", 3862, "Number of output classes")

	flags.DEFINE_integer("batch_size", 1024,
	                     "Mini-batch size for the computation. Note that this "
	                     "is the global batch size and not the per-shard batch.")
	
	flags.DEFINE_integer("train_steps", 100000,
	                     "Total number of steps. Note that the actual number of "
	                     "steps is the next multiple of --iterations greater "
	                     "than this value.")

	flags.DEFINE_integer("iterations_per_loop", 10,
	                     "Number of iterations per TPU training loop.")

	flags.DEFINE_string("optimizer_fn", "Adam", "Type of Optimizer for learning")

	flags.DEFINE_string("loss_fn", "sfmax_cross_entropy", "Type of loss function for learning")

	flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

	flags.DEFINE_bool("learning_rate_decay", False, "Use learning rate decay/not")

	flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")

	flags.DEFINE_float("learning_rate_decay_val", 0.9,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")

	FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):

	_optimizer_fn = train_utils.modelParams.optimizer_fn(train_utils.optimizer, FLAGS.learning_rate)

	if FLAGS.use_tpu:
		_optimizer_fn = tpu_optimizer.CrossShardOptimizer(_optimizer_fn)

	_loss_fn = train_utils.modelParams.loss_fn(train_utils.loss_fn, FLAGS.loss_fn)

	_model_graph = train_utils.modelParams.get_model(models, FLAGS.training_model)

	if mode == tf.estimator.ModeKeys.TRAIN:
		_logits = _model_graph(features, FLAGS.num_classes)

		_loss = _loss_fn(_logits, labels)

		_train_op = _optimizer_fn.minimize(loss, global_step=tf.train.get_global_step())

		return tpu_estimator.TPUEstimatorSpec(
			mode=mode,
      		loss=_loss,
      		train_op=_train_op,
      		predictions={
      			"class": tf.argmax(_logits, axis = -1)
      			"probabilities": _logits
      		}
      		)


def input_fn():

	files = gfile.Glob(FLAGS.train_file)

	if not files:
	raise IOError("Unable to find training files. data_pattern='" +
	              FLAGS.train_file + "'.")

	logging.info("Number of training files: %s.", str(len(files)))
	#filename_queue = tf.train.string_input_producer(files, shuffle=True)

	map_func = data_reader.VideoLevelReader(FLAGS.num_classes).parser

	dataset = tf.data.TFRecordDataset(files, num_parallel_reads = FLAGS.num_shards)
	dataset = dataset.map(map_func, num_parallel_calls = FLAGS.num_shards)
	dataset = dataset.prefetch(4 * FLAGS.batch_size).cache()

	#dataset = dataset.shard(FLAGS.num_shards)
	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
	dataset.prefetch(FLAGS.num_shards)



def main(argv):
	del argv  # Unused.

	if FLAGS.use_tpu:
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
	  FLAGS.tpu,
	  zone=FLAGS.tpu_zone,
	  project=FLAGS.gcp_project)
	tpu_grpc_url = tpu_cluster_resolver.get_master()

	else:
	tpu_grpc_url = None

	run_config = tpu_config.RunConfig(
	  master=tpu_grpc_url,
	  model_dir=FLAGS.model_dir,
	  session_config=tf.ConfigProto(
	      allow_soft_placement=True, log_device_placement=True),
	  tpu_config=tpu_config.TPUConfig(
	      iterations_per_loop=FLAGS.iterations_per_loop),
	)

	estimator = tpu_estimator.TPUEstimator(
	  model_fn=model_fn,
	  use_tpu=FLAGS.use_tpu,
	  config=run_config,
	  train_batch_size=FLAGS.batch_size)
	estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)