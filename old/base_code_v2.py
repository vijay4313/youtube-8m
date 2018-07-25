
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf # pylint: disable=protected-access 

from tensorflow import logging
from tensorflow import gfile
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

# Cloud TPU Cluster Resolvers
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
# Model specific paramenters
flags.DEFINE_integer("batch_size", 2048,
                     "Mini-batch size for the computation. Note that this "
                     "is the global batch size and not the per-shard batch.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
flags.DEFINE_string("train_file", "gs://youtube8m-ml-us-east1/2/video/train/train*.tfrecord", "Path to cifar10 training data.")
flags.DEFINE_integer("train_steps", 100000,
                     "Total number of steps. Note that the actual number of "
                     "steps is the next multiple of --iterations greater "
                     "than this value.")
flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_string("model_dir", "gs://memory-box-1", "Estimator model_dir")
flags.DEFINE_integer("iterations_per_loop", 10,
                     "Number of iterations per TPU training loop.")
flags.DEFINE_integer("num_shards", 16, "Number of shards (TPU chips).")


FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
  """Define a CIFAR model in Keras."""
  del params  # unused
  #layers = tf.keras.layers
  #models = tf.keras.models


  # Pass our input tensor to initialize the Keras input layer.
  #mdl = layers.Input(tensor=features)
  #mdl = layers.Dense(2048, activation="relu")(mdl)
  #op = layers.Dense(3862, activation="softmax")(features)
  
  op = tf.contrib.layers.fully_connected(inputs = features, num_outputs = 3862, activation_fn = tf.nn.softmax)

  # Instead of constructing a Keras model for training, build our loss function
  # and optimizer in Tensorflow.
  #
  # N.B.  This construction omits some features that are important for more
  # complex models (e.g. regularization, batch-norm).  Once
  # `model_to_estimator` support is added for TPUs, it should be used instead.
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=op, labels=tf.cast(labels, tf.float32)
      )
  )
  optimizer = tf.train.AdamOptimizer(0.01)
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions={
          "probabilities": op
      }
  )



def input_fn(params):
  """Read CIFAR input data from a TFRecord dataset."""
  del params
  batch_size = FLAGS.batch_size
  num_readers= 1
  num_classes = 3862
  

  def parser(serialized_examples):
    # set the mapping from the fields to data types in the proto

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64),
                   "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
                   "mean_audio": tf.FixedLenFeature([128], tf.float32)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], num_classes)
    labels.set_shape([num_classes, ])
    concatenated_features = tf.concat([features["mean_rgb"], features["mean_audio"]], axis=0)

    return concatenated_features, labels



  files = gfile.Glob(FLAGS.train_file)

  if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                  FLAGS.train_file + "'.")

  logging.info("Number of training files: %s.", str(len(files)))
  #filename_queue = tf.train.string_input_producer(files, shuffle=True)

  dataset = tf.data.TFRecordDataset(files, num_parallel_reads = FLAGS.num_shards)
  dataset = dataset.map(parser, num_parallel_calls = FLAGS.num_shards)
  dataset = dataset.prefetch(4 * FLAGS.batch_size).cache()

  #dataset = dataset.shard(FLAGS.num_shards)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
  dataset.prefetch(FLAGS.num_shards)
  

    # Make input_fn for the TPUEstimator train step
  #logging.info("Created Dataset with shape: %s", str(dataset.get_shape()))
  
  return dataset


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