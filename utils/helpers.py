import os
import tensorflow as tf


def assembly_sets_path(FLAGS):
    if FLAGS.force_splits_dir_path:
        sets_path = FLAGS.sets_dir
    else:
        sets_path = os.path.join(FLAGS.sets_dir, FLAGS.split_number, FLAGS.split_type, '{}_fps'.format(FLAGS.sample_rate) )
    return sets_path

def assembly_snippets_path(FLAGS):
    if FLAGS.force_splits_dir_path:
        snippets_path = FLAGS.snippets_dir
    else:
        snippets_path = os.path.join(FLAGS.sets_dir, FLAGS.split_number, FLAGS.split_type, '{}_fps'.format(FLAGS.sample_rate), 'w_{}_l_{}'.format(FLAGS.snippet_width, FLAGS.snippet_size) )
    return snippets_path

def assembly_model_dir(FLAGS):
    model_dir_path = os.path.join(FLAGS.model_dir, FLAGS.model_name, FLAGS.experiment_tag)
    return model_dir_path 

def check_and_create_directories(FLAGS):
    directories = [assembly_model_dir(FLAGS)]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_splits(FLAGS):
    network_training_set = []
    network_validation_set = []

    with open(os.path.join(assembly_sets_path(FLAGS), 'network_training_set.txt'), 'r') as f:
        network_training_set = f.read().split('\n')[:-1]
    with open(os.path.join(assembly_sets_path(FLAGS), 'network_validation_set.txt'), 'r') as f:
        network_validation_set = f.read().split('\n')[:-1]

    return network_training_set, network_validation_set


def configure_optimizer(FLAGS, learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def configure_learning_rate(FLAGS, num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
#   if FLAGS.sync_replicas:
#     decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=False,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     0.0,
#                                     FLAGS.end_learning_rate,
#                                     power=1.0,
                                     power=0.5,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)