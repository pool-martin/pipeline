import os
import tensorflow as tf

def define_flags():
    tf.app.flags.DEFINE_string(
        'model_name', 'i3d', 'The name of the cnn to train.')
    tf.app.flags.DEFINE_string(
        'model_dir', '/Exp/2kporn/checkpoints', 'path to save checkpoints.')
    tf.app.flags.DEFINE_string(
        'experiment_tag', 'initial_training', 'tag used in model dir places where we want to diferentiate subsequent executions.')
    tf.app.flags.DEFINE_string(
        'dataset_dir', '/DL/2kporn/', 'The sets to be used.')
    tf.app.flags.DEFINE_string(
        'sets_dir', '/Exp/2kporn/splits', 'The sets to be used.')
    tf.app.flags.DEFINE_string(
        'snippets_dir', '', 'The sets to be used.')
    tf.app.flags.DEFINE_boolean(
        'force_splits_dir_path', False, 'split and sets dirs will be assembled or received.')
    tf.app.flags.DEFINE_string(
        'split_number', 's1', 'Split number to be used.')
    tf.app.flags.DEFINE_string(
        'split_type', '3D', 'Set type to be used.')
    tf.app.flags.DEFINE_integer(
        'num_gpus', 2, 'The number of gpus that should be used')
    tf.app.flags.DEFINE_integer(
        'sample_rate', 5, 'sample rate of the dataset in fps')
    tf.app.flags.DEFINE_integer(
        'snippet_size', 32, 'The number of frames in the snippet')
    tf.app.flags.DEFINE_integer(
        'snippet_width', 5, 'The length in seconds the snippet should represent')
    tf.app.flags.DEFINE_integer(
        'batch_size', 8, 'The number of snippets in the batch')
    tf.app.flags.DEFINE_integer(
        'epochs', 2, 'The number of epochs to run the training')
    tf.app.flags.DEFINE_integer(
        'eval_interval_secs', 3600, 'Do not re-evaluate unless the last evaluation was started at least this many seconds ago.')
    tf.app.flags.DEFINE_list(
        'image_shape', [224, 224], 'The dimensions to use as entry for the images.')
    tf.app.flags.DEFINE_integer(
        'image_channels', 3, 'channels of the entry images.')
    tf.app.flags.DEFINE_boolean(
        'distributed_run', False, 'split and sets dirs will be assembled or received.')
    tf.app.flags.DEFINE_string(
        'gpu_to_use', '', 'gpus to use')


    ######################
    # Optimization Flags #
    ######################

    tf.app.flags.DEFINE_float(
        'weight_decay', 0.00004, 'The weight decay on the model weights.')
    #    'weight_decay', 0.005, 'The weight decay on the model weights.')

    tf.app.flags.DEFINE_string(
        'optimizer', 'rmsprop',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

    tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

    tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

    tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

    tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

    tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

    tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                            'The learning rate power.')

    tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

    tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

    tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

    tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

    #######################
    # Learning Rate Flags #
    #######################

    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

    #tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

    tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.00000001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

    tf.app.flags.DEFINE_float(
        'label_smoothing', 0.0, 'The amount of label smoothing.')

    tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

    tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 1.0,
        'Number of epochs after which learning rate decays.')

    #######################
    # Extraction Flags #
    #######################

    tf.app.flags.DEFINE_string(
    'output_file', None, 'File to output predictions or features, by default the standard output.')

    tf.app.flags.DEFINE_string(
    'metrics_file', None, 'File to append metrics, in addition to the standard output.')

    tf.app.flags.DEFINE_string(
    'output_format', 'text', 'Format of the output: text or (only with --extract_features) pickle.')

    tf.app.flags.DEFINE_bool(
    'extract_features', False,
    'Extracts features instead of predictions to output_file. No metrics will be computed.')

    tf.app.flags.DEFINE_string(
    'inception_layer', 'PreLogitsFlatten',
    'Network Layer used to extrack features. Valid options are one of network layers. '
    'Must be used with --extract_features')

    return tf.app.flags.FLAGS

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