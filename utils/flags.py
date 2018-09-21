import tensorflow as tf

def define_flags():
    tf.app.flags.DEFINE_string(
        'model_name', 'i3d', 'The name of the cnn to train.')
    tf.app.flags.DEFINE_string(
        'model_dir', '/Exp/2kporn/experiments', 'path to save checkpoints.')
    tf.app.flags.DEFINE_string(
        'checkpoint_path', '', 'path to load seed weigths.')
    tf.app.flags.DEFINE_string(
        'ws_checkpoint_dir', '/DL/initial_weigths/', 'warm start checkpoint dir.')
    tf.app.flags.DEFINE_string(
        'ws_checkpoint', 'rgb_imagenet', 'warm start checkpoint to be used.')
    tf.app.flags.DEFINE_string(
        'experiment_tag', 'initial_training', 'tag used in model dir places where we want to diferentiate subsequent executions.')
    tf.app.flags.DEFINE_string(
        'dataset_dir', '/DL/2kporn/', 'The sets to be used.')
    tf.app.flags.DEFINE_string(
        'sets_dir', '/Exp/2kporn/splits', 'The sets to be used.')
    tf.app.flags.DEFINE_bool(
        'mini_sets', False, 'use mini sets (to debug).')
    
    tf.app.flags.DEFINE_string(
        'snippets_dir', '', 'The sets to be used.')
    tf.app.flags.DEFINE_boolean(
        'force_splits_dir_path', False, 'split and sets dirs will be assembled or received.')
    tf.app.flags.DEFINE_string(
        'split_number', 's1', 'Split number to be used.')
    tf.app.flags.DEFINE_string(
        'split_type', '3D', 'Set type to be used.')
    tf.app.flags.DEFINE_integer(
        'num_gpus', 1, 'The number of gpus that should be used')
    tf.app.flags.DEFINE_integer(
        'sample_rate', 1, 'sample rate of the dataset in fps')
    tf.app.flags.DEFINE_integer(
        'snippet_size', 32, 'The number of frames in the snippet')
    tf.app.flags.DEFINE_integer(
        'snippet_width', 4, 'The length in seconds the snippet should represent')
    tf.app.flags.DEFINE_integer(
        'batch_size', 8, 'The number of snippets in the batch')
    tf.app.flags.DEFINE_integer(
        'epochs', 25, 'The number of epochs to run the training')
    tf.app.flags.DEFINE_integer(
        'eval_interval_secs', 3* 3600, 'Do not re-evaluate unless the last evaluation was started at least this many seconds ago.')
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
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays.')

    #######################
    # Extraction Flags #
    #######################

    tf.app.flags.DEFINE_string(
    'output_file', None, 'File to output predictions or features, by default the standard output.')

    tf.app.flags.DEFINE_string(
    'metrics_file', None, 'File to append metrics, in addition to the standard output.')

    tf.app.flags.DEFINE_string(
    'output_format', 'pickle', 'Format of the output: text or pickle.')

    tf.app.flags.DEFINE_bool(
        'train', True,
        'Should train.')

    tf.app.flags.DEFINE_bool(
        'eval', True,
        'Should evaluate.')

    tf.app.flags.DEFINE_bool(
        'predict', True,
        'Should predict and extract features to output_file.')

    tf.app.flags.DEFINE_bool(
        'predict_from_initial_weigths', False,
        'Predict using initial imagenet or kinetics weigths.')

    tf.app.flags.DEFINE_string(
        'inception_layer', 'PreLogitsFlatten',
        'Network Layer used to extrack features. Valid options are one of network layers. '
        'Must be used with --extract_features')

    #######################
    # Preprocessing Flags #
    #######################

    tf.app.flags.DEFINE_integer(
        'normalize_per_image', 1, 'Normalization per image: 0 (None), 1 (Mean), 2 (Mean and Stddev)')
    tf.app.flags.DEFINE_list(
        'image_shape', [224, 224], 'The dimensions to use as entry for the images.')
    tf.app.flags.DEFINE_integer(
        'image_channels', 3, 'channels of the entry images.')

    if(tf.app.flags.FLAGS.model_name in ['i3d']):
        tf.app.flags.FLAGS.split_type = '3D'
    elif(tf.app.flags.FLAGS.model_name in ['inception_v4']):
        tf.app.flags.FLAGS.split_type = '2D'


    return tf.app.flags.FLAGS
