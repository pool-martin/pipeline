import os
import time

import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
#tf.enable_eager_execution()

import numpy as np

from nets import i3d, keras_i3d, keras_inception_v4
from utils.get_file_list import getListOfFiles
from utils.time_history import TimeHistory
import utils.helpers as helpers
from utils.opencv import get_video_frames
#from preprocessing import preprocessing

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
    'eval_interval_secs', 600, 'Do not re-evaluate unless the last evaluation was started at least this many seconds ago.')
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

labels = ['Porn', 'NonPorn']
labels = ['1', '0']
FLAGS = tf.app.flags.FLAGS

def input_fn(videos_in_split, labels,
             image_size=tuple(FLAGS.image_shape),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(frame_identificator):
        frame_info = tf.string_split([frame_identificator], delimiter='_')
        video_name = frame_info.values[0]
        label = frame_info.values[1]
        video_path = tf.string_join( inputs=[os.path.join(FLAGS.dataset_dir, 'videos'), '/' , video_name, '.mp4'])

        snippet_path = tf.string_join( inputs=[helpers.assembly_snippets_path(FLAGS), '/', video_name, '/', frame_identificator, '.txt' , ])
        frames_identificator = tf.string_to_number(frame_info.values[2], out_type=tf.int32)

        snippet = tf.py_func(get_video_frames, [video_path, frames_identificator, snippet_path, image_size, FLAGS.split_type], tf.float32, stateful=False, name='retrieve_snippet')
        # print('snippet shape: ', snippet[0].shape, 'len', len(snippet))

        # snippet = [tf.image.convert_image_dtype(item, dtype=tf.float32) for item in snippet]
        snippet = tf.image.per_image_standardization(snippet)
        snippet_size = 1 if FLAGS.split_type == '2D' else FLAGS.snippet_size

        # snippet = tf.stack(snippet)
        snippet.set_shape([snippet_size] + list(image_size) + [3])
        snippet = tf.squeeze(snippet)
#        preprocess_image

        print('snippet shape: ', snippet.shape)
        
        return (snippet, tf.one_hot(table.lookup(label), num_classes))
#        return (snippet, table.lookup(label))
#        return (dict({'snippet':snippet}), tf.one_hot(table.lookup(label), num_classes))
#        return (dict({'image':snippet}), table.lookup(label))

    dataset = tf.data.Dataset.from_tensor_slices(videos_in_split)
    print('dataset len: ', len(videos_in_split))

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
       tf.contrib.data.map_and_batch(map_func=_map_func,
                                     batch_size=batch_size,
                                     num_parallel_calls=os.cpu_count()))
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def keras_model():

    if FLAGS.model_name == 'i3d-keras':
        base_model = keras_i3d.Inception_Inflated3d(input_shape=((FLAGS.snippet_size,) + tuple(FLAGS.image_shape) + (FLAGS.image_channels,)), include_top=False)
    elif FLAGS.model_name == 'mobilenet-3d':
        print('TODO')
    elif FLAGS.model_name == 'mobilenet':
        base_model = tf.keras.applications.MobileNet(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False, classes=len(labels))
    elif FLAGS.model_name == 'inception-v3':
        base_model = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    elif FLAGS.model_name == 'inception-v4':
        base_model = keras_inception_v4.create_model(num_classes=2, include_top=False)
    elif FLAGS.model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False, classes=len(labels))
    else:
        raise ValueError('Unsupported deep network model')

    top_model = tf.keras.models.Sequential()
    top_model.add(tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))
    keras_model = tf.keras.Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # for layer in keras_model.layers[:-4]:
    #     layer.trainable = False

    keras_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    return keras_model

def model_fn(features, labels, mode, params=None, config=None):
    train_op = None
    loss = None
    predictions = None
    eval_metrics = None

    if FLAGS.model_name == 'i3d-sonnet':
        i3d_model = i3d.InceptionI3d(num_classes=2, spatial_squeeze=True)
        predictions, end_points = i3d_model(features, is_training=False, dropout_keep_prob=1.0)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=end_points['Logits'])
        predicted_classes = tf.argmax(end_points['Logits'], 1)


    if mode == ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        predictions, end_points = i3d_model(features, is_training=True, dropout_keep_prob=1.0)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=end_points['Logits'])
        predicted_classes = tf.argmax(end_points['Logits'], 1)
        learning_rate = helpers.configure_learning_rate(FLAGS, 33000, global_step)
        optimizer = helpers.configure_optimizer(FLAGS, learning_rate)

        train_op = slim.optimize_loss(loss=loss,
                                        global_step=global_step,
                                        learning_rate=0.001,
                                        clip_gradients=10.0,
                                        optimizer=optimizer,
                                        summaries=slim.OPTIMIZER_SUMMARIES
                                        )
    elif mode == ModeKeys.PREDICT:
        predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(end_points['Logits']),
        'logits': end_points['Logits'],
        }
#    elif mode == ModeKeys.EVAL:

    # accuracy = tf.metrics.accuracy(labels=labels,
    #                            predictions=predicted_classes,
    #                            name='acc_op')
    # accuracy = tf.reduce_mean(tf.stack(accuracy, axis=0))
    # eval_metrics = {'accuracy': accuracy}
    # tf.summary.scalar('accuracy/training', accuracy[1])

    return EstimatorSpec(train_op=train_op, loss=loss, eval_metric_ops=eval_metrics, predictions=predictions,
                            mode=mode)

def adjust_image(data):
    # Reshape to [batch, height, width, channels].
    imgs = tf.reshape(data, [-1, 28, 28, 1])
    # Adjust image size to Inception-v3 input.
    imgs = tf.image.resize_images(imgs, (299, 299))
    # Convert to RGB image.
    imgs = tf.image.grayscale_to_rgb(imgs)
    return imgs

def inceptionv3_model_fn(features, labels, mode):
    # Load Inception-v3 model.
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
#    input_layer = adjust_image(features["snippet"])
#    outputs = module(input_layer)
    outputs = module(features)

    logits = tf.layers.dense(inputs=outputs, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def create_estimator():
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    config = tf.estimator.RunConfig(train_distribute= strategy if FLAGS.distributed_run else None, 
                                    session_config=sess_config,
                                    model_dir=helpers.assembly_model_dir(FLAGS),
                                    tf_random_seed=1,
                                    save_checkpoints_secs=600,
                                    keep_checkpoint_every_n_hours=1,
                                    keep_checkpoint_max=10,
                                    save_summary_steps=100)


    if(FLAGS.model_name in ['mobilenet', 'VGG16', 'inception-v3', 'i3d-keras']):
       estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model(), config=config)
    if(FLAGS.model_name in ['i3d-sonnet']):
        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                                config=config,
                                                model_dir=config.model_dir)
    if(FLAGS.model_name in ['inception-v3-hub']):
        estimator = tf.estimator.Estimator(model_fn=inceptionv3_model_fn,
                                                config=config,
                                                model_dir=config.model_dir)

    return estimator

def main():
    if FLAGS.gpu_to_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_to_use

    helpers.check_and_create_directories(FLAGS)
    network_training_set, network_validation_set = helpers.get_splits(FLAGS)

    estimator = create_estimator()
    time_hist = TimeHistory()
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    tf.train.create_global_step()

    print('#######################', int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size))
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(network_training_set,
                                            labels,
                                            shuffle=True,
                                            batch_size=FLAGS.batch_size,
                                            buffer_size=2048,
                                            num_epochs=FLAGS.epochs,
                                            prefetch_buffer_size=4),
                                            max_steps= int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size),
                                            hooks=[time_hist])

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(network_validation_set,
                                            labels, 
                                            shuffle=False,
                                            batch_size=FLAGS.batch_size,
                                            buffer_size=2048,
                                            num_epochs=1),
                                            steps=None,
                                            throttle_secs=FLAGS.eval_interval_secs)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # estimator.train(input_fn=lambda:input_fn(network_training_set,
    #                                         labels,
    #                                         shuffle=True,
    #                                         batch_size=FLAGS.batch_size,
    #                                         buffer_size=2048,
    #                                         num_epochs=FLAGS.epochs,
    #                                         prefetch_buffer_size=4),steps=150)
    # estimator.evaluate(input_fn=lambda:input_fn(network_training_set,
    #                                         labels,
    #                                         shuffle=True,
    #                                         batch_size=FLAGS.batch_size,
    #                                         buffer_size=2048,
    #                                         num_epochs=FLAGS.epochs,
    #                                         prefetch_buffer_size=4),steps=int((FLAGS.epochs * len(network_validation_set))/FLAGS.batch_size))

   # outfile = open(FLAGS.output_file, 'w') if FLAGS.output_file else sys.stdout
    # results = estimator.predict( input_fn=lambda:input_fn(network_validation_set,
    #                                         labels, 
    #                                         shuffle=False,
    #                                         batch_size=FLAGS.batch_size,
    #                                         buffer_size=2048,
    #                                         num_epochs=1) )
    for result in results:
        print('result: {}'.format(result))

    total_time =  sum(time_hist.times)
    print('total time with ', FLAGS.num_gpus, 'GPUs:', total_time, 'seconds')

    avg_time_per_batch = np.mean(time_hist.times)
    print(FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch, 'images/second with', FLAGS.num_gpus, 'GPUs')



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()