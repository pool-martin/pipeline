import os
import time

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np

import nets
from utils.get_file_list import getListOfFiles
from utils.time_history import TimeHistory
from utils.sets_helper import assembly_sets_path, assembly_snippets_path
from utils.opencv import get_video_frames

tf.app.flags.DEFINE_string(
    'model_name', 'i3d', 'The name of the cnn to train.')
tf.app.flags.DEFINE_string(
    'model_dir', '/Exp/2kporn/i3d/checkpoint', 'path to save checkpoints.')
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
    'split_type', '2D', 'Set type to be used.')
tf.app.flags.DEFINE_integer(
    'num_gpus', 2, 'The number of gpus that should be used')
tf.app.flags.DEFINE_integer(
    'sample_rate', 1, 'sample rate of the dataset in fps')
tf.app.flags.DEFINE_integer(
    'snippet_size', 32, 'The number of frames in the snippet')
tf.app.flags.DEFINE_integer(
    'snippet_width', 5, 'The length in seconds the snippet should represent')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of snippets in the batch')
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
        frame_number = tf.string_to_number(frame_info.values[2], out_type=tf.int32)
        video_path = tf.string_join( inputs=[os.path.join(FLAGS.dataset_dir, 'videos'), '/' , video_name, '.mp4'])
        if ('3D' in FLAGS.split_type):
            with open(os.path.join(assembly_sets_path(FLAGS), video_name,  '{}.txt'.format(frame_identificator)), 'r') as f:
                frames = f.read().split('\n')
            snippet = tf.py_func(get_video_frames, [video_path, frames, image_size], [tf.float32], stateful=False, name='flow_transform')
        else:
            frames = [frame_number]
            snippet = tf.py_func(get_video_frames, [video_path, frames, image_size], [tf.float32], stateful=False, name='flow_transform')
        
        snippet = tf.stack(snippet)
        snippet.set_shape([len(frames)] + list(image_size) + [3])
        snippet = tf.squeeze(snippet)
        
        return (snippet, tf.one_hot(table.lookup(label), num_classes))

    dataset = tf.data.Dataset.from_tensor_slices(videos_in_split)
    print('len ########: ', len(videos_in_split))

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
    dataset_iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_iterator.initializer)
    return dataset_iterator.get_next()
#    return dataset


def keras_model():

    if FLAGS.model_name == 'i3d':
        keras_model = nets.keras_i3d.Inception_Inflated3d(input_shape=((FLAGS.snippet_size,) + tuple(FLAGS.image_shape) + (FLAGS.image_channels,)), include_top=False)
    elif FLAGS.model_name == 'mobilenet-3d':
        print('TODO')
    elif FLAGS.model_name == 'mobilenet-3d':
        print('TODO')
    elif FLAGS.model_name == 'inception-v4':
        keras_model = nets.keras_inception_v4.create_model(num_classes=2, include_top=False)
    elif FLAGS.model_name == 'VGG16':
        keras_model = tf.keras.applications.VGG16(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False)
    else:
        raise ValueError('Unsupported deep network model')

    output = keras_model.output
    output = tf.keras.layers.Flatten()(output)
    predictions = tf.keras.layers.Dense(len(labels), activation=tf.nn.softmax)(output)

    model = tf.keras.Model(inputs=keras_model.input, outputs=predictions)

    for layer in keras_model.layers[:-4]:
        layer.trainable = False

    optimizer = tf.train.AdamOptimizer()

    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy'])
    return model

def create_estimator():
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
    config = tf.estimator.RunConfig(train_distribute= strategy if FLAGS.distributed_run else None, 
                                    model_dir=FLAGS.model_dir,
                                    tf_random_seed=1,
                                    save_checkpoints_secs=600,
                                    keep_checkpoint_max=10,
                                    save_summary_steps=600)


    estimator = tf.keras.estimator.model_to_estimator(keras_model(), config=config)

    return estimator

def check_and_create_directories():
    directories = [FLAGS.model_dir]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_splits():
    network_training_set = []
    network_validation_set = []

    with open(os.path.join(assembly_sets_path(FLAGS), 'network_training_set.txt'), 'r') as f:
        network_training_set = f.readlines()
    with open(os.path.join(assembly_sets_path(FLAGS), 'network_validation_set.txt'), 'r') as f:
        network_validation_set = f.readlines()

    return network_training_set, network_validation_set

def main():
    check_and_create_directories()
    network_training_set, network_validation_set = get_splits()

    estimator = create_estimator()
    time_hist = TimeHistory()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.train.create_global_step()

#    print('#######################', int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size))
    # train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(network_training_set,
    #                                         labels,
    #                                         shuffle=True,
    #                                         batch_size=FLAGS.batch_size,
    #                                         buffer_size=2048,
    #                                         num_epochs=FLAGS.epochs,
    #                                         prefetch_buffer_size=4),
    #                                         max_steps= int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size),
    #                                         hooks=[time_hist])

    # eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(network_validation_set,
    #                                         labels, 
    #                                         shuffle=False,
    #                                         batch_size=FLAGS.batch_size,
    #                                         buffer_size=2048,
    #                                         num_epochs=1),
    #                                         steps=None,
    #                                         throttle_secs=FLAGS.eval_interval_secs)
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.train(lambda:input_fn(network_training_set,
                                            labels,
                                            shuffle=True,
                                            batch_size=FLAGS.batch_size,
                                            buffer_size=2048,
                                            num_epochs=FLAGS.epochs,
                                            prefetch_buffer_size=4),
                                            steps=100,
                                            hooks=[time_hist])

    total_time =  sum(time_hist.times)
    print('total time with ', FLAGS.num_gpus, 'GPUs:', total_time, 'seconds')

#    avg_time_per_batch = np.mean(time_hist.times)
#    print(FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch, 'images/second with', FLAGS.num_gpus, 'GPUs')



if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    main()