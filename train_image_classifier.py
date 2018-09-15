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

FLAGS = helpers.define_flags()

# Global vars
dataset_labels = ['Porn', 'NonPorn']
dataset_labels = ['1', '0'] #We are extracting labels from filenames and there is is as '1' and '0'
training_set_length = 0

def input_fn(videos_in_split,
             image_size=tuple(FLAGS.image_shape),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(dataset_labels))

    def _map_func(frame_identificator):
        frame_info = tf.string_split([frame_identificator], delimiter='_')
        video_name = frame_info.values[0]
        label = frame_info.values[1]
        video_path = tf.string_join( inputs=[os.path.join(FLAGS.dataset_dir, 'videos'), '/' , video_name, '.mp4'])

        snippet_path = tf.string_join( inputs=[helpers.assembly_snippets_path(FLAGS), '/', video_name, '/', frame_identificator, '.txt' , ])
        frames_identificator = tf.string_to_number(frame_info.values[2], out_type=tf.int32)

        snippet = tf.py_func(get_video_frames, [video_path, frames_identificator, snippet_path, image_size, FLAGS.split_type], tf.float32, stateful=False, name='retrieve_snippet')
        snippet = tf.image.per_image_standardization(snippet)
        snippet_size = 1 if FLAGS.split_type == '2D' else FLAGS.snippet_size

        snippet.set_shape([snippet_size] + list(image_size) + [3])
        snippet = tf.squeeze(snippet)

        print('snippet shape: ', snippet.shape)
        
#        return (snippet, tf.one_hot(table.lookup(label), len(dataset_labels)))
        return (snippet, table.lookup(label))

    dataset = tf.data.Dataset.from_tensor_slices(videos_in_split)
    print('dataset len: ', len(videos_in_split))

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(len(videos_in_split), num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(len(videos_in_split))
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
        base_model = tf.keras.applications.MobileNet(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False, classes=len(dataset_labels))
    elif FLAGS.model_name == 'inception-v3':
        base_model = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    elif FLAGS.model_name == 'inception-v4':
        base_model = keras_inception_v4.create_model(num_classes=2, include_top=False)
    elif FLAGS.model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False, classes=len(dataset_labels))
    else:
        raise ValueError('Unsupported deep network model')

    top_model = tf.keras.models.Sequential()
    top_model.add(tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(tf.keras.layers.Dense(len(dataset_labels), activation='softmax'))
    keras_model = tf.keras.Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # for layer in keras_model.layers[:-4]:
    #     layer.trainable = False

    global_step = tf.train.get_or_create_global_step()
    learning_rate = helpers.configure_learning_rate(FLAGS, training_set_length, global_step)
    optimizer = helpers.configure_optimizer(FLAGS, learning_rate)

    keras_model.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    return keras_model

def model_fn(features, labels, mode, params=None, config=None):
    train_op = None
    loss = None
    predictions = None

    if FLAGS.model_name == 'i3d-sonnet':
        dnn_model = i3d.InceptionI3d(num_classes=2, spatial_squeeze=True)
        probabilities, end_points = dnn_model(features, is_training=False, dropout_keep_prob=1.0)
        logits = end_points['Logits']

    if mode in (ModeKeys.PREDICT, ModeKeys.EVAL):
        predicted_indices = tf.argmax(probabilities, 1)

    if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, len(dataset_labels)), logits=logits)
        tf.summary.scalar('loss', loss)

    if mode == ModeKeys.PREDICT:
      # Convert predicted_indices back into strings
      predictions = {
          'classes': tf.gather(label_values, predicted_indices),
          'scores': tf.reduce_max(probabilities, axis=1),
          'probabilities': probabilities,
          'logits': logits,
      }
      export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(
          mode, predictions=predictions, export_outputs=export_outputs)

    if mode == ModeKeys.TRAIN:
        learning_rate = helpers.configure_learning_rate(FLAGS, training_set_length, global_step)
        optimizer = helpers.configure_optimizer(FLAGS, learning_rate)

        train_op = slim.optimize_loss(loss=loss,
                                        global_step=global_step,
                                        learning_rate=0.001,
                                        clip_gradients=10.0,
                                        optimizer=optimizer,
                                        summaries=slim.OPTIMIZER_SUMMARIES
                                        )
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices),
            'auroc': tf.metrics.auc(tf.one_hot(labels, len(dataset_labels)), probabilities)
        }
        tf.summary.scalar('accuracy', eval_metric_ops['accuracy'])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
                                    keep_checkpoint_every_n_hours=3,
                                    keep_checkpoint_max=32,
                                    save_summary_steps= (training_set_length / 100),
                                    log_step_count_steps = 400)


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

    # Getting validation and training sets
    network_training_set, network_validation_set = helpers.get_splits(FLAGS)
    training_set_length = int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size)
    validation_set_length = int((FLAGS.epochs * len(network_validation_set))/FLAGS.batch_size)
    print('training set length', training_set_length)
    print('validation set length', validation_set_length)

    estimator = create_estimator()
    time_hist = TimeHistory()
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    tf.train.get_or_create_global_step()

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(network_training_set,
                                            shuffle=True,
                                            batch_size=FLAGS.batch_size,
                                            num_epochs=FLAGS.epochs,
                                            prefetch_buffer_size=FLAGS.batch_size * 3),
                                            max_steps= int((FLAGS.epochs * training_set_length)/FLAGS.batch_size),
                                            hooks=[time_hist])

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(network_validation_set,
                                            shuffle=False,
                                            batch_size=FLAGS.batch_size,
                                            num_epochs=1),
                                            steps=validation_set_length,
                                            throttle_secs=FLAGS.eval_interval_secs)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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
    # for result in results:
    #     print('result: {}'.format(result))

    total_time =  sum(time_hist.times)
    print('total time with ', FLAGS.num_gpus, 'GPUs:', total_time, 'seconds')

    avg_time_per_batch = np.mean(time_hist.times)
    print(FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch, 'images/second with', FLAGS.num_gpus, 'GPUs')



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()