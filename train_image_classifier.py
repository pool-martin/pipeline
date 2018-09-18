import os
import time

import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub
#tf.enable_eager_execution()

import numpy as np

from nets import i3d, inception_v4
from utils.get_file_list import getListOfFiles
from utils.time_history import TimeHistory
import utils.helpers as helpers
from utils.opencv import get_video_frames
from utils.flags import define_flags
from utils.save_features import save_extracted_features
from utils import fine_tune
from preprocessing import preprocessing_factory

FLAGS = define_flags()

# Global vars
dataset_labels = ['NonPorn', 'Porn']
dataset_labels = ['0', '1'] #We are extracting labels from filenames and there is is as '1' and '0'
training_set_length = 0

def input_fn(videos_in_split,
             image_size=tuple(FLAGS.image_shape),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(dataset_labels))

    image_preprocessing_fn = preprocessing_factory.get_preprocessing( 'preprocessing', is_training= not FLAGS.predict_and_extract)

    def _map_func(frame_identificator):
        frame_info = tf.string_split([frame_identificator], delimiter='_')
        video_name = frame_info.values[0]
        label = frame_info.values[1]
        video_path = tf.string_join( inputs=[os.path.join(FLAGS.dataset_dir, 'videos'), '/' , video_name, '.mp4'])

        snippet_path = tf.string_join( inputs=[helpers.assembly_snippets_path(FLAGS), '/', video_name, '/', frame_identificator, '.txt' , ])
        frames_identificator = tf.string_to_number(frame_info.values[2], out_type=tf.int32)

        snippet = tf.py_func(get_video_frames, [video_path, frames_identificator, snippet_path, image_size, FLAGS.split_type], tf.float32, stateful=False, name='retrieve_snippet')
        snippet_size = 1 if FLAGS.split_type == '2D' else FLAGS.snippet_size

        snippet.set_shape([snippet_size] + list(image_size) + [3])

        snippet = tf.map_fn(lambda img: image_preprocessing_fn(img, image_size[0], image_size[1],
                                                                normalize_per_image=FLAGS.normalize_per_image), snippet)
        snippet = tf.squeeze(snippet)
        
        snippet_id = tf.string_join( inputs=[video_name, '_', frame_info.values[2] ] )
        return ({'snippet_id': snippet_id, 'snippet': snippet, 'label': table.lookup(label) }, table.lookup(label))

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
                                     num_parallel_calls=os.cpu_count()/2 ))
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def keras_model():

    # if FLAGS.model_name == 'i3d-keras':
    #     base_model = keras_i3d.Inception_Inflated3d(input_shape=((FLAGS.snippet_size,) + tuple(FLAGS.image_shape) + (FLAGS.image_channels,)), include_top=False)
    # elif FLAGS.model_name == 'inception-v4':
    #     base_model = keras_inception_v4.create_model(num_classes=2, include_top=False)
    if FLAGS.model_name == 'mobilenet-3d':
        raise ValueError('Unsupported deep network model')
    elif FLAGS.model_name == 'mobilenet':
        base_model = tf.keras.applications.MobileNet(input_shape=tuple(FLAGS.image_shape) + (FLAGS.image_channels,), include_top=False, classes=len(dataset_labels))
    elif FLAGS.model_name == 'inception-v3':
        base_model = tf.keras.applications.inception_v3.InceptionV3(weights=None)
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
    scaffold = None


    if mode == ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    if FLAGS.model_name == 'i3d':
        with tf.variable_scope('RGB'):
            dnn_model = i3d.InceptionI3d(num_classes=2, spatial_squeeze=True)
            probabilities, end_points = dnn_model(features['snippet'], is_training=is_training)
            logits = end_points['Logits']
            extracted_features = tf.layers.Flatten()(end_points['Mixed_5c'])

        scope_to_exclude = ["RGB/inception_i3d/Logits"]
        ws_checkpoint = FLAGS.i3d_ws_checkpoint
        pattern_to_exclude = []


    if FLAGS.model_name == 'inception-v4':
        logits, end_points = inception_v4.inception_v4(features['snippet'], is_training=is_training, num_classes=2)
        probabilities = end_points['Predictions']
        extracted_features = end_points['PreLogitsFlatten']
        scope_to_exclude = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        ws_checkpoint = FLAGS.inception_v4_ws_checkpoint
        pattern_to_exclude = ['biases', "global_step"]

    training_model_dir = helpers.assembly_model_dir(FLAGS)
    if FLAGS.predict_from_initial_weigths or not(os.path.exists(training_model_dir) and (os.path.isdir(training_model_dir) and os.listdir(training_model_dir))):
        #tf.train.init_from_checkpoint(ws_checkpoint, {v.name.split(':')[0]: v for v in variables_to_restore})
        scaffold = tf.train.Scaffold(init_op=None, init_fn=fine_tune.init_weights(scope_to_exclude, pattern_to_exclude, ws_checkpoint))

    
    if mode in (ModeKeys.PREDICT, ModeKeys.EVAL):
        predicted_indices = tf.argmax(logits, axis=-1)

    if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(features['label'], len(dataset_labels)), logits=logits)
        tf.summary.scalar('loss', loss)


    if mode == ModeKeys.PREDICT:
        print('_model_func label', features['label'])
        tf.Print(features['label'], [features['label']], '_model_func label')
        predictions = {
           'snippet_id': features['snippet_id'],
           'truth_label': features['label'],
           'predicted_label': predicted_indices,
           'probabilities': probabilities,
           'features': extracted_features,
           'scores': tf.reduce_max(probabilities, axis=1),
           'logits': logits,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == ModeKeys.TRAIN:
        #learning_rate = helpers.configure_learning_rate(FLAGS, training_set_length, global_step)
        #optimizer = helpers.configure_optimizer(FLAGS, learning_rate)
        

        #train_op = slim.optimize_loss(loss=loss,
        #                                global_step=global_step,
        #                                learning_rate=0.001,
        #                                clip_gradients=10.0,
        #                                optimizer=optimizer,
        #                                summaries=slim.OPTIMIZER_SUMMARIES
        #                                )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)

    if mode == ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices)
            # 'auroc': tf.metrics.auc(tf.one_hot(labels, len(dataset_labels)), logits)
        }
        tf.summary.scalar('accuracy', eval_metric_ops['accuracy'])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def create_estimator(checkpoint_path=None):
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # if FLAGS.num_gpus == 0:
    #     distribution = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
    # elif FLAGS.num_gpus == 1:
    #     distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
    # else:
    #     distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)

    config = tf.estimator.RunConfig(train_distribute= strategy if FLAGS.distributed_run else None, 
                                    session_config=sess_config,
                                    model_dir=checkpoint_path or helpers.assembly_model_dir(FLAGS),
                                    tf_random_seed=1,
                                    save_checkpoints_secs=(FLAGS.eval_interval_secs / 2 ),
                                    keep_checkpoint_max=32,
                                    save_summary_steps= (training_set_length / 100),
                                    log_step_count_steps = 100)

    if(FLAGS.model_name in ['mobilenet', 'VGG16', 'inception-v3', 'i3d-keras']):
       estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model(), config=config)
    elif(FLAGS.model_name in ['i3d', 'inception-v4']):

        if FLAGS.model_name == 'i3d':
            ws_checkpoint = FLAGS.i3d_ws_checkpoint
            exclude_from_ws_checkpoint = []

        # elif FLAGS.model_name == 'inception-v4':
        #     ws_checkpoint = FLAGS.inception_v4_ws_checkpoint
        #     exclude_from_ws_checkpoint = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]

        #  _variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude_from_ws_checkpoint)
        # # _variables_to_restore =  tf.global_variables()
        # # _variables_to_restore = [v for v in _variables_to_restore if 'global_step' not in v.name ]
        #  rgb_variable_map = {v.name.split(':')[0]: v.name for v in _variables_to_restore}
    
        # ws = tf.estimator.WarmStartSettings(
        #     ckpt_to_initialize_from=ws_checkpoint)
        #     # var_name_to_prev_var_name= teste_JP)
        
        # rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        # rgb_saver.restore() (sess, _CHECKPOINT_PATHS['rgb_imagenet'])

        ws = None

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                                config=config,
                                                params=None,
                                                model_dir=config.model_dir,
                                                warm_start_from=ws)
    else:
        raise ValueError('Unsupported network model')

    return estimator

def main():
    if FLAGS.gpu_to_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_to_use

    helpers.check_and_create_directories(FLAGS)

    time_hist = TimeHistory()
    tf.train.get_or_create_global_step()

    if not FLAGS.predict_and_extract:
        estimator = create_estimator()

        # Getting validation and training sets
        network_training_set, network_validation_set = helpers.get_splits(FLAGS)
        training_set_max_steps = int((FLAGS.epochs * len(network_training_set))/FLAGS.batch_size)
        validation_set_max_steps = int((FLAGS.epochs * len(network_validation_set))/FLAGS.batch_size)
        print('training set length: {}, max steps: {}'.format(len(network_training_set), training_set_max_steps))
        print('validation set length: {}, max steps {}'.format(len(network_validation_set), validation_set_max_steps))

        train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(network_training_set,
                                                    shuffle=True,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=FLAGS.epochs,
                                                    prefetch_buffer_size=FLAGS.batch_size * 3),
                                                max_steps= training_set_max_steps,
                                                hooks=[time_hist])

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(network_validation_set,
                                                    shuffle=False,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=1),
                                                steps=validation_set_max_steps,
                                                start_delay_secs=FLAGS.eval_interval_secs,
                                                throttle_secs=FLAGS.eval_interval_secs,
                                                hooks=[time_hist])
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        sets_to_extract = helpers.get_sets_to_extract(FLAGS)

        if not tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            estimator = create_estimator(FLAGS.checkpoint_path)


        for set_name, set_to_extract in sets_to_extract.items():
            pred_generator = estimator.predict( input_fn=lambda:input_fn(set_to_extract,
                                                    shuffle=False,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=1),
                                                predict_keys=['snippet_id', 'truth_label', 'features'],
                                                hooks=[time_hist])
            save_extracted_features(FLAGS, set_name, set_to_extract, pred_generator)



    total_time =  sum(time_hist.times)
    print('total time with ', FLAGS.num_gpus, 'GPUs:', total_time, 'seconds')

    avg_time_per_batch = np.mean(time_hist.times)
    print(FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch, 'images/second with', FLAGS.num_gpus, 'GPUs')



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()