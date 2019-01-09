import os
import time

import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys #pylint: disable=E0611
import tensorflow.contrib.slim as slim
# import tensorflow_hub as hub
import gc
from threading import Event
import time
#tf.enable_eager_execution()

import numpy as np

from nets import i3d, i3d_v4, i3d_v4_slim, c3d
from nets import nets_factory
from utils.get_file_list import getListOfFiles
from utils.time_history import TimeHistory
import utils.helpers as helpers
from utils.opencv import get_video_frames
from utils.video_loader import VideoLoader
from utils.flags import define_flags
from utils.save_features import save_extracted_features
from utils import fine_tune
from preprocessing import preprocessing_factory

FLAGS = define_flags()

# Global vars
dataset_labels = ['NonPorn', 'Porn']
dataset_labels = ['0', '1'] #We are extracting labels from filenames and there is is as '1' and '0'
training_set_length = 0

dataset_loader = None

def input_fn(videos_in_split,
             image_size=tuple([FLAGS.image_shape,FLAGS.image_shape]),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None,
             fragments_count=None,
             dataset_in_memory=False):
    # print('1##############################################################################################################\n#####################################################################')

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(dataset_labels))
    fragments_count_table = tf.contrib.lookup.HashTable(  tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(list(fragments_count.keys())), tf.constant(list(fragments_count.values()))), -1 )

    image_preprocessing_fn = preprocessing_factory.get_preprocessing( 'preprocessing', is_training= not FLAGS.predict)
    # print('2##############################################################################################################\n#####################################################################')
        # tf.print(snippet, [snippet_id, snippet], "\n\nsnippet values: \n" )

    def _map_func(frame_identificator):
        # tf.print(frame_identificator, [frame_identificator], "\nframe_identificator: \n" )
        frame_info = tf.string_split([frame_identificator], delimiter='_')
        video_name = frame_info.values[0]
        label = frame_info.values[1]
        video_path = tf.string_join( inputs=[os.path.join(FLAGS.dataset_dir, 'videos'), '/' , video_name, '.mp4'])

        snippet_path = tf.string_join( inputs=[helpers.assembly_snippets_path(FLAGS, dataset_in_memory), '/', video_name, '/', frame_identificator, '.txt' , ])
        frames_identificator = tf.string_to_number(frame_info.values[2], out_type=tf.int32)
        # tf.print(snippet_path, [snippet_path], "\n\snippet_path: \n" )

        if dataset_in_memory:
#            global dataset_loader
            video_fragment_count = fragments_count_table.lookup(video_name)
            # tf.print(video_fragment_count, [video_fragment_count], "\n\video_fragment_count: \n" )
            snippet = tf.py_func(dataset_loader.get_video_frames, [video_name, frames_identificator, snippet_path, image_size, FLAGS.split_type, video_fragment_count, FLAGS.debug], tf.float32, stateful=False, name='retrieve_snippet')
            # tf.print(snippet, [video_fragmesnippetnt_count], "\n\snippet: \n" )
            # snippet.set_shape([FLAGS.snippet_size, FLAGS.image_shape, FLAGS.image_shape, 3])
            # tf.print(snippet, [snippet], "\n\snippet: \n" )
        else:
            snippet = tf.py_func(get_video_frames, [video_path, frames_identificator, snippet_path, image_size, FLAGS.split_type], tf.float32, stateful=False, name='retrieve_snippet')

        snippet_size = 1 if FLAGS.split_type == '2D' else FLAGS.snippet_size
        snippet.set_shape([snippet_size] + list(image_size) + [3])

        snippet = tf.map_fn(lambda img: image_preprocessing_fn(img, FLAGS.image_shape, FLAGS.image_shape,
                                                                normalize_per_image=FLAGS.normalize_per_image), snippet)
        snippet = tf.squeeze(snippet)
        
        snippet_id = tf.string_join( inputs=[video_name, '_', frame_info.values[2] ] )
        # tf.print(snippet, [snippet_id], "\n\nsnippet values: \n" )
        return ({'snippet_id': snippet_id, 'snippet': snippet, 'label': table.lookup(label) }, table.lookup(label))

    # print('3##############################################################################################################\n#####################################################################')
    dataset = tf.data.Dataset.from_tensor_slices(videos_in_split)

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(len(videos_in_split), num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(len(videos_in_split))
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epoci3d_v4_slimhs)

    # print('4##############################################################################################################\n#####################################################################')
    dataset = dataset.apply(
       tf.data.experimental.map_and_batch(map_func=_map_func,
                                     batch_size=batch_size,
                                     num_parallel_calls=int(os.cpu_count()/2) ))
    dataset = dataset.prefetch(buffer_size= 2 * FLAGS.batch_size)
    # print('5##############################################################################################################\n#####################################################################')
    return dataset


def keras_model():

    # if FLAGS.model_name == 'i3d-keras':
    #     base_model = keras_i3d.Inception_Inflated3d(input_shape=((FLAGS.snippet_size,) + tuple([FLAGS.image_shape, FLAGS.image_shape]) + (FLAGS.image_channels,)), include_top=False)
    # elif FLAGS.model_name == 'inception-v4':
    #     base_model = keras_inception_v4.create_model(num_classes=2, include_top=False)
    if FLAGS.model_name == 'mobilenet-3d':
        raise ValueError('Unsupported deep network model')
    elif FLAGS.model_name == 'mobilenet':
        base_model = tf.keras.applications.MobileNet(input_shape=tuple([FLAGS.image_shape, FLAGS.image_shape]) + (FLAGS.image_channels,), include_top=False, classes=len(dataset_labels))
    elif FLAGS.model_name == 'inception-v3':
        base_model = tf.keras.applications.inception_v3.InceptionV3(weights=None)
    elif FLAGS.model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=tuple([FLAGS.image_shape, FLAGS.image_shape]) + (FLAGS.image_channels,), include_top=False, classes=len(dataset_labels))
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

def model_fn(features, labels, mode, config=None):
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
            dnn_model = i3d.InceptionI3d(num_classes=len(dataset_labels), spatial_squeeze=True)
            probabilities, end_points = dnn_model(features['snippet'], is_training=is_training)
            logits = end_points['Logits']

            # extract_end_point = end_points['Mixed_5c']

            # kernel_size = extract_end_point.get_shape()[1:4]
            # net = tf.nn.avg_pool3d(extract_end_point, ksize=[1, 2, 7, 7, 1],
            #                 strides=[1, 1, 1, 1, 1], padding='VALID')
            # end_points['global_pool'] = net

        end_points['extracted_features'] = tf.layers.Flatten()(end_points['global_pool'])
        extracted_features = end_points['extracted_features']
        # scope_to_exclude = ["RGB/inception_i3d/Logits"]
        # pattern_to_exclude = []

    if FLAGS.model_name == 'i3d_v4' and not FLAGS.is_sonnet:
        dnn_model = i3d_v4_slim.InceptionV4(num_classes=len(dataset_labels), create_aux_logits=False)
        logits, end_points = dnn_model._build(features['snippet'], is_training=is_training)
        # for key, end_point in end_points.items():
        #   print("endpoint: {}, shape: {}".format(key, end_point.shape))
        probabilities = end_points['Predictions']
        extracted_features = end_points['PreLogitsFlatten']
        # ws_path = helpers.assembly_ws_checkpoint_path(FLAGS)
        # scaffold = tf.train.Scaffold(init_op=None, init_fn=fine_tune.assembly_3d_checkpoint(FLAGS.model_name, ws_path))

    if FLAGS.model_name == 'i3d_v4' and FLAGS.is_sonnet:
        dnn_model = i3d_v4.InceptionI3d_v4(num_classes=len(dataset_labels), create_aux_logits=False)
        logits, end_points = dnn_model(features['snippet'], is_training=is_training)
        probabilities = end_points['Predictions']
        extracted_features = end_points['PreLogitsFlatten']

    if FLAGS.model_name == 'c3d':
        logits, end_points = c3d.C3D(input=features['snippet'], num_classes=len(dataset_labels))
        probabilities = tf.nn.softmax(logits)
        extracted_features = end_points['fc7']

    # if FLAGS.model_name == 'inception_v1':
    #     logits, end_points = inception_v1.inception_v1(features['snippet'], is_training=is_training, num_classes=len(dataset_labels))
    #     probabilities = end_points['Predictions']
    #     extracted_features = tf.layers.Flatten()(end_points['Mixed_5c'])
    #     scope_to_exclude = ["InceptionV1/Logits"]
    #     #pattern_to_exclude = ['biases', "global_step"]

    # if FLAGS.model_name == 'inception_v4':
    #     logits, end_points = inception_v4.inception_v4(features['snippet'], is_training=is_training, num_classes=len(dataset_labels))
    #     probabilities = end_points['Predictions']
    #     extracted_features = end_points['PreLogitsFlatten']
    #     scope_to_exclude = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
    #     pattern_to_exclude = ['biases', "global_step"]

    # if FLAGS.model_name == 'mobilenet_v2':
    #     logits, end_points = mobilenet_v2.mobilenet(features['snippet'], is_training=is_training, num_classes=len(dataset_labels))
    #     probabilities = end_points['Predictions']
    #     extracted_features = tf.layers.Flatten()(end_points['global_pool'])
    #     scope_to_exclude = ["MobilenetV2/Logits"]

    if FLAGS.model_name in ['inception_v1', 'inception_v4', 'mobilenet_v2']:
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=len(dataset_labels),
            weight_decay=FLAGS.weight_decay,
            is_training=is_training)

        logits, end_points = network_fn(features['snippet'])
        probabilities = end_points['Predictions']
        if FLAGS.model_name == 'inception_v1':
            extracted_features = tf.layers.Flatten()(end_points['AvgPool_0a_7x7'])
        if FLAGS.model_name == 'inception_v4':
            extracted_features = end_points['PreLogitsFlatten']
        if FLAGS.model_name == 'mobilenet_v2':
            extracted_features = tf.layers.Flatten()(end_points['global_pool'])

    for key, end_point in end_points.items():
        print("endpoint: {}, shape: {}".format(key, end_point.shape))

    # if FLAGS.predict_from_initial_weigths or helpers.is_first_run(FLAGS):
    # ws_path = helpers.assembly_ws_checkpoint_path(FLAGS)
    # tf.train.init_from_checkpoint(str(ws_path), {v.name.split(':')[0]: v for v in fine_tune.get_variables_to_restore(FLAGS.model_name)})
    # scaffold = tf.train.Scaffold(init_op=None, init_fn=fine_tune.init_weights(FLAGS.model_name, ws_path))

    predicted_indices = tf.argmax(logits, axis=-1)
    print('logits shape ', logits.shape, 'predictions shape ', probabilities.shape, 'predicted_indices shape ', predicted_indices.shape)

    if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(features['label'], len(dataset_labels)), logits=logits)
        tf.summary.scalar('loss', loss)


    if mode == ModeKeys.PREDICT:
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
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs, scaffold=scaffold)

    if mode == ModeKeys.TRAIN:
        learning_rate = helpers.configure_learning_rate(FLAGS, training_set_length, global_step)
        optimizer = helpers.configure_optimizer(FLAGS, learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        accuracy = slim.metrics.accuracy(features['label'], predicted_indices)
        tf.summary.scalar('accuracy_train', accuracy)
        tf.summary.scalar('learning_rate', learning_rate)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)

    if mode == ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(features['label'], predicted_indices),
            # 'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(features['label'], predicted_indices, len(dataset_labels)),
            'auc': tf.metrics.auc(features['label'], predicted_indices),
            "mse": tf.metrics.mean_squared_error(features['label'], predicted_indices),
            'precision': tf.metrics.precision(features['label'], predicted_indices),
            'recall': tf.metrics.recall(features['label'], predicted_indices)
        }
        tf.summary.scalar('accuracy_val', eval_metric_ops['accuracy'])
        # tf.summary.scalar('mean_per_class_accuracy_val', eval_metric_ops['mean_per_class_accuracy'])
        tf.summary.scalar('auc_val', eval_metric_ops['auc'])
        tf.summary.scalar('mse_val', eval_metric_ops['mse'])
        tf.summary.scalar('precision_val', eval_metric_ops['precision'])
        tf.summary.scalar('recall_val', eval_metric_ops['recall'])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops, scaffold=scaffold)

def create_estimator(steps_per_epoch, checkpoint = None):
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True #pylint: disable=E1101

    if FLAGS.num_gpus == 0:
        distribution = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
    elif FLAGS.num_gpus == 1:
        distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)

    config = tf.estimator.RunConfig(train_distribute=distribution, 
                                    session_config=sess_config,
                                    model_dir=helpers.define_model_dir(FLAGS),
                                    tf_random_seed=1,
                                    save_checkpoints_steps=int(steps_per_epoch / 2 ),
                                    keep_checkpoint_every_n_hours=2,
                                    keep_checkpoint_max=32,
                                    save_summary_steps= 200,
                                    log_step_count_steps = 100)

    if(FLAGS.model_name in ['mobilenet', 'VGG16', 'inception-v3', 'i3d-keras']):
       estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model(), config=config)
    elif(FLAGS.model_name in ['i3d', 'i3d_v4', 'c3d', 'inception_v1', 'inception_v4', 'mobilenet_v2']):

        if FLAGS.model_name in ['i3d', 'i3d_v4', 'inception_v1', 'inception_v4', 'mobilenet_v2']:
            _, _, pattern = fine_tune.get_scope_and_patterns_to_exclude(FLAGS.model_name)
            ws = tf.estimator.WarmStartSettings(checkpoint or helpers.assembly_ws_checkpoint_path(FLAGS),
                                            pattern)
        else:
            ws = None
        # ws = None
        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                                config=config,
                                                model_dir=config.model_dir,
                                                warm_start_from=ws)
    else:
        raise ValueError('Unsupported network model')

    return estimator

def main(stop_event):

    global dataset_loader
    global training_set_length

    if FLAGS.gpu_to_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_to_use

    helpers.check_and_create_directories(FLAGS)

    time_hist = TimeHistory()
    tf.train.get_or_create_global_step()

    # Getting validation and training sets
    network_training_set, network_training_set_count, network_validation_set, network_validation_set_count = helpers.get_splits(FLAGS)
    training_set_length = len(network_training_set)
    steps_per_epoch = int(training_set_length/(FLAGS.batch_size * max(1, FLAGS.num_gpus)))
    training_set_max_steps = int(FLAGS.epochs * steps_per_epoch)
    print('training set length: {}, epochs: {}, num_gpus: {}, batch_size: {}, steps per epoch: {}, max steps: {}'.format(training_set_length,
            FLAGS.epochs, FLAGS.num_gpus, FLAGS.batch_size, steps_per_epoch, training_set_max_steps))

    validation_set_length = len(network_validation_set)
    validation_set_max_steps = int(validation_set_length/(FLAGS.batch_size)) # * max(1, FLAGS.num_gpus)))
    print('validation set length: {}, max steps {}'.format(len(network_validation_set), validation_set_max_steps))

    dataset_loader = None
    dataset_loader = VideoLoader(FLAGS.dataset_dir, frame_shape=FLAGS.image_shape, stop_event=stop_event)

    if FLAGS.train and FLAGS.eval:

        estimator = create_estimator(steps_per_epoch)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(network_training_set,
                                                    shuffle= not FLAGS.dataset_to_memory,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=FLAGS.epochs,
                                                    prefetch_buffer_size=FLAGS.batch_size * 3,
                                                    fragments_count=network_training_set_count,
                                                    dataset_in_memory=FLAGS.dataset_to_memory),
                                                max_steps= training_set_max_steps,
                                                hooks=[time_hist])

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(network_validation_set,
                                                    shuffle=False,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=None,
                                                    fragments_count=network_validation_set_count,
                                                    dataset_in_memory=True),
                                                steps=validation_set_max_steps,
                                                start_delay_secs=FLAGS.eval_interval_secs,
                                                throttle_secs=FLAGS.eval_interval_secs,
                                                hooks=[time_hist])

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.train and not FLAGS.eval:

        estimator = create_estimator(steps_per_epoch)
        estimator.train(input_fn=lambda:input_fn(network_training_set,
                                            shuffle= not FLAGS.dataset_to_memory,
                                            batch_size=int(FLAGS.batch_size),
                                            num_epochs=FLAGS.epochs,
                                            prefetch_buffer_size=FLAGS.batch_size * 3,
                                            fragments_count=network_training_set_count,
                                            dataset_in_memory=FLAGS.dataset_to_memory),
                                            max_steps=training_set_max_steps,
                                            hooks=[time_hist])

    if not FLAGS.train and FLAGS.eval:
        # network_training_set, network_validation_set = helpers.get_splits(FLAGS)
        # validation_set_length = len(network_validation_set)
        # validation_set_max_steps = int(validation_set_length/(FLAGS.batch_size * max(1, FLAGS.num_gpus)))
        # print('validation set length: {}, max steps {}'.format(len(network_validation_set), validation_set_max_steps))

        estimator = create_estimator(validation_set_max_steps)
        estimator.evaluate(input_fn=lambda:input_fn(network_validation_set,
                                                    shuffle=False,
                                                    batch_size=int(FLAGS.batch_size),
                                                    num_epochs=None,
                                                    fragments_count=network_validation_set_count,
                                                    dataset_in_memory=True),
                                                    steps=validation_set_max_steps,
                                                    hooks=[time_hist])

    if FLAGS.eval_all:
        checkpoints = tf.train.get_checkpoint_state(helpers.assembly_model_dir(FLAGS)).all_model_checkpoint_paths
        for checkpoint in checkpoints:
            estimator = create_estimator(validation_set_max_steps, checkpoint)
            estimator.evaluate(input_fn=lambda:input_fn(network_validation_set,
                                            shuffle=False,
                                            batch_size=int(FLAGS.batch_size),
                                            num_epochs=None,
                                            fragments_count=network_validation_set_count,
                                            dataset_in_memory=True),
                                            steps=validation_set_max_steps,
                                            hooks=[time_hist])

    if FLAGS.predict:
        sets_to_extract = helpers.get_sets_to_extract(FLAGS)

        # complete_set = list(set([x.split('_')[0] for x in sets_to_extract]))
        dataset_loader = None
        gc.collect()
        # dataset_loader = VideoLoader(FLAGS.dataset_dir, videos_to_load=complete_set, frame_shape=FLAGS.image_shape, stop_event=stop_event)
        dataset_loader = VideoLoader(FLAGS.dataset_dir, frame_shape=FLAGS.image_shape, stop_event=stop_event)
        # dataset_loader.start()
        # time.sleep(60)


        for set_name, set_to_extract in sets_to_extract.items():
            print('Will Extract {} set'.format(set_name))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(len(set_to_extract[0])) 
            print("##########################")
            print(len(set_to_extract[1]))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            steps_per_epoch = int(len(set_to_extract[0])/(FLAGS.batch_size * max(1, FLAGS.num_gpus)))
            estimator = create_estimator(steps_per_epoch)
            pred_generator = estimator.predict( input_fn=lambda:input_fn(set_to_extract[0],
                                                    shuffle=False,
                                                    batch_size=FLAGS.batch_size,
                                                    num_epochs=None,
                                                    fragments_count=set_to_extract[1],
                                                    dataset_in_memory=True),
                                                predict_keys=['snippet_id', 'truth_label', 'features'],
                                                hooks=[time_hist])
            save_extracted_features(FLAGS, set_name, len(set_to_extract[0]), pred_generator)
            print('Going out of {} set'.format(set_name))


    total_time =  sum(time_hist.times)
    print('total time with ', FLAGS.num_gpus, 'GPUs:', total_time, 'seconds')

    avg_time_per_batch = np.mean(time_hist.times)
    print(FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch, 'images/second with', FLAGS.num_gpus, 'GPUs')



if __name__ == '__main__':
    stop_event = Event() # used to signal termination to the threads
    try:
        tf.logging.set_verbosity(tf.logging.INFO)
        main(stop_event)
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()