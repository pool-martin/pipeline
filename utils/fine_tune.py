from __future__ import absolute_import
import tensorflow as tf
slim = tf.contrib.slim
import os
import numpy as np

def init_weights(model_name, path):
    if path == None:
        return

    print('Finetunning based on: ', path)
    # look for checkpoint
    model_path = tf.train.latest_checkpoint(path)
    initializer_fn = None

    if model_path:
        variables_to_restore = get_variables_to_restore(model_name)

        # Create the saver which will be used to restore the variables.
        initializer_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore, ignore_missing_vars=True, reshape_variables=True)
    else:
        print("could not find the fine tune ckpt at {}".format(path))
        exit()

    def InitFn(scaffold,sess):
        initializer_fn(sess)
    return InitFn

def get_scope_and_patterns_to_exclude(model_name):
    if model_name == 'inception_v1':
        scopes_to_exclude = ["InceptionV1/Logits"]
        pattern_to_exclude = ["global_step"]
        pattern_to_restore = ".*InceptionV1/[C|M].*"
    if model_name == 'inception_v4':
        scopes_to_exclude = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        pattern_to_exclude = ['biases', "global_step"]
        pattern_to_restore = ".*InceptionV4/[C|M].*"
    if model_name == 'mobilenet_v2':
        scopes_to_exclude = ["MobilenetV2/Logits"]
        pattern_to_exclude = []
    elif model_name == 'i3d':
        scopes_to_exclude = ["RGB/inception_i3d/Logits"]
        pattern_to_exclude = ["global_step"]
        pattern_to_restore = ".*inception_i3d/[C|M].*"
    elif model_name == 'i3d_v4':
        scopes_to_exclude = []
        pattern_to_exclude = ["global_step"]
        pattern_to_restore = ".*InceptionV4/[C|M].*"
    return scopes_to_exclude, pattern_to_exclude, pattern_to_restore

def get_variables_to_restore(model_name):
    scopes_to_exclude, patterns_to_exclude, _ = get_scope_and_patterns_to_exclude(model_name)
    variables_to_restore = slim.get_variables_to_restore(exclude=scopes_to_exclude)
    for pattern in patterns_to_exclude:
        variables_to_restore = [v for v in variables_to_restore if pattern not in v.name ]
    return variables_to_restore


# https://github.com/aponamarev/TF_Object_Det/blob/master/LoadVar_from_exisitng_checkpoint.ipynb

# https://stackoverflow.com/questions/17394882/add-dimensions-to-a-numpy-array
# numpy expand dims

# https://stackoverflow.com/questions/39137597/how-to-restore-variables-using-checkpointreader-in-tensorflow

# https://stackoverflow.com/questions/48138041/modifying-shape-of-tensor-in-tensorflow-checkpoint

# https://www.tensorflow.org/api_docs/python/tf/contrib/framework/assign_from_values_fn

def assembly_3d_checkpoint(model_name, path):

    print('assembly_3d_checkpoint based on: ', path)
    checkpoint = {}
    assert os.path.exists(path), "Provided incorrect path to the file. {} doesn't exist".format(path)
    model_path = tf.train.latest_checkpoint(path)
    reader = tf.train.NewCheckpointReader(model_path)
    variables_to_restore = get_variables_to_restore(model_name)

    for variable in variables_to_restore:
        variable_name = variable.name.split(':')[0]
        # print(variable_name)
        if(reader.has_tensor(variable_name)):
            current_value = reader.get_tensor(variable_name)
            if current_value.shape == variable.shape:
                # print('same shape:', variable_name)
                checkpoint[variable_name] = current_value
            else:
                if current_value.shape == variable.shape[1:]:
                    # print('same shape:', variable_name)
                    target_value = np.empty(variable.shape)

                    for k in range(target_value.shape[0]):
                        target_value[k,:,:,:,:] = current_value
                    checkpoint[variable_name] = target_value
                else:
                    print(variable_name, '2D', current_value.shape, '3D', variable.shape)
        else:
            print(variable.name, 'Not in checkpoint: ')


    initializer_fn = tf.contrib.framework.assign_from_values_fn(checkpoint)
    print('assembly_3d_checkpoint: mount values with success! len:', len(checkpoint))

    def InitFn(scaffold,sess):
        initializer_fn(sess)
    return InitFn
