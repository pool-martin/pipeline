from __future__ import absolute_import
import tensorflow as tf
slim = tf.contrib.slim

def init_weights(scopes_to_exclude, patterns_to_exclude, path):
    if path == None:
        return

    print('Finetunning based on: ', path)
    # look for checkpoint
    model_path = tf.train.latest_checkpoint(path)
    initializer_fn = None

    if model_path:
        # only restore variables in the scope_name scope
        variables_to_restore = slim.get_variables_to_restore()
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=scopes_to_exclude)
        for pattern in patterns_to_exclude:
            variables_to_restore = [v for v in variables_to_restore if pattern not in v.name ]

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
    if model_name == 'inception_v4':
        scopes_to_exclude = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        pattern_to_exclude = ['biases', "global_step"]
    if model_name == 'mobilenet_v2':
        scopes_to_exclude = ["MobilenetV2/Logits"]
        pattern_to_exclude = []
    elif model_name == 'i3d':
        scopes_to_exclude = ["RGB/inception_i3d/Logits"]
        pattern_to_exclude = ["global_step"]
    return scopes_to_exclude, pattern_to_exclude

def get_variables_to_restore(model_name):
    scopes_to_exclude, patterns_to_exclude = get_scope_and_patterns_to_exclude(model_name)
    variables_to_restore = slim.get_variables_to_restore(exclude=scopes_to_exclude)
    for pattern in patterns_to_exclude:
        variables_to_restore = [v for v in variables_to_restore if pattern not in v.name ]
    return variables_to_restore
