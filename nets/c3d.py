import tensorflow as tf
import tensorflow.contrib.slim as slim

def C3D(input, num_classes, keep_pro=0.2):
    end_points = {}
    with tf.variable_scope('C3D'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu,
                            kernel_size=[3, 3, 3],
                            stride=[1, 1, 1]
                            ):
            net = slim.conv3d(input, 64, scope='conv1')
            end_points['conv1'] = net
            net = slim.max_pool3d(net, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding='VALID', scope='max_pool1')
            end_points['max_pool1'] = net
            net = slim.conv3d(net, 128, scope='conv2')
            end_points['conv2'] = net
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='VALID', scope='max_pool2')
            end_points['max_pool2'] = net
            net = slim.repeat(net, 2, slim.conv3d, 256, scope='conv3')
            end_points['conv3'] = net
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='VALID', scope='max_pool3')
            end_points['max_pool3'] = net
            # net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv4')
            net = slim.conv3d(net, 512, scope='conv4a')
            # net = tf.nn.batch_normalization(net)
            net = slim.conv3d(net, 512, scope='conv4b')
            end_points['conv4b'] = net
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='VALID', scope='max_pool4')
            end_points['max_pool4'] = net
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv5')
            end_points['conv5'] = net
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='VALID', scope='max_pool5')
            end_points['max_pool5'] = net

            # net = tf.reshape(net, [-1, 512 * 4 * 4])
            net = tf.layers.Flatten()(net)
            end_points['flatten'] = net
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc6')
            end_points['fc6'] = net
            net = slim.dropout(net, 0.3, scope='dropout1')
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc7')
            end_points['fc7'] = net
            net = slim.dropout(net, keep_pro, scope='dropout2')
            out = slim.fully_connected(net, num_classes, weights_regularizer=slim.l2_regularizer(0.0005), scope='out')
            end_points['Logits'] = out

            return out, end_points