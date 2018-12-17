"""Contains the definition of the Inception V4 inflated architecture.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

slim = tf.contrib.slim


class Unit3D(snt.AbstractModule):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, output_channels,
               kernel_shape=(1, 1, 1),
               stride=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               name='unit_3d', 
               reuse=None,
               padding=snt.SAME,
               tf_library='sonnet'):
    """Initializes Unit3D module."""
    super(Unit3D, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._use_batch_norm = use_batch_norm
    self._activation_fn = activation_fn
    self._use_bias = use_bias
    self._tf_library = tf_library
    self._padding = padding
    self._reuse = reuse

  def _build(self, inputs, is_training):
    """Connects the module to inputs.

    Args:
      inputs: Inputs to the Unit3D component.
      is_training: whether to use training mode for snt.BatchNorm (boolean).

    Returns:
      Outputs from the module.
    """
    if self._tf_library == 'sonnet':
        net = snt.Conv3D(output_channels=self._output_channels,
                        kernel_shape=self._kernel_shape,
                        stride=self._stride,
                        padding=self._padding,
                        use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
    else:
        net = slim.conv3d(inputs, 
                        num_outputs=self._output_channels,
                        kernel_size=self._kernel_shape,
                        stride=self._stride,
                        padding=self._padding,
                        ### use_bias=False,
                        normalizer_fn = tf.contrib.layers.batch_norm(is_training=is_training, reuse = self._reuse) if self._use_batch_norm else None,
                        activation_fn=self._activation_fn,
                        reuse = self._reuse)
    return net

class InceptionI3d_v4(snt.AbstractModule):
    """Inception-v4 Inflated 3D ConvNet architecture.

    The model is introduced in:

    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    Joao Carreira, Andrew Zisserman
    https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception v4 architecture, introduced in:

    As described in http://arxiv.org/abs/1602.07261.
    Inception-v4, Inception-ResNet and the Impact of Residual Connections
      on Learning
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv2d_1a_3x3',
        'Conv2d_2a_3x3',
        'Conv2d_2b_3x3',
        'Mixed_3a',
        'Mixed_4a',
        'Mixed_5a',
        'Mixed_5b',
        'Mixed_5c',
        'Mixed_5d',
        'Mixed_5e',
        'Mixed_6a',
        'Mixed_6b',
        'Mixed_6c',
        'Mixed_6d',
        'Mixed_6e',
        'Mixed_6f',
        'Mixed_6g',
        'Mixed_6h',
        'Mixed_7a',
        'Mixed_7b',
        'Mixed_7c',
        'Mixed_7d'
        'AuxLogits',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=2,
                final_endpoint='Conv2d_2b_3x3',
                name='inception_i3d_v4',
                create_aux_logits=True):
        """Initializes I3D-v4 model instance.

        Args:
        num_classes: The number of outputs in the logit layer (default 400, which
            matches the Kinetics dataset).
        spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
            before returning (default True).
        final_endpoint: The model contains many possible endpoints.
            `final_endpoint` specifies the last endpoint for the model to be built
            up to. In addition to the output at `final_endpoint`, all the outputs
            at endpoints up to `final_endpoint` will also be returned, in a
            dictionary. `final_endpoint` must be one of
            InceptionI3d.VALID_ENDPOINTS (default 'Logits').
        name: A string (optional). The name of this module.

        Raises:
        ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d_v4, self).__init__(name=name)
        self._num_classes = num_classes
        self._final_endpoint = final_endpoint
        self._create_aux_logits = create_aux_logits

    def _block_inception_a(self, inputs, scope=None, is_training=True):
        """Builds Inception-A block for Inception v4 inflated network."""
        with tf.variable_scope(scope, 'BlockInceptionA', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1' )(inputs, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_1 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
                branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                name='Conv3d_0c_3x3')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(inputs, ksize=[1, 2, 3, 3, 1], # TODO Confirm the '2'
                                        strides=[1, 1, 1, 1, 1], padding=snt.VALID, name='AvgPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
            return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3]) #TODO Confirm axis


    def _block_reduction_a(self, inputs, scope=None, is_training=True):
        """Builds Reduction-A block for Inception v4 inflated network."""
        with tf.variable_scope(scope, 'BlockReductionA'):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3], padding=snt.VALID,
                        stride=[2, 2, 2], name='Conv3d_1a_3x3')(inputs, is_training=is_training),

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(inputs, is_training=is_training),
                branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                        name='Conv3d_0b_3x3')(branch_1, is_training=is_training),
                branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3], padding=snt.VALID,
                        name='Conv3d_1a_3x3')(branch_1, is_training=is_training),

            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool3d(inputs, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                        padding=snt.VALID, name='MaxPool3d_1a_3x3')

            return tf.concat(axis=4, values=[branch_0, branch_1, branch_2])

    def _block_inception_b(self, inputs, scope=None, is_training=True):
        """Builds Inception-B block for Inception v4 inflated network."""
        with tf.variable_scope(scope, 'BlockInceptionB'):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_1 = Unit3D(output_channels=256, kernel_shape=[1, 7, 1],
                                  name='Conv3d_0b_1x7')(branch_1, is_training=is_training)
                branch_1 = Unit3D(output_channels=256, kernel_shape=[7, 1, 1],
                                  name='Conv3d_0c_1x7')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_2 = Unit3D(output_channels=192, kernel_shape=[7, 1, 1],
                                  name='Conv3d_0b_7x1')(branch_2, is_training=is_training)
                branch_2 = Unit3D(output_channels=224, kernel_shape=[1, 7, 1],
                                  name='Conv3d_0c_1x7')(branch_2, is_training=is_training)
                branch_2 = Unit3D(output_channels=224, kernel_shape=[7, 1, 1],
                                  name='Conv3d_0d_7x1')(branch_2, is_training=is_training)
                branch_2 = Unit3D(output_channels=256, kernel_shape=[1, 7, 1],
                                  name='Conv3d_0e_1x7')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(inputs, ksize=[1, 2, 3, 3, 1], # TODO Confirm the '2'
                                        strides=[1, 1, 1, 1, 1], padding=snt.VALID, name='AvgPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
            return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3]) # TODO Confirm axis

    def _block_reduction_b(self, inputs, scope=None, is_training=True):
        """Builds Reduction-B block for Inception v4 inflated network."""
        with tf.variable_scope(scope, 'BlockReductionB'):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(inputs, is_training=is_training),
                branch_0 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3], stride=[1, 1, 2, 2, 1], 
                                  padding='VALID', name='Conv3d_0b_3x3')(branch_0, is_training=is_training),

            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                        name='Conv3d_0a_1x1')(inputs, is_training=is_training),
                branch_1 = Unit3D(output_channels=256, kernel_shape=[1, 7, 1],
                        name='Conv3d_0b_1x7')(branch_1, is_training=is_training),
                branch_1 = Unit3D(output_channels=320, kernel_shape=[7, 1, 1],
                        name='Conv3d_0c_7x1')(branch_1, is_training=is_training),
                branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3], padding='VALID',
                        stride=[2, 2, 2], name='Conv3d_1a_3x3')(branch_1, is_training=is_training),

            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool3d(inputs, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                        padding='VALID', name='MaxPool3d_1a_3x3')

            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


    def _block_inception_c(self, inputs, scope=None, is_training=True):
        """Builds Inception-C block for Inception v4 inflated network."""
        with tf.variable_scope(scope, 'BlockInceptionC'):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_1 = tf.concat(axis=4, values=[
                            Unit3D(output_channels=256, kernel_shape=[1, 3, 1],
                                name='Conv3d_0b_3x1')(branch_1, is_training=is_training),
                            Unit3D(output_channels=256, kernel_shape=[3, 1, 1],
                                name='Conv3d_0c_1x3')(branch_1, is_training=is_training)])
                            # Unit3D(output_channels=256, kernel_shape=[1, 1, 3],
                            #     name='Conv3d_0b_1x3')(branch_1, is_training=is_training),
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                name='Conv3d_0a_1x1')(inputs, is_training=is_training)
                branch_2 = Unit3D(output_channels=448, kernel_shape=[3, 1, 1],
                                  name='Conv3d_0b_3x1')(branch_2, is_training=is_training)
                branch_2 = Unit3D(output_channels=512, kernel_shape=[1, 3, 1],
                                  name='Conv3d_0c_1x3')(branch_2, is_training=is_training)
                # branch_2 = Unit3D(output_channels=512, kernel_shape=[1, 1, 3],
                #                   name='Conv3d_0d_3x3')(branch_2, is_training=is_training)
                branch_2 = tf.concat(axis=4, values=[
                            Unit3D(output_channels=256, kernel_shape=[1, 3, 1], name='Conv3d_0d_1x3')(branch_2, is_training=is_training),
                            Unit3D(output_channels=256, kernel_shape=[3, 1, 1], name='Conv3d_0e_3x1')(branch_2, is_training=is_training)
                            ])
                            # Unit3D(output_channels=256, kernel_shape=[1, 1, 3],
                            #     name='Conv3d_0f_1x3')(branch_2, is_training=is_training)])

            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(inputs, ksize=[1, 2, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID, name='AvgPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
            return tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

    def _build(self, inputs, is_training, dropout_keep_prob=1.0):
        """Connects the model to inputs.

        Args:
        inputs: Inputs to the model, which should have dimensions
            `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
        is_training: whether to use training mode for snt.BatchNorm (boolean).
        dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
            [0, 1)).

        Returns:
        A tuple consisting of:
            1. Network output at location `self._final_endpoint`.
            2. Dictionary containing all endpoints up to `self._final_endpoint`,
            indexed by endpoint name.

        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == self._final_endpoint

        # 299 x 299 x 3 | 16 x 224 x 224 x 3
        end_point = 'Conv3d_1a_3x3'
        net = Unit3D(output_channels=32, kernel_shape=[3, 3, 3], stride=[2, 2, 2],
                    padding='VALID', name=end_point)(inputs, is_training=is_training)
        if add_and_check_final(end_point, net): return net, end_points

        # 149 x 149 x 32 | 8 x 112 x 112 x 32
        end_point = 'Conv2d_2a_3x3'
        net = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                    padding='VALID', name=end_point)(net, is_training=is_training)
        if add_and_check_final(end_point, net): return net, end_points

        # 147 x 147 x 32 | 8 x 111 x 111 x 32
        end_point = 'Conv2d_2b_3x3'
        net = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                    name=end_point)(net, is_training=is_training)
        if add_and_check_final(end_point, net): return net, end_points

        # # 147 x 147 x 64 | 8 x 111 x 111 x 64
        # with tf.variable_scope('Mixed_3a'):
        #     with tf.variable_scope('Branch_0'):
        #         branch_0 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
        #                                     padding='VALID', name='MaxPool_0a_3x3')
        #     with tf.variable_scope('Branch_1'):
        #         branch_1 =  Unit3D(output_channels=96, kernel_shape=[3, 3, 3], stride=[2, 2, 2],
        #                         padding='VALID', name='Conv2d_0a_3x3')(net, is_training=is_training)

        #     net = tf.concat(axis=4, values=[branch_0, branch_1])
        #     if add_and_check_final('Mixed_3a', net): return net, end_points

        # # 73 x 73 x 160 | 8 x 54 x 54 x 160
        # with tf.variable_scope('Mixed_4a'):
        #     with tf.variable_scope('Branch_0'):
        #         branch_0 =  Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
        #                         name='Conv2d_0a_1x1')(net, is_training=is_training)
        #         branch_0 =  Unit3D(output_channels=96, kernel_shape=[3, 3, 3], padding='VALID',
        #                         name='Conv2d_1a_3x3')(branch_0, is_training=is_training)

        #     with tf.variable_scope('Branch_1'):
        #         branch_1 =  Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv2d_0a_1x1')(net, is_training=is_training)
        #         branch_1 =  Unit3D(output_channels=64, kernel_shape=[1, 1, 7], name='Conv2d_0b_1x7')(branch_1, is_training=is_training)
        #         branch_1 =  Unit3D(output_channels=64, kernel_shape=[1, 7, 1], name='Conv2d_0c_7x1')(branch_1, is_training=is_training)
        #         branch_1 =  Unit3D(output_channels=64, kernel_shape=[7, 1, 1], name='Conv2d_0d_7x1')(branch_1, is_training=is_training)
        #         branch_1 =  Unit3D(output_channels=96, kernel_shape=[3, 3, 3], name='Conv2d_1a_3x3')(branch_1, is_training=is_training)

        #     net = tf.concat(axis=4, values=[branch_0, branch_1])
        #     if add_and_check_final('Mixed_4a', net): return net, end_points

        # # 71 x 71 x 192
        # with tf.variable_scope('Mixed_5a'):
        #     with tf.variable_scope('Branch_0'):
        #         branch_0 =  Unit3D(output_channels=192, kernel_shape=[3, 3, 3], stride=[2, 2, 2], padding='VALID', name='Conv2d_1a_3x3')(net, is_training=is_training)
        #     with tf.variable_scope('Branch_1'):
        #         branch_1 = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='VALID', name='MaxPool_1a_3x3')
        #     net = tf.concat(axis=4, values=[branch_0, branch_1])
        #     if add_and_check_final('Mixed_5a', net): return net, end_points

        # # 35 x 35 x 384
        # # 4 x Inception-A blocks
        # for idx in range(4):
        #     block_scope = 'Mixed_5' + chr(ord('b') + idx)
        #     net = self._block_inception_a(net, block_scope, is_training)
        #     if add_and_check_final(block_scope, net): return net, end_points

        # # 35 x 35 x 384
        # # Reduction-A block
        # net = self._block_reduction_a(net, 'Mixed_6a', is_training)
        # if add_and_check_final('Mixed_6a', net): return net, end_points

        # # 17 x 17 x 1024
        # # 7 x Inception-B blocks
        # for idx in range(7):
        #     block_scope = 'Mixed_6' + chr(ord('b') + idx)
        #     net = self._block_inception_b(net, block_scope, is_training)
        #     if add_and_check_final(block_scope, net): return net, end_points

        # # 17 x 17 x 1024
        # # Reduction-B block
        # net = self._block_reduction_b(net, 'Mixed_7a', is_training)
        # if add_and_check_final('Mixed_7a', net): return net, end_points

        # # 8 x 8 x 1536
        # # 3 x Inception-C blocks
        # for idx in range(3):
        #     block_scope = 'Mixed_7' + chr(ord('b') + idx)
        #     net = self._block_inception_c(net, block_scope, is_training)
        #     if add_and_check_final(block_scope, net): return net, end_points


    # def _build(self, inputs, is_training, dropout_keep_prob=0.8):

    #     """Connects the model to inputs.

    #     Args:
    #     inputs: Inputs to the model, which should have dimensions
    #         `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
    #     is_training: whether to use training mode for snt.BatchNorm (boolean).
    #     dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
    #         [0, 1)).

    #     Returns:
    #     A tuple consisting of:
    #         1. Network output at location `self._final_endpoint`.
    #         2. Dictionary containing all endpoints up to `self._final_endpoint`,
    #         indexed by endpoint name.
    #     """
    #     end_points = {}
        # with tf.variable_scope('InceptionV4'):
        #     net, end_points = self._build_base(inputs, is_training)

        # # Auxiliary Head logits
        # if self._create_aux_logits and self._num_classes:
        #     with tf.variable_scope('AuxLogits'):
        #         # 17 x 17 x 1024
        #         aux_logits = end_points['Mixed_6h']
        #         aux_logits = slim.avg_pool3d(aux_logits, [1, 2, 5, 5, 1], strides=[1, 1, 3, 3, 1], padding='VALID', scope='AvgPool3D_1a_5x5')
        #         aux_logits =  Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_1b_1x1')(aux_logits, is_training=is_training)
        #         aux_logits =  Unit3D(output_channels=768, kernel_shape=aux_logits.get_shape()[1:4], 
        #                             padding='VALID', name='Conv3d_2a')(aux_logits, is_training=is_training)
        #         aux_logits = slim.flatten(aux_logits)
        #         aux_logits = slim.fully_connected(aux_logits, self._num_classes, 
        #                                         activation_fn=None,
        #                                         scope='Aux_logits')
        #         end_points['AuxLogits'] = aux_logits

        # # Final pooling and prediction
        # # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # # can be set to False to disable pooling here (as in resnet_*()).
        # with tf.variable_scope('Logits'):
        #     # 8 x 8 x 1536
        #     kernel_size = net.get_shape()[1:4]
        #     if kernel_size.is_fully_defined():
        #         net = slim.avg_pool3d(net, kernel_size, padding='VALID', scope='AvgPool3D_1a')
        #     else:
        #         net = tf.reduce_mean(net, [1, 2, 3], keep_dims=True,
        #                             name='global_pool')
        #     end_points['global_pool'] = net
        #     if not self._num_classes:
        #         return net, end_points
        #     # 1 x 1 x 1536
        #     net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
        #     net = slim.flatten(net, scope='PreLogitsFlatten')
        #     end_points['PreLogitsFlatten'] = net
        #     # 1536
        #     logits = slim.fully_connected(net, self._num_classes, activation_fn=None,
        #                                     scope='Logits')
        #     end_points['Logits'] = logits
        #     end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
        # return logits, end_points