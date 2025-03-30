import tensorflow as tf

def net(image, conv_weights):
    """Applies the style transfer transformation"""
    conv1 = _conv_layer(image, conv_weights[0], conv_weights[1], 32, 9, 1)
    conv2 = _conv_layer(conv1, conv_weights[2], conv_weights[3], 64, 3, 2)
    conv3 = _conv_layer(conv2, conv_weights[4], conv_weights[5], 128, 3, 2)

    res1 = _residual_block(conv3, conv_weights[6:10])
    res2 = _residual_block(res1, conv_weights[10:14])
    res3 = _residual_block(res2, conv_weights[14:18])
    res4 = _residual_block(res3, conv_weights[18:22])
    res5 = _residual_block(res4, conv_weights[22:26])

    conv_t1 = _conv_transpose_layer(res5, conv_weights[26], conv_weights[27], 64, 3, 2)
    conv_t2 = _conv_transpose_layer(conv_t1, conv_weights[28], conv_weights[29], 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, conv_weights[30], conv_weights[31], 3, 9, 1, relu=False)

    return tf.nn.tanh(conv_t3) * 150 + 255./2

def _conv_layer(x, weight, bias, num_filters, filter_size, strides, relu=True):
    """Standard Convolutional Layer"""
    x = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME') + bias
    if relu:
        x = tf.nn.relu(x)
    return x

def _conv_transpose_layer(x, weight, bias, num_filters, filter_size, strides):
    # If x has rank 5, squeeze out the extra dimension (assuming it's spurious)
    if x.get_shape().ndims == 5:
        x = tf.squeeze(x, axis=1)
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    new_shape = [batch_size, height * strides, width * strides, num_filters]
    return tf.nn.conv2d_transpose(x, weight, new_shape, strides=[1, strides, strides, 1], padding='SAME') + bias


def _residual_block(x, weights):
    """Residual Block"""
    return x + _conv_layer(_conv_layer(x, weights[0], weights[1], 128, 3, 1), weights[2], weights[3], 128, 3, 1, relu=False)
