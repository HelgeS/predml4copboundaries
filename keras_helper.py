from keras.layers.merge import _Merge
import keras.backend as K
import tensorflow as tf


class Minimum(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = K.minimum(output, inputs[i])
        return output


def minimum(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the element-wise minimum of the inputs.
    """
    return Minimum(**kwargs)(inputs)


class Median(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        v = K.reshape(inputs, [-1])
        m = v.get_shape()[0] // 2
        return K.nn.top_k(v, m).values[m - 1]


def median(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the element-wise minimum of the inputs.
    """
    return Median(**kwargs)(inputs)


class TopKValue(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """
    def __init__(self, k=1, **kwargs):
        super(TopKValue, self).__init__(**kwargs)
        self.k = k

    def _merge_function(self, inputs):
        v = K.reshape(inputs, [-1])
        m = v.get_shape()[0]
        return tf.nn.top_k(v, self.k).values[m - 1]


def topKvalue(inputs, k=1, **kwargs):
    """Functional interface to the `Minimum` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the element-wise minimum of the inputs.
    """
    return TopKValue(k, **kwargs)(inputs)
