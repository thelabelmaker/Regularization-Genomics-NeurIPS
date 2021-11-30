import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomNorm(Layer):
    def __init__(self):
        super(CustomNorm, self).__init__()
    

    def call(self, x, training=False):
        if training:
            return (x - tf.math.reduce_mean(x, axis=0))/(tf.math.reduce_std(x, axis=0) + 1e-10)
        else:
            return (x - tf.math.reduce_mean(x, axis=0))/(tf.math.reduce_std(x, axis=0) + 1e-10)