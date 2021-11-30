import enum
import tensorflow as tf
from tensorflow import keras
import tfomics
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers, Model, Input
import numpy as np
import sys
from models import *
from utils import *
from MixupMode import MixupMode


class NormalModel(keras.Model):
    def __init__(self, alpha=1., name="mixup_model", L=100, A = 4, mixup_mode=MixupMode.NO_MIXUP, bn=[None], d=[None], k = 0, units=None, model='deepnet', weights=None, training=True):
        
        # ks -> List of layer activation indices to use for manifold mixup. 
        
        super().__init__()
        print(d)
        if weights is None:
            if model == 'deepnet':
                self.model = deepnet(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
            elif model == 'shallownet_4':
                self.model = shallownet_4(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
            else:
                self.model = shallownet_25(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
        else:
            self.model = weights
            
        self.ks = []
        self.mixup_mode = mixup_mode
        print(self.mixup_mode)
        self.training = training
        print('Training: '+str(self.training))
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

        
    def train_step(self, data):
        x, y = data
        if self.mixup_mode == MixupMode.GAUSSIAN_NOISE:
            x = x + tf.random.normal(tf.shape(x))

        if self.mixup_mode == MixupMode.RC:
            x_rc = x[:,:,::-1]
            x_rc = x_rc[:,::-1,:]
            x = tf.concat([x,x_rc], axis=0)
            y = tf.concat([y, y], axis=0)

        with tf.GradientTape() as tape:
            y_pred = x
            for i, layer in enumerate(self.model.layers):
                y_pred = layer(y_pred, training=self.training)
            # calcalate total loss 
            loss = self.loss(y, y_pred)
            loss = loss + sum(self.model.losses)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # take optimization step 
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # return dictionary of metrics 
        return {m.name: m.result() for m in self.metrics}