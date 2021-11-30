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
import time


class ManifoldMixupModel(keras.Model):
    def __init__(self, alpha=1., name="mixup_model", L=100, A = 4, mixup_mode=MixupMode.NO_MIXUP, bn=[None], d=[None], k = 0, ka=None, model='deepnet', units=None, training=True, cn=[None], weights=None):
        
        # ks -> List of layer activation indices to use for manifold mixup. 
        
        super().__init__()
        if weights is None:
            if model == 'deepnet':
                self.model = deepnet(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units, cn=cn)
            elif model == 'shallownet_4':
                self.model = shallownet_4(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
            else:
                self.model = shallownet_25(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
        else:
            self.model = weights
        self.lam_dist = tfd.Beta(alpha, alpha)
        self.mixup_mode = mixup_mode
        
        if mixup_mode == MixupMode.NO_MIXUP:
            ks = [None]
        else:
            ks = get_mixup_layers(self.model)
            if ka==None:
                ka = len(ks)
            ks = ks[k:ka]
        self.ks = ks
        self.training = True
        print(self.mixup_mode)
        print(self.model.layers)
        print(self.ks)
        print(self.training)
    
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

        
    def train_step(self, data):
        x, y = data
        if self.mixup_mode == MixupMode.RANDOM_MIXUP:
            k = np.random.randint(0, len(self.ks))
        else:
            k = None
        #g = tf.random.Generator.from_non_deterministic_state()
        #k = int(np.array(g.uniform([1, 1], minval=0, maxval=len(self.ks), dtype=tf.int32)))
        #tf.print(k)
        lam = self.lam_dist.sample(1)
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0])) # shuffled indices 
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0) ## shuffled batch
        ym = yp
        if self.mixup_mode == MixupMode.INPUT_MIXUP:
            #tf.print('input mixup')
            xp = lam*x + (1-lam*xp)
            ym = lam*y + (1-lam)*yp ## mixup the labels
        # record differentiable ops
        with tf.GradientTape() as tape:
            y_pred, ym_pred = x, xp
            for i, layer in enumerate(self.model.layers):
                y_pred, ym_pred = layer(y_pred, training=self.training), layer(ym_pred, training=self.training)
                if self.mixup_mode==MixupMode.RANDOM_MIXUP and layer == self.ks[k]:
                    #tf.print(self.model.layers.index(layer))
                    ym_pred = lam*y_pred + (1-lam)*ym_pred # mixup the representation at the kth layer
                    ym = lam*y + (1-lam)*yp ## mixup the labels
                    #tf.print('mixing')
            
            # calcalate total loss 
            loss1 = self.loss(y, y_pred)
            loss2 = self.loss(ym, ym_pred)
            """
            Dont divide by 2 if doing mixup
            """
            loss = loss1 + loss2
            if self.mixup_mode == MixupMode.NO_MIXUP:
                loss = loss/2
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