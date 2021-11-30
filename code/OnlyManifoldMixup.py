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


class TargetedManifoldMixupModel(keras.Model):
    def __init__(self, alpha=1., name="mixup_model", L=100, A = 4, mixup_mode=MixupMode.NO_MIXUP, bn=[None], d=[None], k = 0, ka=None, model='deepnet', units=None):
        
        # ks -> List of layer activation indices to use for manifold mixup. 
        
        super().__init__()
        if model == 'deepnet':
            self.model = deepnet(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
        elif model == 'shallownet_4':
            self.model = shallownet_4(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
        else:
            self.model = shallownet_25(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
        self.lam_dist = tfd.Beta(alpha, alpha)
        self.mixup_mode = mixup_mode
        if mixup_mode == MixupMode.NO_MIXUP or mixup_mode== MixupMode.INPUT_MIXUP:
            raise ValueError
        else:
            ks = get_mixup_layers(self.model)
            if ka==None:
                ka = len(ks)
            ks = ks[k:ka]
        self.ks = ks
        print(self.model.layers)
        print(self.ks)
    
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

        
    def train_step(self, data):
        x, y = data
        y_pred, yn_pred = x, x
        k = np.random.randint(0, len(self.ks))
        lam = self.lam_dist.sample()
        ym = y
        # record differentiable ops
        with tf.GradientTape() as tape:
            for i, layer in enumerate(self.model.layers):
                y_pred = layer(y_pred, training=True)
                yn_pred = layer(yn_pred, training=True)
                if layer == self.ks[k] and self.mixup_mode==MixupMode.RANDOM_MIXUP:
                    xp, xn, yp, yn = [], [], [], []
                    for j in range(y.shape[0]):
                        if y[j] == [0]:
                            xn.append(y_pred[j])
                            yn.append(y[j])
                        else:
                            xp.append(y_pred[j])
                            yp.append(y[j])
                    while not (len(xp) == len(xn)):
                        if len(xp) < len(xn):
                            ind = np.random.randint(0, len(xp))
                            xp.append(xp[ind])
                            yp.append(yp[ind])
                        else:
                            ind = np.random.randint(0, len(xn))
                            xn.append(xn[ind])
                            yn.append(yn[ind])
                    xc, yc = [], []
                    for m in range(tf.shape(yn_pred)[0]):
                        lam = self.lam_dist.sample()
                        ind_p = np.random.randint(0, len(xp))
                        ind_n = np.random.randint(0, len(xn))
                        xc.append(lam*xn[ind_n] + (1-lam)*xp[ind_p])
                        yc.append(lam*yn[ind_n] + (1-lam)*yp[ind_p])
                    y = tf.convert_to_tensor(yc)
                    y_pred = tf.convert_to_tensor(xc)

            # calcalate total loss 
            loss1 = self.loss(y, y_pred)
            loss2 = self.loss(ym, yn_pred)
            loss = loss1 + loss2
            loss = loss + sum(self.model.losses)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)        
        # take optimization step 
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # update metrics
        self.compiled_metrics.update_state(yn_pred, ym)
        #tf.print("AFTER LOSS")
        # return dictionary of metrics 
        return {m.name: m.result() for m in self.metrics}