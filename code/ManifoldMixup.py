import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
from models import deepnet
from utils import *
from MixupMode import MixupMode


class ManifoldMixupModel(keras.Model):
    def __init__(self, alpha=1., name="mixup_model", L=100, A = 4, mixup_mode=MixupMode.NO_MIXUP, bn=[None], d=[None], k = 0, ka=None, model='deepnet', units=None, training=True, weights=None):
        
        # ks -> List of layer activation indices to use for manifold mixup. 
        
        super().__init__()
        if weights is None:
            if model == 'deepnet':
                self.model = deepnet(input_shape = (L, A), l2=False, num_labels = 1, activation='relu', bn=bn, dropout=d, units=units)
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
    
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

        
    def train_step(self, data):
        x, y = data

        k = np.random.randint(0, len(self.ks))

        lam = self.lam_dist.sample(1)
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0])) # shuffled indices 
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0) ## shuffled batch
        ym = yp
        # record differentiable ops
        with tf.GradientTape() as tape:
            y_pred, ym_pred = x, xp
            for i, layer in enumerate(self.model.layers):
                y_pred, ym_pred = layer(y_pred, training=self.training), layer(ym_pred, training=self.training)
                if self.mixup_mode==MixupMode.RANDOM_MIXUP and layer == self.ks[k]:
                    ym_pred = lam*y_pred + (1-lam)*ym_pred # mixup the representation at the kth layer
                    ym = lam*y + (1-lam)*yp ## mixup the labels
            
            # calcalate total loss 
            loss1 = self.loss(y, y_pred)
            loss2 = self.loss(ym, ym_pred)
            """
            Dont divide by 2 if doing mixup
            """
            loss = loss1 + loss2
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