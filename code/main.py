import argparse
from MixupMode import MixupMode
from NormalModel import NormalModel
from ManifoldMixup import ManifoldMixupModel
from models import deepnet
from utils import *
from DataLoader import load_data
import enum
import tensorflow as tf
from tensorflow import keras
import tfomics
import pandas as pd


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--dropout',action='store_true')
    parser.add_argument('--batchnorm',action='store_true')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--d', type=float)
    parser.add_argument('--f', type=float)
    args = parser.parse_args()
    k = 0
    ka=0
    if args.mixup:
        mixup_mode = MixupMode.RANDOM_MIXUP
        k = 0
        ka = None
    else:
        mixup_mode = MixupMode.NO_MIXUP
    
    path =  args.filepath
    d = [0., 0., 0., 0., 0.]
    if args.dropout:
        d = [0.1, 0.2, 0.3, 0.4, 0.0]
        
        for i in range(len(d)):
            d[i] = args.d
        d[-1] = .5
        
    
    if args.batchnorm:
        bn=[True, True, True, True, True]
    else:
        bn=[False, False, False, False, False,]

    units=np.array([24, 32, 48, 64, 96])*args.f

    alpha = 1.0
    
    x_train, x_valid, x_test, y_train, y_valid, y_test, model_test = load_data()
    N, L, A = x_train.shape
    
    model = NormalModel(L=L, A=A, mixup_mode=MixupMode.NO_MIXUP, bn=bn, d=d, units=units, training=False)
    model = build_model(model, .001)
    history_a = burn_in(model, x_train, y_train, x_valid, y_valid, args.bs, '{}/history'.format(path), model_test, x_test, y_test)
        
    if mixup_mode == MixupMode.RANDOM_MIXUP:
        model = ManifoldMixupModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, k=k, ka=ka, training=True, weights=model.model)
    else:
        model = NormalModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, training=True, weights=model.model)

    model = build_model(model, .01)

    history_b = train_model(model, x_train, y_train, x_valid, y_valid, args.bs, '{}/history'.format(path), model_test, x_test, y_test)

    model.model.save("{}/weights_no_regulated.h5".format(path), save_format='tf')

    get_attribution(model, model_test, x_test, y_test, "{}/results_no_reg.tsv".format(path))
    


if __name__=='__main__':
    main()
