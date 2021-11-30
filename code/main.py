import argparse
from MixupMode import MixupMode
from NormalModel import NormalModel
from ManifoldMixup import ManifoldMixupModel
#from OnlyManifoldMixup import TargetedManifoldMixupModel
from models import deepnet, shallownet_4, shallownet_25
from utils import *
from DataLoader import load_data
import enum
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import tfomics
from pdb import set_trace
import pandas as pd


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--overfitting', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--input_mixup', action='store_true')
    parser.add_argument('--targeted_mixup', action='store_true')
    parser.add_argument('--static', type=str)
    parser.add_argument('--deepnet_x1', action='store_true')
    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--deepnet_x2', action='store_true')
    parser.add_argument('--deepnet_x4', action='store_true')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--dropout',action='store_true')
    parser.add_argument('--batchnorm',action='store_true')
    parser.add_argument('--deepnet_x8', action='store_true')
    parser.add_argument('--deepnet_x32', action='store_true')
    parser.add_argument('--deepnet_x128', action='store_true')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--d', type=float)
    parser.add_argument('--rc', action='store_true')
    parser.add_argument('--f', type=float)
    parser.add_argument('--customnorm', action='store_true')
    args = parser.parse_args()
    k = 0
    ka=0
    if args.mixup:
        if len(args.static)<5:
            k = int(args.static[0])-1
            ka = int(args.static[2])
            mixup_mode = MixupMode.RANDOM_MIXUP
        else:
            mixup_mode = MixupMode.RANDOM_MIXUP
            k = 0
            ka = None
    elif args.no_mixup:
        mixup_mode = MixupMode.NO_MIXUP
        k = None
    elif args.targeted_mixup:
        if len(args.static)<5:
            k = int(args.static[0])-1
            ka = int(args.static[2])
            mixup_mode = MixupMode.TARGETED_MIXUP
        else:
            mixup_mode = MixupMode.TARGETED_MIXUP
            k = 0
            ka = None
    elif args.rc:
        mixup_mode = MixupMode.RC
    else:
        mixup_mode = MixupMode.INPUT_MIXUP
    path =  args.filepath
    print(k)
    print(ka)
    d = [0., 0., 0., 0., 0.]
    if args.dropout:
        d = [0.1, 0.2, 0.3, 0.4, 0.0]
        
        for i in range(len(d)):
            d[i] = args.d
        if args.mixup:
            d[-1] = .5
        else:
            d[-1] = .5
        
        
    
    if args.batchnorm:
        bn=[True, True, True, True, True]
    else:
        bn=[False, False, False, False, False,]

    if args.customnorm:
        cn = [True, True, True, True, True]
    else:
        cn=[False, False, False, False, False,]

    units=np.array([24, 32, 48, 64, 96])*args.f

    alpha = 1.0
    
    x_train, x_valid, x_test, y_train, y_valid, y_test, model_test = load_data()
    N, L, A = x_train.shape
    
    model = NormalModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, training=False)
    model = build_model(model)
    history_a = burn_in(model, x_train, y_train, x_valid, y_valid, args.bs, '{}/history'.format(path), model_test, x_test, y_test)
    
    if mixup_mode == MixupMode.RANDOM_MIXUP or mixup_mode == MixupMode.INPUT_MIXUP:
        model = ManifoldMixupModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, k=k, ka=ka, weights=model.model)
    elif mixup_mode == MixupMode.TARGETED_MIXUP:
        model = TargetedManifoldMixupModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, k=k, ka=ka, weights=model.model)
    else:
        model = NormalModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, weights=model.model)
    
    
    if mixup_mode == MixupMode.RANDOM_MIXUP or mixup_mode == MixupMode.INPUT_MIXUP:
        model = ManifoldMixupModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, k=k, ka=ka, cn=cn, training=True)
    elif mixup_mode == MixupMode.TARGETED_MIXUP:
        model = TargetedManifoldMixupModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, k=k, ka=ka)
    else:
        model = NormalModel(L=L, A=A, mixup_mode=mixup_mode, bn=bn, d=d, units=units, training=True)

   
    model = build_model_lr(model)
    print(model.model.layers)
    history_b = train_model_no_reg(model, x_train, y_train, x_valid, y_valid, args.bs, '{}/history'.format(path), model_test, x_test, y_test)
    print(model.model.summary())
    model.model.save("{}/weights_no_regulated.h5".format(path), save_format='tf')
    get_attribution(model, model_test, x_test, y_test, "{}/results_no_reg.tsv".format(path))
    


if __name__=='__main__':
    main()
