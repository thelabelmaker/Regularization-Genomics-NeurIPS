import numpy as np
import tfomics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
from tensorflow import keras
from models import *
import csv
from MixupMode import MixupMode
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cm


def get_attribution(model, model_test, x_test, y_test, save_path):
    print(save_path)
    
    print("\nAttribution Scores")
    pos_index = np.where(y_test[:,0] == 1)[0]   
    num_analyze = len(pos_index)
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]
    explainer = tfomics.explain.Explainer(model, class_index=0)
    threshold = 0.1
    # instantiate explainer classepoch: 0
    saliency_scores = explainer.saliency_maps(X)
    sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)
    saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, X_model, threshold)
    top_k = 20
    sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = tfomics.evaluate.signal_noise_stats(sal_scores, X_model, top_k, threshold)
    snr = snr = sal_signal/sal_noise_topk
    snr[np.isnan(snr)] = 0
    scores_snr = [snr]
    snr_means = np.mean(scores_snr, axis=1)
    print(np.shape(scores_snr))

    results = evaluate_model(model, x_test, y_test)
    scores_roc = [saliency_roc]
    roc_means = np.mean(scores_roc, axis=1)
    scores_pr = [saliency_pr]
    pr_means = np.mean(scores_pr, axis=1)
    print(np.shape(scores_roc))
    print(np.shape(scores_pr))

    with open(save_path, 'a') as f:
        f.write('%.4f\t'%(results[3]))
        f.write('%.4f\t'%(results[2]))
        for val in roc_means:
            f.write('%.4f\t'%(val))
        for val in snr_means:
            f.write('%.4f\t'%(val))
        for val in pr_means:
            f.write('%.4f\t'%(val))
        f.write("\n")

def get_mixup_layers(model):
    mixup_layers = []
    for l in range(len(model.layers)):
        if isinstance(model.layers[l], tf.keras.layers.Activation):
            mixup_layers.append(model.layers[l])
    print(mixup_layers)
    return mixup_layers

def build_model(model, lr):
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    acc =  keras.metrics.BinaryAccuracy (name='acc')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    if model.mixup_mode==MixupMode.RANDOM_MIXUP:
        eager = True
    else:
        eager = False
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], run_eagerly=eager)
    return model

def burn_in(model, x_train, y_train, x_valid, y_valid, bs, path, model_test, x_test, y_test):
    val_path = path+'_val_data.tsv'
    with open(val_path, 'w') as f:
        f.write('epoch\t')
        f.write('val loss\t')
        f.write('network acc\t')
        f.write('network auroc\t')
        f.write('network aupr\t')
        f.write('\n')

    graph_path = path + ' graph seq'
    path = path+'_no_reg.tsv'
    with open(path, 'w') as f:
            f.write('epoch\t')
            f.write('network aupr\t')
            f.write('network auroc\t')
            f.write('sal auroc\t')
            f.write('snr\t')
            f.write('sal aupr\t')
            f.write('\n')

    for i in range(2):
        with open(path, 'a') as f:
            f.write('{}\t'.format(i))
        get_attribution(model, model_test, x_test, y_test, path)

        with open(val_path, 'a') as f:
            f.write('{}\t'.format(i))
            metrics = evaluate_model(model, x_valid, y_valid)
            for i in metrics:
                f.write(str(i)+'\t')
            f.write('\n')
        history = model.fit(x_train, y_train, 
                            epochs=1,
                            batch_size=bs, 
                            shuffle=True,
                            validation_data=(x_valid, y_valid),
                            verbose=2)
    print(model.model.summary())

    return history


def train_model(model, x_train, y_train, x_valid, y_valid, bs, path, model_test, x_test, y_test):
    # train model
    val_path = path+'_val_data.tsv'
    fig_path = path+'_seq_sal.tsv'
    graph_path = path + ' graph seq'
    path = path+'_no_reg.tsv'
    
    for i in range(38):
        with open(path, 'a') as f:
            f.write('{}\t'.format(i+2))
        get_attribution(model, model_test, x_test, y_test, path)

        with open(val_path, 'a') as f:
            f.write('{}\t'.format(i+2))
            metrics = evaluate_model(model, x_valid, y_valid)
            for j in metrics:
                f.write(str(j)+'\t')
            f.write('\n')
        history = model.fit(x_train, y_train, 
                            epochs=1,
                            batch_size=bs, 
                            shuffle=True,
                            validation_data=(x_valid, y_valid),
                            verbose=2)
    with open(path, 'a') as f:
        f.write('40\t'.format())
    get_attribution(model, model_test, x_test, y_test, path)

    print(model.model.summary())


    return history

def evaluate_model(model, x_test, y_test):
    results = model.evaluate(x_test, y_test)
    print(results)
    return results

