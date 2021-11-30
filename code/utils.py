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

def get_sal_map(model, model_test, x_test, y_test, save_path, i):
    # get positive label sequences and sequence model
    pos_index = np.where(y_test[:,0] == 1)[0]   

    num_analyze = len(pos_index)
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]

    # instantiate explainer class
    explainer = tfomics.explain.Explainer(model, class_index=0)

    # calculate attribution maps
    saliency_scores = explainer.saliency_maps(X)

    # reduce attribution maps to 1D scores
    sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)

    # compare distribution of attribution scores at positions with and without motifs
    threshold = 0.1
    saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, X_model, threshold)
    print("%s: %.3f+/-%.3f"%('saliency', np.mean(saliency_roc), np.std(saliency_roc)))
    print("%s: %.3f+/-%.3f"%('saliency', np.mean(saliency_pr), np.std(saliency_pr)))
    scores = [saliency_roc]
    names = ['Saliency']
    for index in [2, 5, 10, 15, 20]:     # sequence index
        x = np.expand_dims(X[index], axis=0)

        # convert attribution maps to pandas dataframe for logomaker
        scores = np.expand_dims(saliency_scores[index], axis=0)
        saliency_df = tfomics.impress.grad_times_input_to_df(x, scores)
        print('%s: %.3f - %.3f'%('saliency', saliency_roc[index], saliency_pr[index]))
        with open(save_path, 'a') as f:
            f.write('%.4f\t'%(saliency_roc[index]))
            f.write('%.4f\t'%(saliency_pr[index]))
            
        # ground truth sequence model
        model_df = tfomics.impress.prob_to_info_df(X_model[index])

        # plot comparison
        fig = plt.figure(figsize=(20,2))
        ax = plt.subplot(2,1,1)
        tfomics.impress.plot_attribution_map(saliency_df, ax, figsize=(20,1))
        plt.ylabel('Saliency')
        ax = plt.subplot(2,1,2)
        tfomics.impress.plot_attribution_map(model_df, ax, figsize=(20,1))
        plt.ylabel('Model');

        plt.savefig(save_path[:-12]+'_epoch_{}_seq_{}_figure.jpg'.format(i, index))
    with open(save_path, 'a') as f:
        f.write("\n")



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

def sal_graphs(epoch, model, model_test, x_test, y_test, save_path):
    # number of test sequences to analyze (set this to 500 because expintgrad takes long)
    for index in [5, 10, 20, 25]:
        # get positive label sequences and sequence model
        pos_index = np.where(y_test[:,0] == 1)[0]   

        num_analyze = len(pos_index)
        X = x_test[pos_index[:num_analyze]]
        X_model = model_test[pos_index[:num_analyze]]

        # instantiate explainer class
        explainer = tfomics.explain.Explainer(model, class_index=0)

        # calculate attribution maps
        saliency_scores = explainer.saliency_maps(X)
        # reduce attribution maps to 1D scores
        sal_scores = tfomics.explain.grad_times_input(X, saliency_scores)
        fig = plt.figure(figsize=(20,7))
        ax = plt.subplot(6, 1, 1)
        x = np.array(range(len(sal_scores[index])))
        y = sal_scores[index]
        #cubic_interploation_model = interp1d(x, y, kind = "cubic")
        #X_ = np.linspace(x.min(), x.max(), 500)
        #Y_ = cubic_interploation_model(X_)
        norm = colors.Normalize(vmin=0, vmax=40)
        ax.plot(x, y, color=plt.cm.hot(norm(epoch)))
        plt.xticks([], [])
        plt.yticks([], [])
        #plt.ylim([-.25, 1])
        plt.xlim([0, len(sal_scores[index])-1])
        x = np.expand_dims(X[index], axis=0)
        #plt.savefig(save_path+' {}, epoch {}.pdf'.format(index, epoch), bbox_inches='tight', dpi=150)
        scores_df = pd.DataFrame(np.transpose(saliency_scores[index]))
        scores_df.to_csv(save_path+' {}, epoch {}.csv'.format(index, epoch), index=False)

def make_heatmap(model, model_test, x_test, y_test, save_path):
    norm = colors.Normalize(vmin=0, vmax=40) 
    cmap = plt.cm.jet
    threshold=0.1
    for index in [5, 10, 20, 25]:
        fig = plt.figure(figsize=(20,4))
        ax = plt.subplot(3, 1, 1)
        for epoch in range(0, 40):
            
            pos_index = np.where(y_test[:,0] == 1)[0]   
            num_analyze = len(pos_index)
            X = x_test[pos_index[:num_analyze]]
            X_model = model_test[pos_index[:num_analyze]]

            # instantiate explainer class

            # calculate attribution maps
            saliency_scores = np.expand_dims(np.transpose(np.array(pd.read_csv(save_path+' {}, epoch {}.csv'.format(index, epoch)))), axis=0)
            # reduce attribution maps to 1D scores
            top_k = 20
            sal_scores = tfomics.explain.grad_times_input(np.expand_dims(X[index], axis=0), saliency_scores)
            saliency_roc, saliency_pr = tfomics.evaluate.interpretability_performance(sal_scores, np.expand_dims(X_model[index], axis=0), threshold)
            x = np.array(range(len(sal_scores[0])))
            y = sal_scores[0]
            #cubic_interploation_model = interp1d(x, y, kind = "cubic")
            #X_ = np.linspace(x.min(), x.max(), 500)
            #Y_ = cubic_interploation_model(X_)
            
            
            ax.plot(x, y, color=cmap(norm(epoch)), alpha=.5)
            #plt.colorbar()
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlim([0, len(sal_scores[0])-1])
            # get positive label sequences and sequence model
            
            sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = tfomics.evaluate.signal_noise_stats(sal_scores, np.expand_dims(X_model[index], axis=0), top_k, threshold)
            with open(save_path+'seq {}.tsv'.format(index), 'a') as f:
                f.write(str(epoch)+'\t')
                f.write(str(sal_signal[0])+'\t')
                f.write(str(sal_noise_mean[0])+'\t')
                f.write(str(sal_noise_topk[0])+'\t')
                f.write(str(saliency_roc[0])+'\t')
                f.write(str(saliency_pr[0])+'\t')
                f.write('\n')
            
        
        gt_info = np.log2(4) + np.sum(X_model[index]*np.log2(X_model[index]+1e-10),axis=1)
        # set label if information is greater than 0
        label = np.zeros(gt_info.shape)
        label[gt_info > threshold] = 1

        x = np.array(range(len(model_test[0])))
        y = label
        ax = plt.subplot(3, 1, 2)
        ax.plot(x, y, color='red')
        plt.xlim([0, len(sal_scores[0])-1])
        plt.xticks([], [])
        plt.yticks([], [])
        ax = plt.subplot(3, 1, 2)
        plt.xticks([], [])
        plt.yticks([], [])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=5)
        plt.savefig(save_path + ' sal map {}.pdf'.format(index), bbox_inches='tight', dpi=150)




#plt.savefig('../sal map 5.pdf', bbox_inches='tight', dpi=150)


def get_mixup_layers(model):
    mixup_layers = []
    for l in range(len(model.layers)):
        if isinstance(model.layers[l], tf.keras.layers.Activation):
            mixup_layers.append(model.layers[l])
    print(mixup_layers)
    return mixup_layers

def build_model(model):
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    acc =  keras.metrics.BinaryAccuracy (name='acc')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    if len(model.ks)>1 or model.mixup_mode == MixupMode.TARGETED_MIXUP:
        eager = True
    else:
        eager = False
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], run_eagerly=eager)
    return model

def build_model_lr(model):
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    acc =  keras.metrics.BinaryAccuracy (name='acc')
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    if len(model.ks)>1 or model.mixup_mode == MixupMode.TARGETED_MIXUP:
        eager = True
    else:
        eager = False
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], run_eagerly=eager)
    return model

def train_model(model, x_train, y_train, x_valid, y_valid, bs):
# early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=10, 
                                                verbose=1, 
                                                mode='min', 
                                                restore_best_weights=True)
    # reduce learning rate callback
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.2,
                                                    patience=4, 
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1) 
    
    # train model
    history = model.fit(x_train, y_train, 
                        epochs=100,
                        batch_size=bs, 
                        shuffle=True,
                        validation_data=(x_valid, y_valid), 
                        callbacks=[es_callback, reduce_lr],
                        verbose=2)
    
    
    return history

def burn_in(model, x_train, y_train, x_valid, y_valid, bs, path, model_test, x_test, y_test):
    val_path = path+'_val_data.tsv'
    with open(val_path, 'w') as f:
        f.write('epoch\t')
        f.write('val loss\t')
        f.write('network acc\t')
        f.write('network auroc\t')
        f.write('network aupr\t')
        f.write('\n')
    """
    fig_path = path+'_seq_sal.tsv'
    with open(fig_path, 'w') as f:
        f.write('epoch\t')
        f.write('seq 2 roc\t')
        f.write('seq 2 pr\t')
        f.write('seq 5 roc\t')
        f.write('seq 5 pr\t')
        f.write('seq 10 roc\t')
        f.write('seq 10 pr\t')
        f.write('seq 15 roc\t')
        f.write('seq 15 pr\t')
        f.write('seq 20 roc\t')
        f.write('seq 20 pr\t')
        f.write('\n')
    """    
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

        """
        with open(fig_path, 'a') as f:
            f.write('{}\t'.format(i))
        get_sal_map(model, model_test, x_test, y_test, fig_path, i)
        """
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


def train_model_no_reg(model, x_train, y_train, x_valid, y_valid, bs, path, model_test, x_test, y_test):
    # train model
    val_path = path+'_val_data.tsv'
    fig_path = path+'_seq_sal.tsv'
    graph_path = path + ' graph seq'
    path = path+'_no_reg.tsv'
    
    for i in range(38):
        with open(path, 'a') as f:
            f.write('{}\t'.format(i+2))
        get_attribution(model, model_test, x_test, y_test, path)

        """
        if i+2 in [0, 1, 3, 5, 7, 10, 15, 25, 50,]:
            with open(fig_path, 'a') as f:
                f.write('{}\t'.format(i+2))
            get_sal_map(model, model_test, x_test, y_test, fig_path, i+2)
        """
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
    #get_sal_map(model, model_test, x_test, y_test, fig_path, 55)

    print(model.model.summary())


    return history

def evaluate_model(model, x_test, y_test):
    results = model.evaluate(x_test, y_test)
    print(results)
    return results

