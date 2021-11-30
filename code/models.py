import tensorflow as tf
from tensorflow import keras
from tfomics import layers, utils
import CustomNorm
from CustomNorm import CustomNorm

def deepnet(input_shape, num_labels, activation='relu', 
            units=[24, 32, 48, 64, 96], dropout=[0.1, 0.2, 0.3, 0.4, 0.5], 
            bn=[True, True, True, True, True], l2=None, cn=[False, False, False, False, False]):
    print(dropout)
    print(units)
    print(bn)
    print(activation)
    if l2 is not None:
        l2 = keras.regularizers.l2(l2)
    
    use_bias = []
    for status in bn:
        if status:
            use_bias.append(True)
        else:
            use_bias.append(False)

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=units[0],
                            kernel_size=19,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[0],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(inputs)
    if bn[0]:
        print('batchnorm 0: ' + str(bn[0]))
        nn = keras.layers.BatchNormalization()(nn)
    if cn[0]:
        print('customnorm 0: ' + str(cn[0]))
        nn = CustomNorm()(nn)
    nn = keras.layers.Activation(activation)(nn)
    if dropout[0]>0:
        print('dropout 0: ' + str(dropout[0]))
        nn = keras.layers.Dropout(dropout[0])(nn)
    

    nn = keras.layers.Conv1D(filters=units[1],
                            kernel_size=7,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[1],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(nn)
    if bn[1]:
        print('batchnorm 1: ' + str(bn[1]))
        nn = keras.layers.BatchNormalization()(nn)
    if cn[1]:
        print('customnorm 1: ' + str(cn[1]))
        nn = CustomNorm()(nn)
    nn = keras.layers.Activation('relu')(nn)
    if dropout[1]>0:
        print('dropout 1: ' + str(dropout[1]))
        nn = keras.layers.Dropout(dropout[1])(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=units[2],
                            kernel_size=5,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[2],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(nn)
    if bn[2]:
        print('batchnorm 2: ' + str(bn[2]))
        nn = keras.layers.BatchNormalization()(nn)
    if cn[2]:
        print('customnorm 2: ' + str(cn[2]))
        nn = CustomNorm()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    if dropout[2]>0:
        print('dropout 2: ' + str(dropout[2]))
        nn = keras.layers.Dropout(dropout[2])(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=units[3],
                            kernel_size=5,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[3],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(nn)
    if bn[3]:
        print('batchnorm 3: ' + str(bn[3]))
        nn = keras.layers.BatchNormalization()(nn)
    if cn[3]:
        print('customnorm 3: ' + str(cn[3]))
        nn = CustomNorm()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    if dropout[3]>0:
        print('dropout 3: ' + str(dropout[3]))
        nn = keras.layers.Dropout(dropout[3])(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[4],
                            activation=None,
                            use_bias=use_bias[4],
                            kernel_regularizer=l2, 
                            )(nn)      
    if bn[4]:
        print('batchnorm 4: ' + str(bn[4]))
        nn = keras.layers.BatchNormalization()(nn)
    if cn[4]:
        print('customnorm 4: ' + str(cn[4]))
        nn = CustomNorm()(nn)
    nn = keras.layers.Activation('relu')(nn)
    if dropout[4]>0:
        print('dropout 4: ' + str(dropout[4]))
        nn = keras.layers.Dropout(dropout[4])(nn)

    # Output layer 
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    return keras.Model(inputs=inputs, outputs=outputs)


def shallownet_25(input_shape, num_labels, activation='relu', 
            units=[24, 48, 96], dropout=[0.1, 0.2, 0.5], 
            bn=[True, True, True], l2=None):
    if l2 is not None:
        l2 = keras.regularizers.l2(l2)
    
    use_bias = []
    for status in bn:
        if status:
            use_bias.append(True)
        else:
            use_bias.append(False)

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=units[0],
                            kernel_size=19,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[0],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(inputs)
    if bn[0]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)
    nn = keras.layers.MaxPool1D(pool_size=25)(nn)
    nn = keras.layers.Conv1D(filters=units[1],
                            kernel_size=7,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[1],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(nn)
    if bn[1]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout[1])(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)


    # layer 2 - Fully-connected 
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[2],
                            activation=None,
                            use_bias=use_bias[2],
                            kernel_regularizer=l2, 
                            )(nn)      
    if bn[2]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout[2])(nn)

    # Output layer 
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    return keras.Model(inputs=inputs, outputs=outputs)

def shallownet_4(input_shape, num_labels, activation='relu', 
            units=[24, 48, 96], dropout=[0.1, 0.2, 0.5], 
            bn=[True, True, True], l2=None):
    if l2 is not None:
        l2 = keras.regularizers.l2(l2)
    
    use_bias = []
    for status in bn:
        if status:
            use_bias.append(True)
        else:
            use_bias.append(False)

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=units[0],
                            kernel_size=19,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[0],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(inputs)
    if bn[0]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Conv1D(filters=units[1],
                            kernel_size=7,
                            strides=1,
                            activation=None,
                            use_bias=use_bias[1],
                            padding='same',
                            kernel_regularizer=l2, 
                            )(nn)
    if bn[1]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout[1])(nn)
    nn = keras.layers.MaxPool1D(pool_size=25)(nn)


    # layer 2 - Fully-connected 
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[2],
                            activation=None,
                            use_bias=use_bias[2],
                            kernel_regularizer=l2, 
                            )(nn)      
    if bn[2]:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(dropout[2])(nn)

    # Output layer 
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    return keras.Model(inputs=inputs, outputs=outputs)
