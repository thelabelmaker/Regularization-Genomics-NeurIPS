import numpy as np
import h5py
import os

def load_data():
    """# Load Synthetic regulatory code data (Task 3 in Koo & Ploenzke, NMI, 2021)"""
    filepath = './data/synthetic_code_dataset.h5'
    with h5py.File(filepath, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32)
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32)
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)
        model_test = np.array(dataset['model_test']).astype(np.float32)

    model_test = model_test.transpose([0,2,1])
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])
    N, L, A = x_train.shape
    return x_train, x_valid, x_test, y_train, y_valid, y_test, model_test

if __name__ == "__main__":
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()