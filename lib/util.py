import os
import sys
import random
import pickle as dill

import cv2
import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, Counter

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow import keras

@nb.jit(nopython=True)
def TMF_image(ts, overlap, TMF, D=3):
    '''
    generate triadic motif field image of time series
    :param ts: 1-d array, time series
    :param shape: int, shape of TMF, np.array([len(range(1, (win-1)//(D-1) + 1)), len(range(0, win-(D-1)*1)), D])  # TMF image shape
    :param overlap: int, overlap cause by the rotation in the TMF image, overlap = win-(D-1)*shape[0] 
    :param TMF: 2-d array, np.zeros(shape)  # placeholder of TMF image
    :param D: int, number of ordinal points, default triad
    return image, [W, H, D]
    '''
    shape = TMF.shape
    for i in range(shape[0]):
        right_bound = len(ts)-(D-1)*(i+1)
        for j in range(right_bound):
            motif_idx = np.arange(j, j+D*(i+1), (i+1))
            if j < right_bound - overlap:
                TMF[i, j, :] = ts[motif_idx]
                TMF[shape[0]-i-1, shape[1]-j-1, :] = ts[motif_idx]
            else:
                TMF[i, j, :] = ts[motif_idx]
    return TMF

def get_GCAM(model, inputs, targets, layers=-11):
    '''
    Grad-CAM
    :param model: keras model
    :param input: 4-d array, shape is [1, W, H, C]
    :param target: 1-d array, shape is [nb_class] (one-hot)
    :param layers: int value, visualize which layer
    return: 2-d array, shape is [W, H]
    '''
    class_idx = np.argmax(targets, axis=-1)
    class_output = model.output[:, class_idx]
    last_conv_layer = model.layers[layers]
    class_output = model.output[:, class_idx]

    x = inputs.copy()
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function(model.input, [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(x)
    for i, grads_value in enumerate(pooled_grads_value):
        conv_layer_output_value[:, :, i] *= grads_value
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap[heatmap<0] = 0  # ReLU
    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
    gcam = (heatmap-heatmap.min())/(heatmap.max() - heatmap.min())
    return gcam

@nb.jit(nopython=True)
def get_sym_GCAM(overlap, gcam, D=3):
    '''
    normalize Grad-CAM into symmetrized Grad-CAM
    :param shape: int, shape of TMF, np.array([len(range(1, (win-1)//(D-1) + 1)), len(range(0, win-(D-1)*1)), D])  # TMF image shape
    :param overlap: int, overlap cause by the rotation in the TMF image, overlap = win-(D-1)*shape[0] 
    :param gcam: 2-d array, np.zeros(shape)  # placeholder of gcam
    :param D: int, number of ordinal points, default triad
    return image, [W, H]
    '''
    shape = gcam.shape
    for i in range(shape[0]):
        right_bound = shape[1]+(D-1)-(D-1)*(i+1)
        for j in range(right_bound):
            if j < right_bound - overlap:
                gcam[i, j] = (gcam[i, j] + gcam[shape[0]-i-1, shape[1]-j-1])/2.0
                gcam[shape[0]-i-1, shape[1]-j-1] = gcam[i, j]
                
            else:
                gcam[i, j] = gcam[i, j]
    gcam = (gcam - gcam.min())/(gcam.max()-gcam.min())
    return gcam

def get_cam_image(net, ts, return_proba=False):
    '''
    generate the SG-CAM of AF and non-AF
    :param net: keras model
    :param ts: 1-d array, time series
    :param return_proba: boolean
    return SG-CAM of non-AF, SG-CAM of AF, predicted probability (if return_proba=True)
    '''
    D = 3
    shape = np.array([len(range(1, (len(ts)-1)//(D-1) + 1)), len(range(0, len(ts)-(D-1)*1)), D])
    overlap = len(ts)-(D-1)*shape[0]
    # TMF image: [1, W, H, 3]
    img = np.zeros(shape)
    img = TMF_image(ts, overlap, img, D)
    img = np.expand_dims(img, axis=0)
    # SG-CAM image of non-AF
    gcam = get_GCAM(net, img, [1,0], layers=-3)
    gcam_norm = get_sym_GCAM(overlap, gcam, D)
    nAF_cam = gcam_norm.copy()
    # SG-CAM image of AF
    gcam = get_GCAM(net, img, [0,1], layers=-3)
    gcam_norm = get_sym_GCAM(overlap, gcam, D)
    AF_cam = gcam_norm.copy()
    if not return_proba:
        return nAF_cam, AF_cam
    else:
        proba = net.predict(img)
        return nAF_cam, AF_cam, proba

def build_fullnet(path='../model/weights_tail_best.hdf5'):
    '''
    build the full network (VGG16-MLP)
    :param path: str, path of the trained MLP
    return: keras network
    '''
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    vec = keras.layers.GlobalAveragePooling2D()(x)
    
    row_input = keras.layers.Input((vec.shape[1:]))
    dnn = keras.layers.Dense(128, activation='relu')(row_input)

    predictions = keras.layers.Dense(2, activation='softmax')(dnn)

    tail_model = keras.models.Model(inputs=row_input, outputs=predictions)
    if path is not None:
        tail_model.load_weights(path)

    out = tail_model(vec)
    network = keras.models.Model(inputs=base_model.input, outputs=out)
    return network

def eval(target, predict):
    '''
    evaluate the results
    :param target: 1-d array, 0 or 1
    :param predict: 1-d array, probability of AF
    return roc_auc, pr_auc and F1
    '''
    ROC_AUC = metrics.roc_auc_score(target, predict)
    # PR_AUC
    precision, recall, _thresholds = metrics.precision_recall_curve(
        target, predict)
    PR_AUC = metrics.auc(recall, precision)
    # F1
    predict = np.array([i > 0.5 for i in predict], dtype=int)
    F1 = metrics.f1_score(target, predict)
    return [ROC_AUC, PR_AUC, F1]

def eval_patient(pid, label, proba):
    '''
    evaluate the patient-wise accuracy
    :param pid: 1-d array, shape is [N,], id of patients
    :param label: 1-d array, shape is [N,], 0 or 1, 1 indicates the AF segment
    :param proba: 1-d array, shape is [N,], probability of AF
    return: 2-d array, [N, 3], columns are pid, count of segment, patient-wise accuracy
    '''
    wrong = pid[(label == 1) & (proba < 0.5)]
    wrong_pid = np.unique(wrong)
    AF = pid[(label == 1)]
    AF_pid = np.unique(AF)
    detail = []
    for i in AF_pid:
        # [id, length, accuracy]
        total = float(np.sum(AF == i))
        detail.append([i, total, (total - np.sum(wrong == i)) / total])
    detail.sort(key=lambda x: x[2])  # sort according to the accuracy
    return np.array(detail)

def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path+' is created!')
        return True
    else:
        print(path+' already exists!')
        return False

def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    '''
    From https://github.com/hsd1503/MINA
    MINA: Multilevel Knowledge-Guided Attention for Modeling Electrocardiography Signals, IJCAI 2019
    '''
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            i_stride = stride//10
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

def make_data_physionet(data_path, n_split=50, window_size=3000, stride=500):
    '''
    From https://github.com/hsd1503/MINA
    MINA: Multilevel Knowledge-Guided Attention for Modeling Electrocardiography Signals, IJCAI 2019
    '''
    # read pkl
    with open(os.path.join(data_path, 'challenge2017.pkl'), 'rb') as fin:
        res = dill.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std  # normalize
    all_data = res['data']
    all_data = np.array(all_data)
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'A':
            all_label.append(1)
        else:
            all_label.append(0)
    all_label = np.array(all_label)

    # split train test
    n_sample = len(all_label)
    split_idx_1 = int(0.75 * n_sample)
    split_idx_2 = int(0.85 * n_sample)

    shuffle_idx = np.random.RandomState(seed=40).permutation(n_sample)
    all_data = all_data[shuffle_idx]
    all_label = all_label[shuffle_idx]

    X_train = all_data[:split_idx_1]
    X_val = all_data[split_idx_1:split_idx_2]
    X_test = all_data[split_idx_2:]
    Y_train = all_label[:split_idx_1]
    Y_val = all_label[split_idx_1:split_idx_2]
    Y_test = all_label[split_idx_2:]

    # slide and cut
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(
        X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val = slide_and_cut(
        X_val, Y_val, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(
        X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.RandomState(seed=42).permutation(
        Y_train.shape[0])  # np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    # save
    res = {'Y_train': Y_train, 'Y_val': Y_val,
           'Y_test': Y_test, 'pid_test': pid_test}
    with open(os.path.join(data_path, 'ECG_info.pkl'), 'wb') as fout:
        dill.dump(res, fout)

    fout = open(os.path.join(data_path, 'ECG_X_train.bin'), 'wb')
    np.save(fout, X_train)
    fout.close()

    fout = open(os.path.join(data_path, 'ECG_X_val.bin'), 'wb')
    np.save(fout, X_val)
    fout.close()

    fout = open(os.path.join(data_path, 'ECG_X_test.bin'), 'wb')
    np.save(fout, X_test)
    fout.close()
