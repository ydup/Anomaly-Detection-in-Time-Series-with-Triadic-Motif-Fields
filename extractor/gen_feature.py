"""
Extract the features of TMF image
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo: map the dataset to 10 nodes and generate the first slice using 5 process
$ mpirun -n 5 python3 gen_feature.py --mode train --freq no --slice 0 --nodes 10
"""
import os
import sys
sys.path.append('../')
from tqdm import tqdm
import numpy as np
import pandas as pd
from mpi4py import MPI
from collections import OrderedDict, Counter
import pickle as dill
import argparse
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from scipy.signal import butter, lfilter
from lib.util import TMF_image as gen_TMF
from lib.util import mkdir
from lib.visual import filt_ECG

def build(model_type):
    # create the base pre-trained model
    if model_type == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif model_type == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False)
    elif model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    else:
        raise(Exception('model_type must be one of vgg16, vgg19 and resnet50.'))
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x)
    return model

parser = argparse.ArgumentParser()

# Get the settings from command line
parser.add_argument('--mode', type=str, help='data type: train, val or test')
parser.add_argument('--freq', type=str, default='mid', help='filter type: low, mid, high, no')
parser.add_argument('--slice', type=int, default=0, help='slice index of the dataset')
parser.add_argument('--nodes', type=int, default=8, help='total slice of the dataset')
args = parser.parse_args()

mode = str(args.mode)  # train, val or test
freq = str(args.freq)  # filter type
slidx = int(args.slice)  # index of node
slice_num = int(args.nodes)  # total nodes
model_type = 'vgg16'  # model for feature extraction

extractor = build(model_type)

comm = MPI.COMM_WORLD
mpisize = int(comm.Get_size())  # total num of the cpu cores, the n_splits of the k-Fold
mpirank = int(comm.Get_rank())  # rank of this core

print(mpisize, mpirank)

if mpirank == 0:
    # Slice and broadcast the data
    with open('../data/ECG_X_{0}.bin'.format(mode), 'rb') as fin:
        X = np.load(fin)
    total_num = len(X)//slice_num  # map to nodes 
    if slidx != slice_num-1:
        X = X[slidx*total_num: (slidx+1)*total_num]
    else:
        X = X[slidx*total_num: ]
    total_num = len(X)
    total_idx = np.arange(total_num)
    chunk_len = total_num//(mpisize-1) if total_num >= mpisize - 1 else total_num  # map to process
    data = [X[total_idx[i*chunk_len: (i+1)*chunk_len]] for i in tqdm(range(mpisize), desc='Scatter data')]  # scatter the data to other process
else:
    data = None
data = comm.scatter(data, root=0)

# TMF
data_path = '../feature-{3}/{0}/{1}/{2}/'.format(*[mode, freq, slidx, model_type])
mkdir(data_path)

all_feature = []

win = 3000  # length of the time series
D = 3  # order of the motif, 3 means triad
shape = np.array([len(range(1, (win-1)//(D-1) + 1)), len(range(0, win-(D-1)*1)), D])  # TMF image shape
overlap = win-(D-1)*shape[0]  # overlap cause by the rotation in the TMF image
TMF = np.zeros(shape)  # placeholder of TMF image

for ts in tqdm(data):
    filt_ts = filt_ECG(ts, freq) 
    TMF = gen_TMF(filt_ts, overlap, TMF, D)  # the gen_TMF function is optimized with numba package
    img = np.expand_dims(TMF, axis=0)  # [1, W, H, 3]
    feature = extractor.predict(img)  # [1, 512]
    all_feature.append(feature)

all_feature = np.concatenate(all_feature, axis=0) if len(all_feature) != 0 else np.array([])
np.save(data_path+str(mpirank), all_feature)

