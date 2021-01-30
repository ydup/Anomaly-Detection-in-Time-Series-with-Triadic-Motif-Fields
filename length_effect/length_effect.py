"""
Get the performance (time and probability) of model with different length of ECG signals
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo: 
$ mpirun -n 5 python3 parallel_flex.py --slice 0 --nodes 10
"""
import os
import sys
sys.path.append('../')
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import pickle as dill
import argparse
import time
from lib.util import TMF_image as gen_TMF
from lib.util import mkdir
from lib.util import build_fullnet as build

net = build()

parser = argparse.ArgumentParser()

# Get the settings from command line
parser.add_argument('--slice', type=int, default=0, help='slice index of the dataset')
parser.add_argument('--nodes', type=int, default=8, help='total slice of the dataset')
args = parser.parse_args()

mode = 'test'
slidx = int(args.slice)
slice_num = int(args.nodes)  # node num

comm = MPI.COMM_WORLD
mpisize = int(comm.Get_size())  # total num of the cpu cores, the n_splits of the k-Fold
mpirank = int(comm.Get_rank())  # rank of this core

if mpirank == 0:
    with open('../data/ECG_X_{0}.bin'.format(mode), 'rb') as fin:
        X = np.load(fin)
    total_num = len(X)//slice_num
    if slidx != slice_num-1:
        X = X[slidx*total_num: (slidx+1)*total_num]
    else:
        X = X[slidx*total_num: ]
    total_num = len(X)
    total_idx = np.arange(total_num)
    chunk_len = total_num//(mpisize-1) if total_num >= mpisize - 1 else total_num
    data = [X[total_idx[i*chunk_len: (i+1)*chunk_len]] for i in tqdm(range(mpisize), desc='Scatter data')]
else:
    data = None
    
data = comm.scatter(data, root=0)

# TMF
data_path = 'slice/{0}/'.format(slidx)
mkdir(data_path)

D = 3
output = []
time_consumption = []

for ts in tqdm(data):
    onesample = []
    onetime = []
    for win in range(100, 3100, 100):
        time_start=time.time()  # start time
        shape = np.array([len(range(1, (win-1)//(D-1) + 1)), len(range(0, win-(D-1)*1)), D])
        overlap = win-(D-1)*shape[0]
        TMF = np.zeros(shape)
        TMF = gen_TMF(ts[0: win], overlap, TMF, D)
        img = np.expand_dims(TMF, axis=0)  # [1, W, H, 3]
        prob = net.predict(img)  # [1, 2]
        time_end=time.time()  # end time
        onesample.append(prob)
        onetime.append(time_end - time_start)
    if len(onetime) != 0 and len(onesample) != 0:
        onesample = np.concatenate(onesample, axis=0)  # [S, 2]
        onetime = np.array(onetime)  # [S]
        output.append(onesample)
        time_consumption.append(onetime)

output = np.stack(output, axis=0) if len(output) != 0 else np.array([]) # [N, S, 2]
time_consumption = np.stack(time_consumption, axis=0) if len(time_consumption) != 0 else np.array([]) # [N, S]

np.save(data_path+'prob_'+str(mpirank), output)
np.save(data_path+'time_'+str(mpirank), time_consumption)


