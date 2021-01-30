"""
Collect the features
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo: collect the features of training dataset
$ python reduce_gen_feature.py train 10 no
"""
import numpy as np
import argparse
import sys

mode = str(sys.argv[1])
slide_num = int(sys.argv[2])  # slice num must be same as the nodes paramters of gen_feature.py
mpi_size = 5  # must be same as the mpirun -n in .sh script
freq = str(sys.argv[3])  # no filter
path = '../feature-vgg16/{0}/{1}/'.format(*[mode, freq])
feature = []
for i in range(slide_num):
    for j in range(mpi_size):
        with open(path+'{0}/{1}.npy'.format(*[i, j]), "rb") as f:
            tmp = np.load(f)
            # print(tmp.shape)
            if len(tmp):
                feature.append(tmp)
                
feature = np.concatenate(feature, axis=0)
np.save(path+freq, feature)
print(feature.shape)



