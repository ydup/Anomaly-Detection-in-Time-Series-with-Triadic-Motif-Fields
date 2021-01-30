"""
Collect the result of the length effect
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python reduce_length_effect.py prob 10
"""
import numpy as np
import argparse
import sys

mode = str(sys.argv[1])  # 'prob' or 'time'
slide_num = int(sys.argv[2])
mpi_size = 5

path = 'slice/'
feature = []
for i in range(slide_num):
    for j in range(mpi_size):
        with open(path+'{0}/{2}_{1}.npy'.format(*[i, j, mode]), "rb") as f:
            tmp = np.load(f)
            if len(tmp):
                feature.append(tmp)
                
feature = np.concatenate(feature, axis=0)
np.save(path+mode, feature)
print(feature.shape)



