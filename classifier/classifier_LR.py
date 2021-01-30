"""
Logistic regression with grid search the best parameters on the validation dataset
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python classifier_LR.py vgg16
"""
import os
import sys
sys.path.append('../')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
from tensorflow import keras
from sklearn import metrics
from lib.util import eval

# Param
data_path = '../'
mode = 'no'
nb_classes = 2
feature = str(sys.argv[1]) if len(sys.argv) > 1 else 'vgg16'

# Load dataset
with open(data_path+'data/ECG_info.pkl', 'rb') as f:
    label = pickle.load(f)

with open(data_path+'feature-{1}/train/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    train_x = np.load(f)
    train_y = label['Y_train']

with open(data_path+'feature-{1}/val/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    val_x = np.load(f)
    val_y = label['Y_val']

with open(data_path+'feature-{1}/test/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    test_x = np.load(f)
    test_y = label['Y_test']

X = np.concatenate([train_x, val_x], axis=0)
y = np.concatenate([train_y, val_y], axis=0)
train_indices = np.arange(train_x.shape[0])
val_indices = np.arange(train_x.shape[0], X.shape[0])

# Number of trees in random forest
random_grid = {'penalty' : ['l1', 'l2'],
    'C' : [0.4, 0.6, 0.8, 1],
    'solver' : ['liblinear', 'saga', 'lbfgs']}

rf = LogisticRegression(n_jobs=20, verbose=2)#random_state = 42)

cv = [(train_indices, val_indices)]

search = RandomizedSearchCV(
    rf,
    param_distributions=random_grid,
    cv=cv,
    n_iter=10,
    verbose=10,
    n_jobs=20
)

search.fit(X, y)
print(search.best_params_)
pred = rf.predict_proba(test_x)
print(feature, eval(test_y, pred[:, 1]))
