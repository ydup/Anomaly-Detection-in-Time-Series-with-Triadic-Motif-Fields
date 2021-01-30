"""
Random forest classifier with grid search the best parameters according to the performance on the validation dataset
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python3 classifier_RF.py vgg16
"""
import sys
sys.path.append('../')
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
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

# Load data
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

n_estimators = list([500, 1000, 1500])
max_features = list([32, 64, 128])
min_samples_split = [512, 1024]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap}

rf = RandomForestClassifier(random_state=43, n_jobs=20)

cv = [(train_indices, val_indices)]

search = RandomizedSearchCV(
    rf,
    param_distributions=random_grid,
    cv=cv,
    n_iter=10,
    verbose=10,
    n_jobs=20)

search.fit(X, y)
print(search.best_params_)
pred=search.predict_proba(test_x)
print(feature, eval(test_y, pred[:, 1]))
