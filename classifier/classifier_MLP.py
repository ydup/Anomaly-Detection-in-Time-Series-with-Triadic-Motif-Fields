"""
MLP classifier of the extracted features with early stopping strategy on the validation dataset.
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python3 classifier_MLP.py vgg16
"""
import sys
sys.path.append('../')
import numpy as np
import pickle
from tensorflow import keras
from lib.util import eval

# Param
data_path = '../'
nb_classes = 2
batch_size = 16
nb_epochs = 10
mode = 'no'
feature = str(sys.argv[1]) if len(sys.argv) > 1 else 'vgg16'

# Load data
with open(data_path+'data/ECG_info.pkl', 'rb') as f:
    label = pickle.load(f)

with open(data_path+'feature-{1}/train/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    train_x = np.load(f)
    train_y = keras.utils.to_categorical(label['Y_train'], num_classes=nb_classes)

with open(data_path+'feature-{1}/val/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    val_x = np.load(f)
    val_y = keras.utils.to_categorical(label['Y_val'], num_classes=nb_classes)

with open(data_path+'feature-{1}/test/{0}/{0}.npy'.format(*[mode, feature]), 'rb') as f:
    test_x = np.load(f)
    test_y = keras.utils.to_categorical(label['Y_test'], num_classes=nb_classes)

dim = train_x.shape[1]

def classifier(nb_classes):
    # classifier
    x = keras.layers.Input(shape=(dim))
    dnn = keras.layers.Dense(128, activation='relu')(x) 
    predictions = keras.layers.Dense(nb_classes, activation='softmax')(dnn)
    model = keras.models.Model(inputs=x, outputs=predictions)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def run(idx):
    net = classifier(nb_classes)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001) 
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='../model/weights_tail_{0}.hdf5'.format(idx), verbose=1, save_best_only=True)

    # Train model on dataset
    hist = net.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(val_x, val_y), callbacks = [reduce_lr, checkpointer], shuffle=True)

    net.load_weights('../model/weights_tail_{0}.hdf5'.format(idx))

    test_pred = net.predict(test_x)
    res = eval(test_y[:, 1], test_pred[:, 1])
    print('ROC_AUC:{0}, PR_AUC:{1}, F1:{2}'.format(*res))

    return [idx] + res


run(feature)





