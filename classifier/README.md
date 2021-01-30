# Classifiers

1. MLP: early stopping
2. RF and LR: random search

## Default

Three classifiers use the feature extracted from VGG16 which is based on the original time series. ```mode = 'no'``` means no filter was used upon the ECG signal. 
```feature = 'vgg16'``` means it use the VGG16 features.

## Demo

1. Run directly in command line
```shell
$ python3 classifier_MLP.py vgg16
```
2. Submit a pbs job
```shell
$ qsub -v classifier=MLP,feature=vgg16 classifier.sh
```

