# Feature extractors

## Default

Features are generated using VGG16 (excluding its top three layers) from the TMF images of original ECG signals. ```model_type = 'vgg16', mode='no'```. MPI size is set as 5 in ```.sh``` and ```reduce*.py``` files.

## Demo

1. Submit the parallel jobs: training set, 10 nodes, and no filter
```
$ python map_gen_feature.py train 10 no
```
2. Collect all the results from the finished jobs
```
$ python reduce_gen_feature.py train 10 no
```