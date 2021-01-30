# Length effect

## Default

Length effect based on trained VGG-MLP model is evaluated on the test dataset. Length are selected from 100 to 3000 with 100 as the gap.

## Demo

1. Submit the parallel jobs: 10 nodes
```
$ python map_length_effect.py 10
```
2. Collect all the results from the finished jobs
```
$ python reduce_length_effect.py prob 10  # collect probabilities
$ python reduce_length_effect.py time 10  # collect time consumptions
```