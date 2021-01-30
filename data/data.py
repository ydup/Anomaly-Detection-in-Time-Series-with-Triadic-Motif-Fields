"""
Create dataset
steps:
1. use the challenge2017.pkl from https://github.com/hsd1503/MINA
2. run:
$ python3 data.py
"""
import sys
sys.path.append('../')
from lib.util import make_data_physionet

make_data_physionet('./')
