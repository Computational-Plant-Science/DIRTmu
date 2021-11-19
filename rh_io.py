# -*- coding: utf-8 -*-
"""
Input/output
"""


#from PIL import Image
#import sys
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    image = plt.imread(path)
    return np.array(image)

def save_table(table,path_name):
    table.to_csv(path_name)