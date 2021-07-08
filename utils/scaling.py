import numpy as np


def minmax(invalues, bound):
    '''Scale the input to [-1, 1]'''
    out =  2 * (invalues - bound) / (2*bound) + 1
    return out