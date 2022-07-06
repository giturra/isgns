from math import floor, ceil
import numpy as np


def round_number(num):
    #print(num)
    c = ceil(num)
    f = floor(num)
    uni = int(np.random.uniform(0.0, 1.0))
    # print(f'uni = {uni}')
    # print(f'num - f = {num - f}')
    if uni < (num - f):
        return c
    else:
        return f