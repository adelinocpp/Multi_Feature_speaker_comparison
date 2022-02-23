#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 22:17:10 2022

@author: adelino
"""
import numpy as np
# -----------------------------------------------------------------------------
def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
# -----------------------------------------------------------------------------
def teager_energy_operator(x):
    xl = shift(x, 1, 0)
    xp = shift(x, -1, 0)
    return np.sum(np.abs(np.power(x,2) - np.multiply(xl,xp)))/len(x)
# -----------------------------------------------------------------------------