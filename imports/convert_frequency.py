#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:28:38 2022

@author: adelino
"""
import numpy as np

# -----------------------------------------------------------------------------
def bark2hertz(bark_freq):
    return 600 * np.sinh(bark_freq/6)
# -----------------------------------------------------------------------------
def hertz2bark(hertz_freq):
    return 6*np.arcsinh(hertz_freq/600)
# -----------------------------------------------------------------------------
def hertz2mel(hertz_freq):
    k=1000/np.log(1+1000/700) # 1127.01048
    af=np.abs(hertz_freq);
    return np.sign(hertz_freq)*np.log(1+af/700)*k
    