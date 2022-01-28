#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:01:47 2022

@author: adelino
"""
import os

# ------------------------------------------------------------------------------
def list_contend(folder='./', pattern=()):
    list_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(folder)
                 for name in files
                     if name.endswith(pattern)]
    return list_files

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------