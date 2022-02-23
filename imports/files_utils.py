#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:01:47 2022

@author: adelino
"""
import os

# -----------------------------------------------------------------------------
def build_folders_to_save(file_name):
    split_path = file_name.split('/')
    for idx in range(1,len(split_path)):
        curDir = '/'.join(split_path[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
# ------------------------------------------------------------------------------
def list_contend(folder='./', pattern=()):
    list_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(folder)
                 for name in files
                     if name.endswith(pattern)]
    return list_files
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------