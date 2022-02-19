#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:18:07 2022

@author: adelino
"""
from pathlib import Path
import subprocess

class AudioData:
    def __init__(self, audioFullPath):
        self.audioFullPath = audioFullPath
        self.suffix = audioFullPath.split('/')[-1].split('.')[-1].upper()
        cmdString = 'mediainfo --Inform=\"Audio;%Format%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.codec =  returnStr.read().decode()[0:-1]
        cmdString = 'mediainfo --Inform=\"Audio;%Channels%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.channels =  returnStr.read().decode()[0:-1]
        cmdString = 'mediainfo --Inform=\"Audio;%SamplingRate%\" ' + audioFullPath
        returnStr =  subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        self.sr =  returnStr.read().decode()[0:-1]
        
    def check(self,ext,codec,channels,sr):
        return (self.sr == sr) and (self.codec == codec) and \
            (self.channels == channels) and (self.suffix == ext)

    def suit(self,ext,channels,sr):
        dir_path = Path(self.audioFullPath).parent
        basename = Path(self.audioFullPath).stem
        ext = Path(self.audioFullPath).suffix
        # --- rename file for temp.wav
        temp_file_name = dir_path.as_posix() + '/temp' + ext
        cmdString = 'mv {:} {:}'.format(self.audioFullPath,temp_file_name)
        subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        new_file_name = dir_path.as_posix() + '/' +  basename + '.' + ext.lower()
        # --- Convert file
        cmdString = 'sox {:} -c {:} -r {:} -e signed-integer -b 16 {:}'.format(temp_file_name,channels,sr,new_file_name)
        subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        # --- remove temp file
        cmdString = 'rm {:}'.format(temp_file_name)
        subprocess.Popen(cmdString, shell=True, stdout=subprocess.PIPE).stdout
        return new_file_name