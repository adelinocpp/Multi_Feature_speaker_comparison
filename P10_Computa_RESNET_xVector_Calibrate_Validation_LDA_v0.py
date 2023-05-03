#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:39:21 2022

@author: adelino
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:44:05 2022

@author: adelino
"""
import torch

import pandas as pd
import math
import os
import config as c
from models.tddnn_utilities import read_feats_structure, get_min_loss_model, read_MFB
import numpy as np
import dill
from imports.files_utils import list_contend, build_folders_to_save
from models.model import background_resnet
import gc
from glob import glob
# ----------------------------------------------------------------------------
class ToTensorTestInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            np_feature = np.expand_dims(np_feature, axis=0)
            np_feature = np.expand_dims(np_feature, axis=1)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() # output type => torch.FloatTensor, fast
            # input size : (1, 1, n_win=200, dim=40)
            # output size : (1, 1, dim=40, n_win=200)
            return ten_feature
# ----------------------------------------------------------------------------        
def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes,TypeBackBone):
    
#    model = X_vector(embedding_size, n_classes)
    model = background_resnet(embedding_size=embedding_size, \
                              num_classes=n_classes,backbone=TypeBackBone)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    fnePathern = 'checkpoint_{:06}_*'.format(cp_num)
    
    file = glob(os.path.join(log_dir, fnePathern))
    
    checkpoint = torch.load(file[0])
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
# ----------------------------------------------------------------------------
def get_num_class(folderName, pattern = ('.p',)):
    file_list = list_contend(folderName,pattern)
    label = np.array([]);
    for idx, filename in enumerate(file_list):
        filenameParts = filename.replace('\\', '/')
        filenameFolder = int(filenameParts.split('/')[-2]) # Quatro primeiros digitos do nome do diretorio 
        label = np.append(label,filenameFolder)
    return len(np.unique(label))
# ----------------------------------------------------------------------------
def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB
# ----------------------------------------------------------------------------
def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    
    tot_segments = math.floor(len(input)/test_frames) # total number of segments with 'test_frames' 
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]
            # print("temp_input: input shape: {:} totseg: {:} len: {:} shape: {:}".format(input.shape,tot_segments, len(temp_input.shape),temp_input.shape))
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    activation = l2_norm(activation, c.ALPHA_LNORM)
    return activation
# ----------------------------------------------------------------------------
def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output
# ----------------------------------------------------------------------------
def enroll_train_spk(use_cuda, test_frames, model, file_list, embedding_dir):
    """
    Output the averaged d-vector for each speaker (enrollment)
    Return the dictionary (length of n_spk)
    """
    n_files = len(file_list) # 10
    # enroll_speaker_list = sorted(set(DB['speaker_id']))
    file_list.sort()
    numEmbeddings = {}
    
    # Aggregates all the activations
    print("Start to aggregate all the d-vectors per enroll speaker")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        
    for i in range(n_files):
        filename = file_list[i]
        file_data = filename.split('/')
        # filename = DB['filename'][i]
        spk_label = file_data[-2]
        # spk_num = int(spk_label)
        out_file_name = file_data[-1].split(".")[0]
        activation = get_embeddings(use_cuda, filename, model, test_frames)
        activation = np.squeeze(activation.numpy())
        
        embedding_path = os.path.join(embedding_dir, \
                    '{:}/{:}'.format(spk_label,out_file_name)+'.pth')    
            
        build_folders_to_save(embedding_path)
        
        ofile = open(embedding_path, "wb")
        dill.dump(activation, ofile)
        ofile.close() 
   
        print("Save the embeddings for (index, speaker, #files) {:04d} {:} {:}"\
              .format(i, spk_label, n_files))
            
    return 0
# ----------------------------------------------------------------------------
use_cuda = False
featureDim = 3*c.NUM_CEPS_COEFS
log_dir = c.RESNET_SAVE_MODELS_DIR


use_cuda = False # use gpu or cpu
if (c.RESNET_BACKBONETYPE == 'resnet18'):
    embedding_size = 32
elif(c.RESNET_BACKBONETYPE == 'resnet34'):
    embedding_size = 64
elif(c.RESNET_BACKBONETYPE == 'resnet50'):
    embedding_size = 128
elif(c.RESNET_BACKBONETYPE == 'resnet101'):
    embedding_size = 256
elif(c.RESNET_BACKBONETYPE == 'resnet152'):
    embedding_size = 512

cp_num,__ = get_min_loss_model(log_dir) # Which checkpoint to use?
n_classes = get_num_class(c.TDDNN_TRAIN_PATH)
test_frames = c.RESNET_NUM_WIN_SIZE
model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes, c.RESNET_BACKBONETYPE)

COMPUTE_XVECTOR_FOR_CALIBRATION = True
COMPUTE_XVECTOR_FOR_VALIDATION = True
COMPUTE_XVECTOR_FOR_LDA = True
COMPUTE_XVECTOR_FOR_UBM = False

# Build CALIBRATION X-VECTORS
if (COMPUTE_XVECTOR_FOR_CALIBRATION):
    print("Iniciando x-vectos para calibracao")
    file_list = list_contend(c.CALIBRATION_DIR, pattern = ('.p',))
    enroll_train_spk(use_cuda, test_frames, model, file_list, c.XVECTOR_RESNET_MODEL_CALIBRATION_DIR)

# Build VALIDATION X-VECTORS
if (COMPUTE_XVECTOR_FOR_VALIDATION):
    print("Iniciando x-vectos para validacao")
    file_list = list_contend(c.VALIDATION_DIR, pattern = ('.p',))
    enroll_train_spk(use_cuda, test_frames, model, file_list, c.XVECTOR_RESNET_MODEL_VALIDATION_DIR)

# Build VALIDATION X-VECTORS
if (COMPUTE_XVECTOR_FOR_LDA):
    print("Iniciando x-vectos para LDA")
    file_list = list_contend(c.LDA_DIR, pattern = ('.p',))
    enroll_train_spk(use_cuda, test_frames, model, file_list, c.XVECTOR_RESNET_MODEL_LDA_DIR)

if (COMPUTE_XVECTOR_FOR_UBM):
    print("Iniciando x-vectos para UBM")
    file_list = list_contend(c.UBM_DIR, pattern = ('.p',))
    enroll_train_spk(use_cuda, test_frames, model, file_list, c.XVECTOR_RESNET_MODEL_UBM_DIR)
print('Calculados...')
   

