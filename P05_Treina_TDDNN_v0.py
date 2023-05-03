#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:47:36 2022

@author: adelino
"""
DEPURACAO = False
if (DEPURACAO):
    import sys

import sys
import config as c
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import torch.utils.data as data
import pandas as pd
import numpy as np

from imports.files_utils import list_contend
import os
import time
import dill
import random
from glob import glob
import warnings
from models.x_vector import X_vector
from models.tddnn_utilities import read_feats_structure, get_min_loss_model, read_MFB
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1" 
# ----------------------------------------------------------------------------
def load_training_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    # model = background_resnet(embedding_size=embedding_size, \
    #                           num_classes=n_classes,backbone=TypeBackBone)
    model =  X_vector(embedding_size, n_classes)
    optimizer =  optim.SGD(model.parameters(), lr=1E-1, momentum=0.9, \
                           dampening=0, weight_decay=1E-4)
    # model = TheModelClass(*args, **kwargs)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    fnePathern = 'checkpoint_{:06}_*'.format(cp_num)
    file = glob(os.path.join(log_dir, fnePathern))
    checkpoint = torch.load(file[0])
    # create new OrderedDict that does not contain `module.`
    
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer']);
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer, epoch
# ----------------------------------------------------------------------------
def collate_fn_feat_padded(batch):
    """
    Sort a data list by frame length (descending order)
    batch : list of tuple (feature, label). len(batch) = batch_size
        - feature : torch tensor of shape [1, 40, 80] ; variable size of frames
        - labels : torch tensor of shape (1)
    ex) samples = collate_fn([batch])
        batch = [dataset[i] for i in batch_indices]. ex) [Dvector_train_dataset[i] for i in [0,1,2,3,4]]
        batch[0][0].shape = torch.Size([1,64,774]). "774" is the number of frames per utterance. 
        
    """
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    feats, labels = zip(*batch)
    
    # Merge labels => torch.Size([batch_size,1])
    labels = torch.stack(labels, 0)
    labels = labels.view(-1)
    
    # Merge frames
    lengths = [feat.shape[2] for feat in feats] # in decreasing order 
    max_length = lengths[0]
    # features_mod.shape => torch.Size([batch_size, n_channel, dim, max(n_win)])
    padded_features = torch.zeros(len(feats), feats[0].shape[0], feats[0].shape[1], feats[0].shape[2]).float() # convert to FloatTensor (it should be!). torch.Size([batch, 1, feat_dim, max(n_win)])
    for i, feat in enumerate(feats):
        end = lengths[i]
        num_frames = feat.shape[2]
        while max_length > num_frames:
            feat = torch.cat((feat, feat[:,:,:end]), 2)
            num_frames = feat.shape[2]
        
        padded_features[i, :, :, :] = feat[:,:,:max_length]
    
    return padded_features, labels
# ----------------------------------------------------------------------------
class ToTensorInput(object):
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
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            
            # input size : (1, n_win=200, dim=40)
            # output size : (1, dim=40, n_win=200)
            return ten_feature
# ----------------------------------------------------------------------------
class ToTensorDevInput(object):
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
            assert np_feature.ndim == 3, 'Data is not a 3D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            # input size : (1, n_win=40, dim=40)
            # output size : (1, dim=40, n_win=40)
            return ten_feature
# ----------------------------------------------------------------------------
class TruncatedInputfromMFB(object):
    """
    input size : (n_frames, dim=40)
    output size : (1, n_win=40, dim=40) => one context window is chosen randomly
    """
    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features):
        network_inputs = []
        # num_frames = len(frames_features)
        num_frames = frames_features.shape[0]
        win_size = c.TDDNN_NUM_WIN_SIZE
        half_win_size = int(win_size/2)
        
        if (num_frames < win_size):
            print("TruncatedInputfromMFB error: num_frames < win_size")
            T1 = tuple()
            for i in range(0,np.ceil(win_size/num_frames)):
                T1 += frames_features
            frames_features = np.concatenate(T1)
            
        
        #if num_frames - half_win_size < half_win_size:
        while (num_frames - half_win_size) <= half_win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames,:], axis=0)
            num_frames =  len(frames_features)
            
        for i in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            if not j:
                frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)
# ----------------------------------------------------------------------------            
class DvectorDataset(data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None, *arg, **kw):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx
        self.feature_dim = 0;
    
    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)
        self.feature_dim = feature.shape[0]
        label = self.spk_to_idx[label]
        label = torch.Tensor([label]).long()
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label
    
    def __len__(self):
        return self.len
# ----------------------------------------------------------------------------            
def load_dataset(dbFolder, val_ratio):
    # Load training set and validation set
    # Split training set into training set and validation set according to "val_ratio"
    train_DB, valid_DB = split_train_dev(dbFolder, val_ratio) 
    file_loader = read_MFB # numpy array:(n_frames, n_dims)     
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
    
    speaker_list = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx)
    # train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=None, spk_to_idx=spk_to_idx)
    # valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=None, spk_to_idx=spk_to_idx)
    
    n_classes = len(speaker_list) # How many speakers? 240
    return train_dataset, valid_dataset, n_classes
# ----------------------------------------------------------------------------
def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB, num_speakers = read_feats_structure(train_feat_dir)
    total_len = len(train_valid_DB) # 148642
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    # --- Homogeniza o treinamento e validacao ---------------------------------
    index_spk = train_valid_DB["speaker_id"].unique()
    idx = train_valid_DB["speaker_count"]
    order = idx.argsort()
    new_order = np.concatenate((np.array(order[:len(index_spk)]),np.random.permutation((order[len(index_spk):]))))
    # order = np.array(order[:len(index_spk)]) np.random.permutation((order[len(index_spk):])
    shuffled_train_valid_DB = train_valid_DB.iloc[new_order,:]
    # --------------------------------------------------------------------------
    # shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    
    # Split the DB into train and valid set
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    # Reset the index
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len))
    
    return train_DB, valid_DB
# ----------------------------------------------------------------------------
def train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 2
    # switch to train mode
    model.train()
    
    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)
        targets = targets.view(-1) # target size:(batch size)
        current_sample = inputs.size(0)  # batch size
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        _, output = model(inputs) # out size:(batch size, #classes), for softmax
        
        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, inputs.size(0))
        
        loss = criterion(output, targets)
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:06.3f} ({batch_time.avg:06.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'Acc {train_acc.avg:08.4f}'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg
# ----------------------------------------------------------------------------                     
def validate(val_loader, model, criterion, use_cuda, epoch):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 4
    # switch to evaluate mode
    model.eval()

    print('Init Validade...')
    with torch.no_grad():
        # end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            current_sample = inputs.size(0)  # batch size
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # compute output
            _, output = model(inputs)
            
            # measure accuracy and record loss
            n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
            n_total += current_sample
            val_acc_temp = 100. * n_correct / n_total
            val_acc.update(val_acc_temp, inputs.size(0))
            
            loss = criterion(output, targets)
            losses.update(loss.item(), inputs.size(0))
            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            if i % log_interval == 0:
                print('Valid run: {:3d}/{:3d}\t'.format(i, len(val_loader)))
        print('  * Validation: '
                  'Loss {loss.avg:.4f}\t'
                  'Acc {val_acc.avg:.4f}'.format(
                  loss=losses, val_acc=val_acc))
    
    return losses.avg
# ----------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# ----------------------------------------------------------------------------
def create_optimizer(optimizer, model, new_lr, wd):
    # setup optimizer
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0,
                              weight_decay=wd)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    elif optimizer == 'ASGD':
        optimizer = optim.ASGD(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    elif optimizer == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(),
                                  lr=new_lr)
    return optimizer

# ----------------------------------------------------------------------------
def visualize_the_losses(train_loss, valid_loss):
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(train_loss)) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig('{:}loss_plot_tddnn.png'.format(c.TDDNN_SAVE_MODELS_DIR), bbox_inches='tight')

# ------------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    # Verifica a etapa de treinamento
    log_dir = c.TDDNN_SAVE_MODELS_DIR # where to save checkpoints
    # embedding_size = 512
    # --- dimensao 129 ---------------------------------------------------------
    featureDim = 3*c.NUM_CEPS_COEFS
    startEP, maxLoss = get_min_loss_model(log_dir)
    # Set hyperparameters
    # use_cuda = True # use gpu or cpu
    endEP = c.TDDNN_N_EPOCHS - startEP # Last epoch
    
    use_cuda = torch.cuda.is_available()
    # Load dataset
    train_dataset, valid_dataset, n_classes = load_dataset(c.TDDNN_TRAIN_PATH,c.TDDNN_TRAIN_TEST_RATIO)
    
    # print the experiment configuration
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    # instantiate model and initialize weights
    if (startEP == 0):
        # model = background_resnet(embedding_size=embedding_size, \
        #                       num_classes=n_classes, backbone=c.BACKBONETYPE)
        model = X_vector(featureDim, n_classes)
        optimizer = create_optimizer(c.TDDNN_OPT_TYPE, model, c.TDDNN_LR, c.TDDNN_WD)
    else:
        model, optimizer, startEP = load_training_model(use_cuda, log_dir, startEP, featureDim, \
                                    n_classes)
    if use_cuda:
        model.cuda()    
    
    
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, max_lr=1, base_lr=1e-4, verbose=1)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, max_lr=1e-3, base_lr=1e-5, mode="exp_range", gamma=0.8, step_size_up = 4,cycle_momentum = False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, min_lr=1e-4)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=c.TDDNN_BATCH_SIZE,
                                                       shuffle=c.TDDNN_USE_SHUFFLE,
                                                       pin_memory=False,
                                                       num_workers = 1) # timeout=240
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=c.TDDNN_VALID_BATCH,
                                                       shuffle=False,
                                                       pin_memory=False,
                                                       num_workers = 1,
                                                       collate_fn = collate_fn_feat_padded)
    
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # load file with data training (if exists)
    training_File_Data = '{:}{:}'.format(c.TDDNN_SAVE_MODELS_DIR,c.TDDNN_TRAIN_DATA_FILE)
    if (os.path.exists(training_File_Data)):
        ofile = open(training_File_Data, "rb")
        allLoss = dill.load(ofile)
        ofile.close()  
        if (startEP > 0):
            avg_train_losses = allLoss['train'][:startEP]
            avg_valid_losses = allLoss['valid'][:startEP]
    
    for epoch in range(startEP, c.TDDNN_N_EPOCHS):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes)
   
        # evaluate on validation set
        valid_loss = validate(valid_loader, model, criterion, use_cuda, epoch)
        
        if (not np.isfinite(train_loss)):
            # scheduler = optim.lr_scheduler.CyclicLR(optimizer, max_lr=1, base_lr=1e-5, mode="exp_range", gamma=0.8, step_size_up = 4, cycle_momentum = False)      
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, min_lr=1e-4) 
        else:
            #  ReduceLROnPlateau: Note  that step should be called after validate()
            scheduler.step(valid_loss)        
            # scheduler.step()
       
        # calculate average loss over an epoch
        avg_valid_losses.append(valid_loss)
        # do checkpointing
        if (valid_loss < maxLoss):
            maxLoss = valid_loss
            print('New checkpoint... validade loss {:8.5f}, train loss {:8.5f}'.format(valid_loss, train_loss))
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       '{}/checkpoint_{:06}_{:08.5f}.pth'.format(log_dir, epoch, valid_loss))
            
        # Save file with data training
        allLoss = {'train':avg_train_losses, 'valid': avg_valid_losses}
        ofile = open(training_File_Data, "wb")
        dill.dump(allLoss, ofile)
        ofile.close()      
        
    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    print('Lowest validation loss at epoch %d' %minposs)
    
    # visualize the loss and learning rate as the network trained
    #visualize_the_losses(avg_train_losses, avg_valid_losses)
    allLoss = {'train':avg_train_losses, 'valid': avg_valid_losses}
    ofile = open(training_File_Data, "wb")
    dill.dump(allLoss, ofile)
    ofile.close()  
