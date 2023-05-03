import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import models.resnet as resnet


class background_resnet(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet18'):
        super(background_resnet, self).__init__()
        self.backbone = backbone
        numInputsTemp = 128
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
            numInputsTemp = 512
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
            numInputsTemp = 512
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
            numInputsTemp = 512
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
            numInputsTemp = 128
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
       
        # BLOCO ORIGINAL 
        self.embedding_size = embedding_size # <- Adicionado ADELINO
        self.numInputs = numInputsTemp
        self.fc0 = nn.Linear(numInputsTemp, embedding_size) # 128
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)
        
        # ADICIONADO ADELINO
        # self.embedding_size = embedding_size # <- Adicionado ADELINO
        # self.fc0 = nn.Linear(numInputsTEmp, self.embedding_size) # <- Adicionado ADELINO
        # self.bn0 = nn.BatchNorm1d(self.embedding_size) # <- Adicionado ADELINO
        # self.relu = nn.ReLU() # <- Adicionado ADELINO
        # self.last = nn.Linear(self.embedding_size, num_classes)  # <- Adicionado ADELINO

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        # getX = x                                                # <- Adicionado ADELINO
        # outX = F.adaptive_avg_pool2d(getX,1) # [batch, ?, 1, 1] # <- Adicionado ADELINO
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        
        out = F.adaptive_avg_pool2d(x,1) # [batch, 128, 1, 1] <- LINHA ORIGINAL
        out = torch.squeeze(out) # [batch, n_embed]
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), -1) # (n_batch, n_embed)        <- LINHA ORIGINAL
        spk_embedding = self.fc0(out)                       # <- LINHA ORIGINAL
        out = F.relu(self.bn0(spk_embedding)) # [batch, n_embed]
        out = self.last(out)

        # ----------------------------------------------------------------------
        # spk_embeddingX = outX.view(getX.size(0), -1)            # <- Adicionado ADELINO 
        # nFeatures = spk_embeddingX.shape[1]                     # <- Adicionado ADELINO 
        # fc = nn.Linear(nFeatures, nFeatures)                # <- Adicionado ADELINO 
        # bn = nn.BatchNorm1d(nFeatures)                      # <- Adicionado ADELINO 
        # spk_embeddingX = fc(spk_embeddingX)                 # <- Adicionado ADELINO 
        # spk_embeddingX = bn(spk_embeddingX)
        # return spk_embedding, out # <- LINHA ORIGINAL
        return spk_embedding, out # <- Adicionado ADELINO 