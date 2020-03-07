import logging
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from models.blocks import *
import torch.nn.functional as F

from models.inception import BasicConv2d
from models.res2net import res2net50
from models.resnet import resnet50

class Aug_Res2net(nn.Module):

    def __init__(self, M=10, num_classes=2, pretrained=False):
        super(Aug_Res2net, self).__init__()
        self.M = M
        self.features = resnet50(pretrained=pretrained).get_features()
        self.num_features = 512 * self.features[-1][-1].expansion
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):

        feature_map = self.features(x)
        x1 = self.avgpool(feature_map)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)

        attention_maps = self.attentions(feature_map)
        attention_map = torch.mean(attention_maps, dim=1, keepdim=True)

        return x1, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(Aug_Res2net, self).load_state_dict(model_dict)


