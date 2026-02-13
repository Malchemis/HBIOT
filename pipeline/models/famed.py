'''
The code is copyrighted by the authors. Permission to copy and use this software for noncommercial use is hereby granted provided: (1) this notice is retained in all copies, (2) the publication describing the method (indicated below) is clearly cited, and (3) the distribution from which the code was obtained is clearly cited. For all other uses, please contact the authors.
 
The software code is provided "as is" with ABSOLUTELY NO WARRANTY
expressed or implied. Use at your own risk.

(C)Copyright 2022 by RICOH Co. Ltd., by R.Hirano 
'''


import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class DropConnect(nn.Module):
    '''https://tzmi.hatenablog.com/entry/2020/02/06/183314'''
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0),1,1,1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x


class SCSEBlock(nn.Module):
    '''https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/models/scse.py'''
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)

        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_connect_rate=0.2):
        super(Bottleneck, self).__init__()
        self.drop_connect = DropConnect(drop_connect_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.scse = SCSEBlock(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.scse(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        res = self.drop_connect(out) + residual
        res = self.relu(res)

        return res


class FAMED_Classification(nn.Module):
    """Originally made for 1024ms@1000Hz applied to 200ms@200Hz.
    """
    def __init__(self, block, layers, ch_num, inplanes, drop_connect_rate=0.2):
        self.inplanes = inplanes
        self.ch_num = ch_num
        super(FAMED_Classification, self).__init__()
        
        # Reduced stride to (1,1) for our shorter input length
        self.init = nn.Sequential(
            nn.Conv2d(1, int(inplanes/2), kernel_size=3, stride=(1,1), padding=(1,1), dilation=(1,1)),
            nn.BatchNorm2d(int(inplanes/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(inplanes/2), inplanes, kernel_size=3, stride=(1,1), padding=(1,1), dilation=(1,1)),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=2, drop_connect_rate=drop_connect_rate)
        self.layer2 = self._make_layer(block, int(inplanes*2), layers[1], stride=2, drop_connect_rate=drop_connect_rate)
        self.layer3 = self._make_layer(block, int(inplanes*4), layers[2], stride=2, drop_connect_rate=drop_connect_rate)
        self.layer4 = self._make_layer(block, int(inplanes*8), layers[3], stride=2, drop_connect_rate=drop_connect_rate)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(int(inplanes*8) * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, drop_connect_rate=0.2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_connect_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.init(x)        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).reshape(batch_size,-1)

        x = self.fc1(x)
        return x.squeeze(1)


class FAMEDWrapper(nn.Module):
    """Wrapper for FAMED model to handle MEG input format."""
    
    def __init__(self, n_classes=1, layers=[2, 2, 2, 2], ch_num=275, inplanes=16, drop_connect_rate=0.2, *args, **kwargs):
        super(FAMEDWrapper, self).__init__()
        self.famed = FAMED_Classification(Bottleneck, layers, ch_num, inplanes, drop_connect_rate)
        
    def forward(self, x, *args, **kwargs):
        """Forward pass through FAMED.

        Args:
            x: Input tensor of shape (batch_size, n_sensors, n_samples_per_window).

        Returns:
            Classification output.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.famed(x)


if __name__ == '__main__':
    model = FAMEDWrapper().to('cuda' if torch.cuda.is_available() else 'cpu')
    from torchsummary import summary
    print(f"FAMED Model Summary for our input shape (275, 40):")
    summary(model, (275, 40), device='cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"FAMED Model Summary for original input shape (234, 1024):")
    # summary(model, (234, 1024), device='cuda' if torch.cuda.is_available() else 'cpu')