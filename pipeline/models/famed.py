"""FAMED model architecture for MEG spike detection.

Feature-Attentive Mega-channel EEG/MEG Detection (FAMED) model implementation.
Based on the architecture from RICOH Co. Ltd., by R.Hirano (2022).

The code is copyrighted by the authors. Permission to copy and use this software for
noncommercial use is hereby granted provided: (1) this notice is retained in all copies,
(2) the publication describing the method is clearly cited, and (3) the distribution from
which the code was obtained is clearly cited. For all other uses, please contact the authors.

The software code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied.
Use at your own risk.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class DropConnect(nn.Module):
    """DropConnect regularization layer."""

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0), 1, 1, 1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x


class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation block."""

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
        return torch.add(input=chn_se, other=spa_se, alpha=1.0)  # Element-wise addition


class Bottleneck(nn.Module):
    """Bottleneck residual block with DropConnect and SCSE attention."""

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
    """FAMED classification network for spike detection."""

    def __init__(self, block, layers, ch_num=275, inplanes=16, drop_connect_rate=0.2):
        self.inplanes = inplanes
        self.ch_num = ch_num
        self.drop_connect_rate = drop_connect_rate
        super(FAMED_Classification, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(1, int(inplanes / 2), kernel_size=3, stride=(1, 2), padding=(1, 2), dilation=(1, 2)),
            nn.BatchNorm2d(int(inplanes / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(inplanes / 2), inplanes, kernel_size=3, stride=(1, 2), padding=(1, 2), dilation=(1, 2)),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(inplanes * 2), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(inplanes * 4), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(inplanes * 8), layers[3], stride=2)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(int(inplanes * 8) * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.drop_connect_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_connect_rate=self.drop_connect_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        emb = self.gap(x).reshape(batch_size, -1)
        x = self.fc1(emb)
        x = torch.sigmoid(x)
        return x.squeeze(1), emb


class BasicBlock(nn.Module):
    """Basic residual block for upsampling in segmentation."""
    def __init__(self, in_channels, out_channels, kernel, padding, stride, output_padding, upsample=None,
                 drop_connect_rate=0.2):
        super(BasicBlock, self).__init__()
        self.drop_connect = DropConnect(drop_connect_rate)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=False,
                                        stride=stride, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.scse = SCSEBlock(out_channels)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        if self.upsample is not None:
            residual = self.upsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.scse(out)
        res = self.drop_connect(out) + residual
        res = self.relu(res)

        return res


class FAMED_segmentation(nn.Module):
    """FAMED segmentation network with U-Net style decoder."""

    def __init__(self, base_model, n_class, input_shape):
        super(FAMED_segmentation, self).__init__()
        self.base_model = base_model
        module_list = [name for name, _ in self.base_model.named_modules()]
        self.inplanes = self.base_model.layer1[0].conv1.in_channels
        self.n_chs = [np.ceil(input_shape[0] / (2 ** i)).astype(int) for i in range(0, 4)]
        self.n_chs.insert(0, input_shape[0])
        self.n_tps = [np.ceil(input_shape[1] / (2 ** i)).astype(int) for i in range(0, 6) if i != 1]  # timepoints

        self.base_layers = list(base_model.children())
        self.layer0 = self.base_model.init

        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4

        self.upsample4 = self.upconv(self.inplanes * 32, self.inplanes * 16, 3, 1, 2, 1)  # 15,16 -> 30, 32
        self.upsample3 = self.upconv(self.inplanes * 16 * 2, self.inplanes * 8, 3, 1, 2, (0, 1))  # 30,32 -> 59,64
        self.upsample2 = self.upconv(self.inplanes * 8 * 2, self.inplanes * 4, 3, 1, 2, (0, 1))  # 59,64 -> 117,128
        self.upsample1 = self.upconv(self.inplanes * 4 * 2, self.inplanes, 3, 1, 2, 1)  # 117, 128 -> 234, 256
        self.upsample0 = nn.Sequential(
            self.upconvrelu(self.inplanes * 2, self.inplanes, 3, 1, (1, 2), (0, 1)),
            self.upconvrelu(self.inplanes, self.inplanes, 3, 1, (1, 2), (0, 1))
        )  # 234, 256 -> 234, 1024

        self.conv_up4 = self._make_layer(self.inplanes * 32, self.inplanes * 16, 3, 1, 2, 1, self.upsample4)
        self.conv_up3 = self._make_layer(self.inplanes * 16 * 2, self.inplanes * 8, 3, 1, 2, (0, 1), self.upsample3)
        self.conv_up2 = self._make_layer(self.inplanes * 8 * 2, self.inplanes * 4, 3, 1, 2, (0, 1), self.upsample2)
        self.conv_up1 = self._make_layer(self.inplanes * 4 * 2, self.inplanes, 3, 1, 2, 1, self.upsample1)

        self.logit = nn.Conv2d(self.inplanes, n_class, 1)

        for n, m in self.named_modules():
            if 'base_model' in n:
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, in_channels, out_channels, kernel, padding, stride, output_padding, upsample=None):
        upsample = upsample
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, kernel, padding, stride, output_padding, upsample))
        return nn.Sequential(*layers)

    def forward(self, input):

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.conv_up4(layer4)
        x = torch.cat([x, layer3], dim=1)

        x = self.conv_up3(x)
        x = torch.cat([x, layer2], dim=1)

        x = self.conv_up2(x)
        x = torch.cat([x, layer1], dim=1)

        x = self.conv_up1(x)
        x = torch.cat([x, layer0], dim=1)

        x = self.upsample0(x)
        x_out = self.logit(x)
        return x_out

    def convrelu(self, in_channels, out_channels, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconvrelu(self, in_channels, out_channels, kernel, padding, stride, output_padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, padding=padding, stride=stride,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels, kernel, padding, stride, output_padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, padding=padding, stride=stride,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels)
        )


class FAMEDWrapper(nn.Module):
    """Wrapper for FAMED model to handle MEG input format."""
    
    def __init__(self, n_classes=1, layers=[2, 2, 2, 2], ch_num=275, inplanes=16, drop_connect_rate=0.2, **kwargs):
        super(FAMEDWrapper, self).__init__()
        self.famed = FAMED_Classification(Bottleneck, layers, ch_num, inplanes, drop_connect_rate)
        
    def forward(self, x):
        """Forward pass through FAMED.

        Args:
            x: Input tensor of shape (batch_size, n_sensors, n_samples_per_window).

        Returns:
            Classification output.
        """
        x = x.unsqueeze(1)
        output, embedding = self.famed(x)
        return output


def get_FAMED_Cls(window_size, layers=[2, 2, 2, 2], ch_num=275, inplanes=16, drop_connect_rate=0.2):
    model = FAMED_Classification(Bottleneck,
                                 layers,
                                 ch_num,
                                 inplanes,
                                 drop_connect_rate
                                 )
    return model


def get_FAMED_Seg(window_size, layers=[2, 2, 2, 2], ch_num=275, inplanes=16, drop_connect_rate=0.2):
    encoder = FAMED_Classification(Bottleneck,
                                   layers,
                                   ch_num,
                                   inplanes,
                                   drop_connect_rate
                                   )
    model = FAMED_segmentation(encoder, n_class=1,
                               input_shape=(234, window_size))
    return model


if __name__ == '__main__':
    # Example usage
    model = FAMEDWrapper()
    # input_tensor = torch.randn(32, 275, 40)
    # # output = model(input_tensor)
    # # print(output)
    
    
    # print summary
    from torchsummary import summary
    summary(model, (275, 40))