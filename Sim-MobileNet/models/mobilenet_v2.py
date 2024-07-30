import torch.nn as nn
from utils import MyAPSeDCTConv2d, MultiAPSeDCT


def ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )


# Pointwise Convolution
def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super(InvertedResidual, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.conv1 = Conv1x1BNReLU(in_channels, mid_channels)

        self.conv2 = ConvBNReLU(mid_channels, mid_channels, kernel_size, stride, mid_channels)

        self.conv3 = Conv1x1BN(mid_channels, out_channels)

        self.has_skip = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.has_skip:
            out = out + x

        return out


class DCTInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super(DCTInvertedResidual, self).__init__()
        mid_channels = in_channels * expand_ratio
        self.conv1 = Conv1x1BNReLU(in_channels, mid_channels)

        self.conv2 = nn.Sequential(MultiAPSeDCT(mid_channels, out_channels, kernel_size),
                                   nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
                                   nn.BatchNorm2d(out_channels))

        self.has_skip = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.has_skip:
            out = out + x

        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=200, use_dct=(False, False, False)):
        super(MobileNetV2, self).__init__()
        input_channel = 64
        last_channel = 1024

        inverted_residual_setting = [
            # t, c, n, s, use_dct
            [4, 64, 2, 1, use_dct[0]],
            [4, 128, 4, 2, use_dct[1]],
            [4, 256, 3, 2, use_dct[2]],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual  blocks
        for t, c, n, s, dct_used in inverted_residual_setting:
            block = DCTInvertedResidual if dct_used else InvertedResidual
            output_channel = c
            for i in range(n):
                # block = DCTInvertedResidual if dct_used and i > 0 else InvertedResidual
                stride = s if i == 0 else 1  # s=2 in first layer, others s=1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(last_channel, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
