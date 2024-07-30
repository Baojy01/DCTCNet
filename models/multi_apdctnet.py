from torch import nn
from utils import MultiAPSeDCT


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size: Union[int, Dict[int, int]] = 3, downsample=None, use_dct=False):
        super(BasicBlock, self).__init__()

        if use_dct:
            self.conv1 = nn.Sequential(MultiAPSeDCT(in_planes, planes, kernel_size=kernel_size),
                                       nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True))

            self.conv2 = MultiAPSeDCT(planes, planes, kernel_size=kernel_size)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(inplace=True))

            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, kernel_size=3, downsample=None, use_dct=False):
#         super(BasicBlock, self).__init__()

#         self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
#                                    nn.BatchNorm2d(planes),
#                                    nn.ReLU(inplace=True))
#         if use_dct:
#             self.conv2 = MultiAPSeDCT(planes, planes, kernel_size=kernel_size)
#         else:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

#         self.bn2 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(residual)

#         out += residual
#         out = self.relu(out)
#         return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1, kernel_size=3, downsample=None, use_dct=False):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        if use_dct:

            self.conv2 = nn.Sequential(MultiAPSeDCT(channel, channel, kernel_size=kernel_size),
                                       nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
                                       nn.BatchNorm2d(channel),
                                       nn.ReLU(inplace=True))

        else:
            self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False),
                                       nn.BatchNorm2d(channel),
                                       nn.ReLU(inplace=True))

        self.conv3 = nn.Conv2d(channel, channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, kernel_size=3):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel_size=kernel_size, use_dct=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BottleNeck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))  # bs, 64, 112, 112
        out = self.maxpool(out)  # bs, 64, 56, 56

        out = self.layer1(out)  # bs, 64*4, 56, 56
        out = self.layer2(out)  # bs, 128*4, 28, 28
        out = self.layer3(out)  # bs, 256*4, 14, 14
        out = self.layer4(out)  # bs, 512*4, 7, 7

        out = self.avgpool(out)  # bs, 512*4, 1, 1
        out = out.view(out.size(0), -1)  # bs, 512*4
        out = self.fc(out)  # bs, 1000

        return out

    def _make_layer(self, block, channel, blocks, stride=1, kernel_size=3, use_dct=False):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = [block(self.in_channel, channel, stride=stride, kernel_size=kernel_size, downsample=downsample,
                        use_dct=use_dct)]
        self.in_channel = channel * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel, kernel_size=kernel_size, use_dct=use_dct))
        return nn.Sequential(*layers)


def MultiAPSeDCTNet18(num_classes, kernel_size):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, kernel_size=kernel_size)


def MultiAPSeDCTNet34(num_classes, kernel_size):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, kernel_size=kernel_size)


def MultiAPSeDCTNet50(num_classes, kernel_size):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, kernel_size=kernel_size)
