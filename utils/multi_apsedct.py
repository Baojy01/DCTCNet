import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Dict, Union


class MyAPSeDCTConv2d(nn.Module):
    """
        All phase sequency DCT(APSeDCT) which is implemented as convolution.
        kernel_size: the convolution kernel size, it must be an odd and not less than 3;
    """

    def __init__(self, channels, kernel_size=3, bias=False):
        super(MyAPSeDCTConv2d, self).__init__()

        assert kernel_size >= 3 and (kernel_size + 1) % 2 == 0

        self.dct_size = (kernel_size + 1) // 2
        self.channel = channels

        # self.Se_weight = nn.Parameter(torch.ones(channels, self.dct_size, self.dct_size))
        self.Se_weight = nn.Parameter(torch.randn(channels, self.dct_size, self.dct_size))
        self.Bias = nn.Parameter(torch.zeros(channels)) if bias else None

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.channel

        dd = {'device': x.device, 'dtype': x.dtype}
        G = self.ap_dct(self.channel, self.dct_size, dd)  # C, dct_size, dct_size

        Q_quarter = torch.matmul(torch.matmul(G, torch.exp(self.Se_weight)), G.transpose(1, 2))  # C, dct_size, dct_size
        # Q_quarter = G @ torch.exp(self.Se_weight) @ G.transpose(1, 2)  # C, dct_size, dct_size
        Q = self.GetQ(Q_quarter)  # C, 1, kernel_size, kernel_size
        out = F.conv2d(x, Q, bias=self.Bias, padding=self.dct_size - 1, groups=self.channel)  # B, C, H, W

        return out

    @staticmethod
    def ap_dct(channel, N, dd):
        T = torch.zeros((N, N), **dd)
        pi = math.pi
        for i in range(N):
            for u in range(N):
                if u == 0:
                    T[i, u] = (N - i) / N ** 2
                else:
                    T[i, u] = ((N - i) * math.cos(pi * u * i / N) - math.sin(pi * u * i / N) / math.sin(pi * u / N)) / (
                            N ** 2)

        return T.repeat(channel, 1, 1)

    @staticmethod
    def GetQ(y):
        n = y.shape[-1] - 1
        Q = F.pad(y, pad=[n, 0, n, 0], mode='reflect')

        return Q.unsqueeze(1)


class MultiAPSeDCT(nn.Module):
    """
        Multi-Scale All phase sequency DCT.
        channel: the input channels;
        kernel_size: the convolution kernel_size (such as 1,3,5,7); It can have two forms:
                    1. An integer of kernel_size, which assigns all heads with the same kernel_size.
                    2. A dict mapping kernel_size to #head splits (e.g. {kernel_size 1: #head split 1, kernel_size 2: #head split 2}).
                    It will apply different kernel_size to the head splits.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Dict[int, int]], groups=1):
        super(MultiAPSeDCT, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = {kernel_size: 1}
            heads = 1
        elif isinstance(kernel_size, dict):
            self.kernel_size = kernel_size
            heads = sum(kernel_size.values())
        else:
            raise ValueError()

        assert in_channels % heads == 0
        Ch = in_channels // heads

        self.conv_list = nn.ModuleList()
        self.head_splits = []

        for cur_kernel_size, cur_heads in self.kernel_size.items():
            if cur_kernel_size == 1:
                cur_conv = nn.Conv2d(cur_heads * Ch, cur_heads * Ch, kernel_size=3, padding=1, groups=cur_heads * Ch, bias=False)
            else:
                cur_conv = MyAPSeDCTConv2d(cur_heads * Ch, kernel_size=cur_kernel_size, bias=False)

            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_heads)

        self.channel_splits = [x * Ch for x in self.head_splits]

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x_list = torch.split(x, self.channel_splits, dim=1)  # Split according to channels.
        out_list = [conv(y) for conv, y in zip(self.conv_list, x_list)]
        out = torch.cat(out_list, dim=1)
        out = self.proj(out)

        return out
