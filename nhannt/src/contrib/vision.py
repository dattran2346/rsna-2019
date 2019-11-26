import torch
import torch.nn as nn 
import torch.nn.functional as F 

from .activation import swish, Swish


# Default args for PyTorch BN impl
_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)

# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)


class ConvBNAct(nn.Module):
    """
    Conv-BatchNorm-Act block.
    """
    def __init__(self, in_channels, out_channels, act, bn_args=_BN_ARGS_PT, **kwargs):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, **bn_args)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "swish":
            self.act = Swish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PyramidPooling(nn.Module):
    """
    Pyramid pooling module from `'Asymmetric Non-local Neural Networks for Semantic Segmentation' (https://arxiv.org/pdf/1908.07678.pdf)`.
    """
    def __init__(self, sizes=[1, 3, 6, 8]):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(sizes[0])
        self.pool2 = nn.AdaptiveAvgPool2d(sizes[1])
        self.pool3 = nn.AdaptiveAvgPool2d(sizes[2])
        self.pool4 = nn.AdaptiveAvgPool2d(sizes[3])

    def forward(self, x):
        n, c, _, _ = x.size()
        feat1 = self.pool1(x).view(n, c, -1)
        feat2 = self.pool2(x).view(n, c, -1)
        feat3 = self.pool3(x).view(n, c, -1)
        feat4 = self.pool4(x).view(n, c, -1)
        return torch.cat((feat1, feat2, feat3, feat4), -1)


class SelfAttentionBlock(nn.Module):
    """
    Asymmetric pyramid (fusion) non-local block from `'Asymmetric Non-local Neural Networks for Semantic Segmentation' (https://arxiv.org/pdf/1908.07678.pdf)`.
    """
    def __init__(self, high_in_channels, low_in_channels,
                 key_channels, value_channels, out_channels,
                 act):
        super(SelfAttentionBlock, self).__init__()
        self.key_channels = key_channels  # query / key channels
        self.value_channels = value_channels
        self.out_channels = out_channels

        self.W_q = ConvBNAct(high_in_channels, self.key_channels, act, kernel_size=1)
        self.W_k = ConvBNAct(low_in_channels, self.key_channels, act, kernel_size=1)
        self.W_v = ConvBNAct(low_in_channels, self.value_channels, act, kernel_size=1)
        self.W_o = ConvBNAct(self.value_channels, out_channels, act, kernel_size=1)
        self.pool = PyramidPooling()

    def forward(self, high_feat, low_feat):
        bsize, _, h, w = high_feat.size()
        query = self.W_q(high_feat).view(bsize, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # B x N x key_channels
        key = self.pool(self.W_k(low_feat))  # B x key_channels x S (S << N)
        sim_map = torch.matmul(query, key)  # B x N x S
        sim_map = sim_map * (self.key_channels ** -.5)
        sim_map = F.softmax(sim_map, dim=-1)

        value = self.pool(self.W_v(low_feat))
        value = value.permute(0, 2, 1)  # B x S x value_channels
        context = torch.matmul(sim_map, value)  # B x N x value_channels
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(bsize, self.value_channels, h, w)
        out = self.W_o(context) + high_feat
        return out