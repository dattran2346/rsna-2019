from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    Utility module to flatten a Torch tensor.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


def flatten(tensor):
    """
    Utility function to flatten a Torch tensor.
    """
    return tensor.view(tensor.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=1, concat_dim=1):
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = output_size
        self.concat_dim = concat_dim
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(self.output_size)
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, inputs):
        return torch.cat([self.adaptive_avg_pool2d(inputs),
                          self.adaptive_max_pool2d(inputs)], dim=self.concat_dim)


def adaptive_concat_pool2d(tensor, output_size=1, concat_dim=1):
    return torch.cat([F.adaptive_avg_pool2d(tensor, output_size),
                      F.adaptive_max_pool2d(tensor, output_size)], dim=concat_dim)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    use_relu=False,
    kaiming_init=True):
    """
    Conv2d with kaiming_normal_ init and optionally GroupNorm. + ReLU
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    modules = [('conv', conv)]
    if use_gn:
        modules.append(('gn', group_norm(out_channels)))
    if use_relu:
        modules.append(('relu', nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(modules))


def fc_with_kaiming_uniform(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    if use_gn:
        fc=nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        modules=[('fc', fc), ('gn', group_norm(hidden_dim))]
    fc=nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0) 
    modules=[('fc', fc)]
    return nn.Sequential(OrderedDict(modules))


def conv_with_kaiming_uniform(in_channels, out_channels, kernel_size,
                              stride=1, dilation=1, use_gn=False, use_relu=False):
    """
    Conv2d with kaiming_uniform_ init and optionally GroupNorm. + ReLU
    """
    conv=nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * (kernel_size - 1) // 2,
        dilation=dilation,
        bias=False if use_gn else True
    )
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(conv.weight, a=1)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    modules=[('conv', conv)]
    if use_gn:
        modules.append(('gn', group_norm(out_channels)))
    if use_relu:
        modules.append(('relu', nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(modules))
