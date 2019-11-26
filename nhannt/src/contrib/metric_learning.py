import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 


class ArcFace(nn.Module):
    """
    Implement large margin arc distance.

    Args:
        embedding_size (int, required): size of each input sample.
        num_class (int, required): size of each output sample.
        s (float): norm of input feature (default=30.0).
        m (float): additive margin e.g. cos(theta + m) (default=0.50).
    """

    def __init__(self, embedding_size, num_class,
                 s=30.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.scaler = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_class, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels=None):
        # calculate cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.clamp(-1., 1.)
        if self.training:
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # convert label to one-hot 
            one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            # cast to float16 for mp training
            if inputs.dtype == torch.float16:
                one_hot = one_hot.half()
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # torch.where(out_i = {x_i if condition_i else y_i)
            output *= self.scaler
            return output
        cosine *= self.scaler
        return cosine