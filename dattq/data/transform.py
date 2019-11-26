import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping
    """
    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if h < w and largest:
            w, h = size, int(size * h / w)
        else:
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)


EIG_VALS = [0.2175, 0.0188, 0.0045]
EIG_VECS = [[-0.5675,  0.7192,  0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948,  0.4203]]
ALPHA_STD = 0.1
class Lighting(object):
    """
    PCA jitter transform on tensors
    """
    def __init__(self, alpha_std=ALPHA_STD, eig_val=EIG_VALS, eig_vec=EIG_VECS):
        self.alpha_std = alpha_std
        self.eig_val = torch.as_tensor(eig_val, dtype=torch.float).view(1, 3)
        self.eig_vec = torch.as_tensor(eig_vec, dtype=torch.float)

    def __call__(self, data):
        if self.alpha_std == 0:
            return data
        alpha = torch.empty(1, 3).normal_(0, self.alpha_std)
        rgb = ((self.eig_vec * alpha) * self.eig_val).sum(1)
        data += rgb.view(3, 1, 1)
        data /= 1. + self.alpha_std
        return data


class Bound(object):
    def __init__(self, lower=0., upper=1.):
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        return data.clamp_(self.lower, self.upper)


class ResizeMinMax(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image
