import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Reference: https://nervanasystems.github.io/distiller/knowledge_distillation.html.

    Args:
        temperature (float): Temperature value used when calculating soft targets and logits.
    """
    def __init__(self, temperature=4.):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logit, teacher_logit):
        soft_log_prob = F.log_softmax(student_logit / self.temperature)
        soft_target = torch.softmax(teacher_logit / self.temperature)
        loss = F.kl_div(soft_log_prob, soft_target.detach(), reduction='sum') / teacher_logit.shape[0]
        return loss


class SigmoidFocalLoss(nn.Module):
    """
    Compute focal loss from 
    `'Focal Loss for Dense Object Detection' (https://arxiv.org/pdf/1708.02002.pdf)`.
    
    Args:
        gamma (float): (default=2.).
        alpha (float): (default=0.25).
    """
    def __init__(self, gamma=2., alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def _sigmoid_focal_loss_cpu(self, logit, target, gamma, alpha):
        p = torch.sigmoid(logit)
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        return -target * term1 * alpha - (1 - target) * term2 * (1 - alpha)

    def forward(self, logit, target):
        loss = self._sigmoid_focal_loss_cpu(logit, target, self.gamma, self.alpha)
        pos_inds = torch.nonzero(target > 0).squeeze(1)
        N = target.size(0)
        loss = loss.sum() / (pos_inds.numel() + N)
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


class SoftmaxFocalLoss(nn.Module):
    """
    Compute the softmax version of focal loss.
    Loss value is normalized by sum of modulating factors.

    Args:
        gamma (float): (default=2.).
    """
    def __init__(self, gamma=2.):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target):
        logp = self.ce(logit, target)
        p = torch.exp(-logp)
        modulate = (1 - p) ** self.gamma
        loss = modulate * logp / modulate.sum()
        return loss.sum()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Negative log-likelihood loss with label smoothing.

    Args:
        smoothing (float): label smoothing factor (default=0.1).
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logit, target):
        logprobs = F.log_softmax(logit, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
