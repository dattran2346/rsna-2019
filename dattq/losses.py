import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RSNALoss(nn.Module):

    def __init__(self, args):
        super(RSNALoss, self).__init__()
        if args.individual_loss:
            self.bce_loss = nn.BCEWithLogitsLoss(weight=args.class_weight, reduction='none')
        else:
            criterion = nn.BCEWithLogitsLoss(weight=args.class_weight, reduction='mean')
        self.focal_loss = SigmoidFocalLoss(args)

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        loss = args.bce_weight*bce_loss + args.focal_loss*focal_loss

        if torch.isnan(bce_loss):
            print('NAAAAAAAAAAAAAAAAAAAAAAAAAN')
            np.save('test/logits', logits.detach().cpu().numpy())
            np.save('test/targets', targets.detach().cpu().numpy())
        return loss


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -targets * term1 * alpha - (1 - targets) * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, args, gamma=2.0, alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.individual_loss = args.individual_loss
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss_cpu(logits, targets, self.gamma, self.alpha)
        pos_inds = torch.nonzero(targets > 0).squeeze(1)
        N = targets.size(0)
        if self.individual_loss:
            loss = loss.sum(0) / (pos_inds.numel() + N)
        else:
            loss = loss.sum() / (pos_inds.numel() + N)
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

class SoftDiceLoss(nn.Module):
    def forward(self, logits, targets):
        smooth = 1.
        intersection = logits * targets
        score = (2. * intersection.sum(1) + smooth) / (logits.sum(1) + targets.sum(1) + smooth)
        dice = score.sum() / targets.size(0)
        return 1. - dice

def sigmoid_kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=8, bsize=256, name='kappa', weights=None):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()

    # repeat_op = torch.reshape(torch.arange(0, N), [N, 1]).repeat((1, N))
    # repeat_op_sq = (repeat_op - torch.transpose(repeat_op, 0, 1))**2

    # weights = repeat_op_sq / (N-1)**2
    # weights = weights.float().cuda()


    pred_ = y_pred ** y_pow

    try:
        pred_norm = pred_ / (eps + torch.reshape(torch.sum(pred_, 1), [-1, 1]))
    except Exception:
        pred_norm = pred_ / (eps + torch.reshape(torch.sum(pred_, 1), [bsize, 1]))

    hist_rater_a = torch.sum(pred_norm, 0)
    hist_rater_b = torch.sum(y_true, 0)

    conf_mat = torch.matmul(torch.transpose(pred_norm, 0, 1), y_true)
    rater_mat = torch.matmul(
                    torch.reshape(hist_rater_a, [N, 1]),
                    torch.reshape(hist_rater_b, [1, N]))

    # nom = torch.sum(weights*conf_mat)
    # denom = torch.sum(weights*rater_mat/bsize)

    nom = torch.sum(conf_mat)
    denom = torch.sum(rater_mat/bsize)

    return nom / (denom + eps)

def roc_auc_loss_pt(
    logits,
    labels,
    weights=1.0,
    surrogate_type='xent',
    scope=None):

    # Convert weights to tensors
    weights = torch.FloatTensor([weights])
    if isinstance(weights, float):
        weights = torch.FloatTensor([weights])
    elif isinstance(weights, list):
        weights = torch.FloatTensor(weights)
    elif torch.is_tensor(weights):
        pass

    if weights.ndimension() == 1:
        # Weights has shape [batch_size]. Reshape to [batch_size, 1].
        weights = torch.reshape(weights, [-1, 1])
    if weights.ndimension() == 0:
        # Weights is a scalar. Change shape of weights to match logits.
        weights *= torch.ones_like(logits)

    weights = weights.to(logits.device)
    original_shape = logits.shape

    # Create tensors of pairwise differences for logits and labels, and
    # pairwise products of weights. These have shape
    # [batch_size, batch_size, num_labels].
    logits_difference = torch.unsqueeze(logits, 0) - torch.unsqueeze(logits, 1)
    labels_difference = torch.unsqueeze(labels, 0) - torch.unsqueeze(labels, 1)
    weights_product = torch.unsqueeze(weights, 0) * torch.unsqueeze(weights, 1)

    signed_logits_difference = labels_difference * logits_difference
    raw_loss = weighted_surrogate_loss_pt(
        labels=torch.ones_like(signed_logits_difference),
        logits=signed_logits_difference,
        surrogate_type=surrogate_type)
    weighted_loss = weights_product * raw_loss

    # Zero out entries of the loss where labels_difference zero (so loss is only
    # computed on pairs with different labels).
    loss = torch.mean(torch.abs(labels_difference) * weighted_loss, dim=0) * 0.5
    loss = torch.reshape(loss, original_shape)
    return loss

def weighted_surrogate_loss_pt(labels,
                            logits,
                            surrogate_type='xent',
                            positive_weights=1.0,
                            negative_weights=1.0):
    if surrogate_type == 'xent':
        return weighted_sigmoid_cross_entropy_with_logits_pt(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights)
    elif surrogate_type == 'hinge':
        return weighted_hinge_loss_pt(
          logits=logits,
          labels=labels,
          positive_weights=positive_weights,
          negative_weights=negative_weights)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)


def weighted_sigmoid_cross_entropy_with_logits_pt(labels,
                                               logits,
                                               positive_weights=1.0,
                                               negative_weights=1.0,
                                               name=None):
    positive_weights = torch.FloatTensor([positive_weights])
    negative_weights = torch.FloatTensor([negative_weights])
    positive_weights = expand_outer_pt(positive_weights, logits.ndimension())
    negative_weights = expand_outer_pt(negative_weights, logits.ndimension())
    positive_weights = positive_weights.to(logits.device)
    negative_weights = negative_weights.to(logits.device)

    softplus_term = torch.add(F.relu(-logits),
                              torch.log(1. + torch.exp(-torch.abs(logits))))

    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softplus_term)

def weighted_hinge_loss_pt(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0,
                        name=None):
    positive_weights = torch.FloatTensor([positive_weights])
    negative_weights = torch.FloatTensor([negative_weights])
    positive_weights = expand_outer_pt(positive_weights, logits.ndimension())
    negative_weights = expand_outer_pt(negative_weights, logits.ndimension())
    positive_weights = positive_weights.to(logits.device)
    negative_weights = negative_weights.to(logits.device)

    positives_term = positive_weights * labels * F.relu(1.0 - logits)
    negatives_term = (negative_weights * (1.0 - labels)
                      * F.relu(1.0 + logits, 0))
    return positives_term + negatives_term


def expand_outer_pt(tensor, rank):
    if len(tensor.shape) > rank:
        raise ValueError(
            '`rank` must be at least the current tensor dimension: (%s vs %s).' %
            (rank, len(tensor.get_shape())))
    while len(tensor.shape) < rank:
        tensor = tensor[None, ...]
    return tensor
