from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
import torch
import torch.nn as nn

# https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_pt = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-log_pt)
        loss = ((1 - pt) ** self.gamma) * log_pt
        return loss.mean()