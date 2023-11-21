import sys
import torch
import numpy as np

def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    # print(p_i_j)
    return p_i_j


def crossview_contrastive_Loss(view1, view2):
    """Contrastive loss for maximizng the consistency"""

    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    loss = - p_i_j * (- torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()

    return loss


def category_contrastive_loss(repre, gt, classes):
    """Category-level contrastive loss.

    This function computes loss on the representation corresponding to its groundtruth (repre, gt).  A

    Args:
      repre: [N, D] float tensor.
      gt: [N, 1] float tensor.
      classes:  int tensor.

    Returns:
      loss:  float tensor.
    """

    batch_size = gt.size()[0]
    F_h_h = torch.matmul(repre, repre.t())
    F_hn_hn = torch.diag(F_h_h)
    F_h_h = F_h_h - torch.diag_embed(F_hn_hn)

    label_onehot = torch.nn.functional.one_hot(gt, classes).float()

    label_num = torch.sum(label_onehot, 0, keepdim=True)
    F_h_h_sum = torch.matmul(F_h_h, label_onehot)
    label_num_broadcast = label_num.repeat([gt.size()[0], 1]) - label_onehot
    label_num_broadcast[label_num_broadcast == 0] = 1
    F_h_h_mean = torch.div(F_h_h_sum, label_num_broadcast)
    gt_ = torch.argmax(F_h_h_mean, dim=1)  # gt begin from 0
    F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]
    theta = (gt == gt_).float()
    F_h_hn_mean_ = F_h_h_mean.mul(label_onehot)
    F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1)
    return torch.sum(torch.relu(torch.add(theta, torch.sub(F_h_h_mean_max, F_h_hn_mean))))
