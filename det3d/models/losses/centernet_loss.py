import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat
import pdb

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class ForecastLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(ForecastLoss, self).__init__()
  
  def forward(self, current_pred, future_pred, current_gt, future_gt, mask, ind, reverse):
    current_pred = _transpose_and_gather_feat(current_pred, ind)
    future_pred = _transpose_and_gather_feat(future_pred, ind)

    mask = mask.float().unsqueeze(2) 
    current_pred_reg = current_pred[:,:,:2]
    current_pred_vel = current_pred[:,:,6:8]
    target = future_pred[:,:,:2]

    current_gt_reg = current_gt[:,:,:2]
    current_gt_vel = current_gt[:,:,6:8]
    future_gt_reg = future_gt[:,:,:2]
    future_gt_vel = future_gt[:,:,6:8]

    time = (future_gt_reg - current_gt_reg) / (current_gt_vel + 1e-4)

    if reverse:
      pred = current_pred_reg - time * current_pred_vel 
      loss = F.l1_loss(pred*mask, target*mask, reduction='none')

    else:
      pred = current_pred_reg + time * current_pred_vel 
      loss = F.l1_loss(pred*mask, target*mask, reduction='none')

    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)

    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
