import functorch.dim
import torch.nn as nn

from myUtils.config import Config
from myUtils.others import *


class CELoss(nn.Module):
    def __init__(self, binary=False):
        super().__init__()
        self.binary = binary

        if self.binary:
            self.forward = self.__forward_binary__
        else:
            self.forward = self.__forward_multi__

    def __forward_multi__(self, x, gt, mask=None, softmax_gt=False, weight=None, focal_r=0):
        # x [B, Cx, H, W]
        # gt [B, 1, H, W] or [B, Cx, H, W]
        # mask [B, 1, H, W]

        b, k, h, w = x.shape

        x = F.softmax(x, dim=1)

        focal = None

        if mask is not None or softmax_gt or focal_r > 0:

            if softmax_gt:
                gt = F.softmax(gt, dim=1)

            if mask is not None:
                mask = mask

            if focal_r > 0:
                focal = (1 - x).pow(focal_r)
                focal = focal[gt.to(torch.bool)].reshape(b, 1, h, w)
        else:
            if gt.shape[1] == 1:
                gt = gt.squeeze(1)

        loss_ce = F.cross_entropy(x, gt, weight=weight, reduction="none").unsqueeze(1)

        if focal is not None:
            loss_ce *= focal

        if mask is not None:
            loss_ce *= mask

        loss_ce = loss_ce.mean()

        return loss_ce

    def __forward_binary__(self, x, gt, mask=None, softmax_gt=False, weight=None, focal_r=0):
        # x [B, Cx, H, W]
        # gt [B, 1, H, W] or [B, Cx, H, W]
        # mask [B, 1, H, W]

        x = F.sigmoid(x)

        focal = None

        if mask is not None or softmax_gt or focal_r > 0:
            if softmax_gt:
                gt = F.sigmoid(gt)

            if mask is not None:
                mask = mask

            if focal_r > 0:
                focal0 = x.pow(focal_r)
                focal1 = (1 - x).pow(focal_r)
                focal = torch.where(gt >= 0.5, focal1, focal0)
        else:
            pass

        loss_ce = F.binary_cross_entropy(x, gt, weight=weight, reduction="none")

        if focal is not None:
            loss_ce *= focal

        if mask is not None:
            loss_ce *= mask

        loss_ce = loss_ce.mean()

        return loss_ce


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask=None):
        # [B, C, H, W]
        if mask is None:
            loss = F.mse_loss(x, y)
        else:
            loss = (F.mse_loss(x, y, reduction="none") * mask).mean()

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask=None):
        # [B, C, H, W]

        x = F.softmax(x, dim=1)

        loss = (2 * x * y + 1) / (x + y + 1)

        if mask is None:
            pass
        else:
            loss *= mask

        loss = 1 - loss

        loss = loss.mean()

        return loss

