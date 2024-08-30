from torch import nn as nn
from torch.nn import functional as F

from myUtils.NcutUltils import get_ncut_cost
from myUtils.config import Config


class NCutterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_ncut_cost = get_ncut_cost

    def forward(self, w, x):
        b, t, d = x.shape

        x = F.softmax(x, dim=-1)

        cut_cost = self.get_ncut_cost(w, x)

        return cut_cost
