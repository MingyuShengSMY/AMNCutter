import numpy as np
import torch
import torch.nn.functional as F


def get_ncut_cost(w, index_abcd, batch_kept=False):
    if len(w.shape) != 3:
        w = w.unsqueeze(0)   # [b, t, t]
        index_abcd = index_abcd.unsqueeze(0)

    xw = index_abcd.permute(0, 2, 1).bmm(w)

    xwx = xw.bmm(index_abcd)  # [b, c, t]

    d = xw.sum(dim=-1)

    d += 1e-10

    xwx_diag = xwx.diagonal(dim1=1, dim2=2)

    cut_cost = xwx_diag.div(d)

    if batch_kept:
        cut_cost = 1 - cut_cost.mean(dim=-1)
    else:
        cut_cost = 1 - cut_cost.mean()

    return cut_cost
