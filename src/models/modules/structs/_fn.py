"""
Used for parallelized cyk
related codebase:
parser: https://github.com/yzhangcs/parser/blob/main/supar/utils/fn.py
tn-pcfg: https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/fn.py
nner_as_parsing: https://github.com/LouChao98/nner_as_parsing/blob/main/src/modules/loss/parsing/_fn.py
"""

import torch

def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

def diagonal_copy_(x, y, w, r_closed = False):
    """
    r_closed: whether the right range is closed. True: [i,j] False: [i,j)
    size of x: (batch, N, N, nt)
    size of y: (batch, N, nt)
    the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    """
    # r_offeset = 1 if r_closed else 0
    r_offset = 0
    
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset=(w - r_offset) * stride[2]).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=(w - r_offset) * stride[2]).copy_(y)

def diagonal(x, w, r_closed = False):
    """
    r_closed: whether the right range is closed. True: [i,j] False: [i,j)
    """
    # r_offeset = 1 if r_closed else 0
    r_offset = 0
    
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=(w - r_offset) * stride[2])
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=(w - r_offset) * stride[2])

def entropy_sum(xs, dim):
    part = torch.logsumexp(xs[..., 0], dim=dim)
    log_sm = xs[..., 0] - part.unsqueeze(dim)
    sm = log_sm.exp()
    # breakpoint()
    return torch.stack((part, torch.sum(xs[..., 1].mul(sm) - log_sm.mul(sm), dim=dim)), dim = -1)
