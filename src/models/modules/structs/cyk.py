import pdb
import torch
from torch import autograd
from ._fn import diagonal, diagonal_copy_, stripe, entropy_sum

@torch.enable_grad()
def cyk(s_span, lens, r_closed=False, decode=False, mbr=False, marginals = None):
    """
    r_closed: whether the right range is closed. True: [i,j] False: [i,j)
    CYK alogrithm for treecrf
    ref:
    nner_as_parsing: https://github.com/LouChao98/nner_as_parsing/blob/main/src/modules/loss/parsing/_fn.py
    """
    if not s_span.requires_grad:
        s_span.requires_grad_(True)
        
    if marginals is not None and not marginals.requires_grad:
        marginals.requires_grad_(True)
        s_span = marginals.logsumexp(-1)
        
    bsz, seqlen = s_span.shape[:2]
    r_offset = 0 if r_closed else 1
    
    s = s_span.new_zeros(bsz, seqlen, seqlen, *s_span.shape[3:]).fill_(-1e9)
    s[:, torch.arange(seqlen - r_offset), torch.arange(seqlen - r_offset) + r_offset] = \
        s_span[:, torch.arange(seqlen - r_offset), torch.arange(seqlen - r_offset) + r_offset]
        
    for w in range(1 + r_offset, seqlen):
        n = seqlen - w
        left = stripe(s, n, w - r_offset, (0, r_offset))
        #TODO right offset: w or w-1? 
        right = stripe(s, n, w - r_offset, (1, w), 0)
        if decode:
            merge = (left + right).max(2)[0]
        else:
            merge = (left + right).logsumexp(2)
        # pdb.set_trace()
        merge = merge + diagonal(s_span, w, r_closed)
        diagonal_copy_(s, merge, w, r_closed)
    
    # if decode:
    #     breakpoint()
    logz = s[torch.arange(bsz), 0, lens + r_offset - 1]
    
    if not decode and marginals is None:
        return logz
    
    # if retain_graph:
        # logz.retain_grad()
    # pdb.set_trace()
    # logz.sum().backward()
    
    # if retain_graph:
        # s_span.retain_grad()
        # grads = autograd.grad()
    if marginals is not None:
        grads = autograd.grad(logz.sum(), marginals, create_graph=True)
        # pdb.set_trace()
        return grads[0] if isinstance(marginals, torch.Tensor) else grads
    else:
        grads = autograd.grad(logz.sum(), s_span, create_graph=True)
        grads = grads[0] if isinstance(s_span, torch.Tensor) else grads
        
    # predicted_spans = s_span.grad.nonzero()
    predicted_spans = grads.nonzero()
    return predicted_spans

@torch.enable_grad()
def treecrf_entropy(s_span_score, lens, r_closed=True, fix = False):
    """
    r_closed: whether the right range is closed. True: [i,j] False: [i,j)
    CYK alogrithm for treecrf
    ref:
    nner_as_parsing: https://github.com/LouChao98/nner_as_parsing/blob/main/src/modules/loss/parsing/_fn.py
    """
        
    bsz, seqlen = s_span_score.shape[:2]
    r_offset = 0 if r_closed else 1
    
    s_span = s_span_score.new_zeros(s_span_score.shape + (2,))
    s_span[..., 0] = s_span_score

    if fix:
        s_span = entropy_sum(s_span, -1)

    s = s_span.new_zeros(bsz, seqlen, seqlen, 2).fill_(-1e9)
    s[:, torch.arange(seqlen - r_offset), torch.arange(seqlen - r_offset) + r_offset] = \
        s_span[:, torch.arange(seqlen - r_offset), torch.arange(seqlen - r_offset) + r_offset]
    # breakpoint()
    for w in range(1 + r_offset, seqlen):
        n = seqlen - w
        left = stripe(s, n, w - r_offset, (0, r_offset))
        #TODO right offset: w or w-1? 
        right = stripe(s, n, w - r_offset, (1, w), 0)
        
        merge = entropy_sum(left + right, dim = 2)
        
        s_score = diagonal(s_span, w, r_closed)
        merge = merge + s_score
        diagonal_copy_(s, merge, w, r_closed)
    
    # if decode:
    #     breakpoint()
    logz = s[torch.arange(bsz), 0, lens + r_offset - 1][..., 1]
    # breakpoint()
    # if logz.sum() < 0:
        # breakpoint()
    # print(logz)
    return logz
    
if __name__ == '__main__':
    a = torch.randn(3,5,5,2)
    
    e = torch.randint(0,2, size=(3,5,5)).bool()
    a.masked_fill_(e[..., None], -1e9)
    # a[...,0].masked_fill_(e.bool(), -1e9)
    # a[...,1].masked_fill_(~e.bool(), -1e9)
    
    b = a.roll(1,-2)
    
    # print(a[0])
    # print(b)
    
    lens = torch.tensor([3,4,4], dtype=torch.long)
    
    # print(a)
    alogz = cyk(a, lens, True).logsumexp(-1)
    # print(alogz)
    blogz = cyk(b, lens, False).logsumexp(-1)
    
    # print(alogz)
    # print(blogz)
    print(alogz == blogz)
    
    c = a.logsumexp(-1)
    d = b.logsumexp(-1)
    
    # print(c[0])
    # print(d)
    
    clogz = cyk(c, lens, True)
    dlogz = cyk(d, lens, True)
    
    # print(clogz)
    # print(dlogz)
    
    # z = torch.zeros((3,50,50,2))
    # zlogz = cyk(z, lens, True).logsumexp(-1)
    # print(zlogz)
    print(alogz == clogz)
    print(blogz == dlogz)
    
    # a_pred = cyk(a, lens, True, decode = True)
    # b_pred = cyk(b, lens, False, decode = True)
    
    # print(a_pred)
    # print(b_pred)
    # print(a_pred[:,:2] == b_pred[:, :2])
    # print(a_pred[...,2] == b_pred[...,2]-1)
    a = torch.randn(3, 30, 30, 2)
    lens = torch.tensor([29,25,20], dtype = torch.long)
    # a_ = a.logsumexp(-1)
    # a_ = a.max(-1)[0]
    a_ = a[..., 1]
    b = cyk(a_, lens, r_closed = True, marginals=a)
    print(b[0,0,...,1])
    print(b[0,0,...])
    a_ = a.logsumexp(-1)
    b = cyk(a_, lens, r_closed = True, marginals=a)
    print(b[0,0,...,1])
    print(b[0,0,...])
    