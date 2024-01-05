import logging
import pdb
import torch
from functools import partial
from torch import nn, unsqueeze
from opt_einsum import contract

log = logging.getLogger(__name__)
class SecondOrderMF(nn.Module):
    r"""
    ref: https://github.com/yzhangcs/parser/blob/16ad39534957bc4ee7af6ca8874de79332e8e8a2/supar/structs/vi.py
    """

    def __init__(self, conf):
        super().__init__()
        
        self.conf = conf
        self.score_types = conf.score_types
        self.structured = conf.structured
        self.max_iter = conf.max_iter
        self.max_iter_decode = conf.max_iter_decode
        
        self.mf_split = False
        if hasattr(conf, 'mf_split'):
            self.mf_split = conf.mf_split

        self.reset_score_types()
        
    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    def reset_score_types(self):
        if not isinstance(self.score_types, str):
            score_types = ""
            if 'pspan' in self.score_types:
                score_types += 'pI'
            if 'psib' in self.score_types:
                score_types += 'sI'
            if 'pcop' in self.score_types:
                score_types += 'c'
            self.score_types = score_types

    @torch.enable_grad()
    def forward(self, foscores, soscores, masks, inference = False):
        
        if inference:
            self.max_iter = self.max_iter_decode

        if self.structured:
            # if self.mf_split:
            #     span_logits, ph_logits, pt_logits = self.structured_mfvi_split(foscores, soscores, maskspan, maskarc, mask2o_pspan, mask2o_psib, mask2o_pcop)
            # else:
            span_logits, ph_logits, pt_logits = self.structured_mfvi(foscores, soscores, masks)
        else:
            span_logits, ph_logits, pt_logits = self.mfvi(foscores, soscores, masks)

        # results = {'logits':[span_logits, ph_logits, pt_logits]}
        
        return span_logits, ph_logits, pt_logits
        
    def mfvi(self, foscores, soscores, masks):
                
        maskspan, maskarc, mask2o_p2rspan,\
        mask2o_psib, mask2o_pcop, mask2o_pspan2r = \
        masks["maskspan"], masks["maskarc"], \
        masks["mask2o_p2rspan"], masks["mask2o_psib"],\
        masks["mask2o_pcop"], masks["mask2o_pspan2r"]
        
        bsz, t, t = maskspan.shape

        span, ph, pt = foscores
        rspan_psh, rspan_pst, fspan_psh, fspan_pst,\
                ph_sib, pt_sib, ph_cop, pt_cop = soscores
        
        zero = mask2o_p2rspan.new_zeros(*mask2o_p2rspan.shape)
        
        # [bsz, t, t, t] (p, sh, st) grandparent p->sh->st & co-parent p,sh->st
        # if 'pspan' in self.score_types:
        # if 'pspan' in self.score_types:

        if 'p' in self.score_types:
            rspan_psh = rspan_psh * mask2o_p2rspan
            rspan_pst = rspan_pst * mask2o_p2rspan
            
            fspan_psh = fspan_psh * mask2o_pspan2r
            fspan_pst = fspan_pst * mask2o_pspan2r
            
        else:
            rspan_psh = rspan_psh * zero
            rspan_pst = rspan_pst * zero
            
            fspan_psh = fspan_psh * zero
            fspan_pst = fspan_pst * zero
        
        # [bsz, t, t, t] (p, p, s) co-parent & (p, s, s) sibling
        # if 'psib' in self.score_types:
        if 's' in self.score_types:
            ph_sib = ph_sib * mask2o_psib
            pt_sib = pt_sib * mask2o_psib
        else:
            ph_sib = ph_sib * zero
            pt_sib = pt_sib * zero
        # pdb.set_trace()
        if 'c' in self.score_types:
            ph_cop = ph_cop * mask2o_pcop
            pt_cop = pt_cop * mask2o_pcop
        else:
            ph_cop = ph_cop * zero
            pt_cop = pt_cop * zero
            
        q_span = span
        q_ph = ph
        q_pt = pt
        
        for _ in range(self.max_iter):
            q_span  = q_span.sigmoid()
            q_ph = q_ph.sigmoid()
            q_pt = q_pt.sigmoid()
            
            tmp_span = span + (q_ph.unsqueeze(3) * rspan_psh + q_pt.unsqueeze(2) * rspan_pst).sum(1) + (q_ph.unsqueeze(2) * fspan_psh + q_pt.unsqueeze(1) * fspan_pst).sum(3)
            tmp_ph = ph + (q_span.unsqueeze(1) * rspan_psh + q_ph.unsqueeze(2) * ph_sib).sum(3) + (q_ph.unsqueeze(1) * ph_cop + q_span.unsqueeze(3) * fspan_psh).sum(2)
            tmp_pt = pt + (q_pt.unsqueeze(2) * pt_sib).sum(3) + (q_span.unsqueeze(1) * rspan_pst + q_pt.unsqueeze(1) * pt_cop).sum(2) + (q_span.unsqueeze(3) * fspan_pst).sum(1)
            # tmp_ph = ph + (q_span.unsqueeze(1) * span_psh + q_ph.unsqueeze(2) * ph_sib).sum(3)
            # tmp_pt = pt + (q_pt.unsqueeze(2) * pt_sib).sum(3) + (q_span.unsqueeze(1) * span_pst).sum(2)
            
            q_span = tmp_span
            q_ph = tmp_ph
            q_pt = tmp_pt
            
        return q_span, q_ph, q_pt
        
    def structured_mfvi(self, foscores, soscores, masks):
    
        from .cyk import cyk        
                
        maskspan, maskarc, mask2o_p2rspan,\
        mask2o_psib, mask2o_pcop, mask2o_pspan2r = \
        masks["maskspan"], masks["maskarc"], \
        masks["mask2o_p2rspan"], masks["mask2o_psib"],\
        masks["mask2o_pcop"], masks["mask2o_pspan2r"]

        bsz, t, t = maskspan.shape
        cyk_m = partial(cyk, lens = maskspan[:, 0].sum(-1), r_closed = True)     

        
        span, ph, pt = foscores
        rspan_psh, rspan_pst, fspan_psh, fspan_pst,\
                ph_sib, pt_sib, ph_cop, pt_cop = soscores
        
        zero = mask2o_p2rspan.new_zeros(*mask2o_p2rspan.shape)
        
        # [bsz, t, t, t] (p, sh, st) grandparent p->sh->st & co-parent p,sh->st
        # if 'pspan' in self.score_types:
        # if 'pspan' in self.score_types:
        if 'p' in self.score_types:
            rspan_psh = rspan_psh * mask2o_p2rspan
            rspan_pst = rspan_pst * mask2o_p2rspan
            
            fspan_psh = fspan_psh * mask2o_pspan2r
            fspan_pst = fspan_pst * mask2o_pspan2r
            
        else:
            rspan_psh = rspan_psh * zero
            rspan_pst = rspan_pst * zero
            
            fspan_psh = fspan_psh * zero
            fspan_pst = fspan_pst * zero
        
        # [bsz, t, t, t] (p, p, s) co-parent & (p, s, s) sibling
        # if 'psib' in self.score_types:
        if 's' in self.score_types:
            ph_sib = ph_sib * mask2o_psib
            pt_sib = pt_sib * mask2o_psib
        else:
            ph_sib = ph_sib * zero
            pt_sib = pt_sib * zero
        # pdb.set_trace()
        if 'c' in self.score_types:
            ph_cop = ph_cop * mask2o_pcop
            pt_cop = pt_cop * mask2o_pcop
        else:
            ph_cop = ph_cop * zero
            pt_cop = pt_cop * zero
            
        q_span = span.clone()
        q_span_0 = q_span[...,0]
        q_span_1 = q_span[...,1]
        q_ph = ph.clone()
        q_pt = pt.clone()

        for _ in range(self.max_iter):
            q_span = torch.stack([q_span_0, q_span_1],dim=-1)
            
            if self.conf.treemarginal:
                q_span_ = q_span.logsumexp(-1)
                q_span_ = cyk_m(q_span_, marginals = q_span)
                q_span_1 = q_span_[...,1]
            else:
                q_span_1 = q_span[..., 1].sigmoid()
            
            q_ph = q_ph.sigmoid()
            q_pt = q_pt.sigmoid()
            
            tmp_span = span[..., 1] + (q_ph.unsqueeze(3) * rspan_psh + q_pt.unsqueeze(2) * rspan_pst).sum(1) + (q_ph.unsqueeze(2) * fspan_psh + q_pt.unsqueeze(1) * fspan_pst).sum(3)
            tmp_ph = ph + (q_span_1.unsqueeze(1) * rspan_psh + q_ph.unsqueeze(2) * ph_sib).sum(3) + (q_ph.unsqueeze(1) * ph_cop + q_span_1.unsqueeze(3) * fspan_psh).sum(2)
            tmp_pt = pt + (q_pt.unsqueeze(2) * pt_sib).sum(3) + (q_span_1.unsqueeze(1) * rspan_pst + q_pt.unsqueeze(1) * pt_cop).sum(2) + (q_span_1.unsqueeze(3) * fspan_pst).sum(1)
            
            q_span_1 = tmp_span
            q_ph = tmp_ph
            q_pt = tmp_pt
            
        q_span = torch.stack([q_span_0, q_span_1],dim=-1)

        return q_span, q_ph, q_pt


    # def structured_mfvi_split(self, foscores, soscores, maskspan, maskarc,  mask2o_pspan, mask2o_psib, mask2o_pcop):
    
    #     from .cyk import cyk        
    #     cyk_m = partial(cyk, lens = maskspan[:, 0].sum(-1), r_closed = True)        
        
    #     # import pdb
    #     # pdb.set_trace()
        
    #     bsz, t, t = maskspan.shape
        
    #     span, ph, pt = foscores
    #     span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop = soscores
        
    #     zero = mask2o_pspan.new_zeros(*mask2o_pspan.shape)
    #     # [bsz, t, t, t] (p, sh, st) grandparent p->sh->st & co-parent p,sh->st
    #     # if 'pspan' in self.score_types:

    #     if 'p' in self.score_types:
    #         span_psh = span_psh * mask2o_pspan[...,None]
    #         span_pst = span_pst * mask2o_pspan[...,None]
    #     else:
    #         span_psh = span_psh * zero[...,None]
    #         span_pst = span_pst * zero[...,None]
        
    #     # [bsz, t, t, t] (p, p, s) co-parent & (p, s, s) sibling
    #     if 's' in self.score_types:
    #         # ph_sib = ph_sib * mask2o_pss
    #         # pt_sib = pt_sib * mask2o_pss
    #         ph_sib = ph_sib * mask2o_psib[...,None]
    #         pt_sib = pt_sib * mask2o_psib[...,None]
    #     else:
    #         ph_sib = ph_sib * zero[...,None]
    #         pt_sib = pt_sib * zero[...,None]

    #     # if 'pcop' in self.score_types:
    #     if 'c' in self.score_types:
    #         ph_cop = ph_cop * mask2o_pcop[...,None]
    #         pt_cop = pt_cop * mask2o_pcop[...,None]
    #     else:
    #         ph_cop = ph_cop * zero[...,None]
    #         pt_cop = pt_cop * zero[...,None]
            
    #     q_span = span.clone()
    #     # q_span_0 = q_span[...,0]
    #     # q_span_1 = q_span[...,1]
    #     q_ph = ph.clone()
    #     q_pt = pt.clone()
        
    #     for _ in range(self.max_iter):
    #         # q_span = torch.stack([q_span_0, q_span_1],dim=-1)
            
    #         if self.conf.treemarginal:
    #             q_span_ = q_span.logsumexp(-1)
    #             q_span = cyk_m(q_span_, marginals = q_span)
    #         else:
    #             q_span = q_span.sigmoid()
            
    #         q_ph = q_ph.sigmoid()
    #         q_pt = q_pt.sigmoid()
            
    #         # tmp_span, tmp_ph, tmp_pt = self.propagate()
    #         # breakpoint()
    #         # 00: span, ph/pt all 0; 01: span 0, ph/pt 1; 10: span 1, ph/pt 0; 11: span 1, ph/pt 1
    #         ph_m = torch.stack([q_ph[...,0], q_ph[...,1], q_ph[...,0], q_ph[..., 1]], dim = -1)
    #         pt_m = torch.stack([q_pt[...,0], q_pt[...,1], q_pt[...,0], q_pt[..., 1]], dim = -1)
    #         span_m = torch.stack([q_span[..., 0], q_span[..., 0], q_span[..., 1], q_span[..., 1]], dim = -1)

    #         propogate2span = (ph_m.unsqueeze(3) * span_psh + pt_m.unsqueeze(2) * span_pst).sum(1)
    #         tmp_span_0 = span[..., 0] + propogate2span[..., 0] + propogate2span[..., 1]
    #         tmp_span_1 = span[..., 1] + propogate2span[..., 2] + propogate2span[..., 3]
    #         tmp_span = torch.stack([tmp_span_0, tmp_span_1], dim = -1)

    #         propogate2ph_p = (ph_m.unsqueeze(2) * ph_sib).sum(3) + (ph_m.unsqueeze(1) * ph_cop).sum(2)
    #         propogate2ph_span = (span_m.unsqueeze(1) * span_psh).sum(3)
    #         tmp_ph_0 = ph[..., 0] + propogate2ph_p[..., 0] + propogate2ph_p[..., 1] + propogate2ph_span[..., 0] + propogate2ph_span[..., 2]
    #         tmp_ph_1 = ph[..., 1] + propogate2ph_p[..., 2] + propogate2ph_p[..., 3] + propogate2ph_span[..., 1] + propogate2ph_span[..., 3]
    #         tmp_ph = torch.stack([tmp_ph_0, tmp_ph_1], dim = -1)

    #         propogate2pt_p = (pt_m.unsqueeze(2) * pt_sib).sum(3) + (pt_m.unsqueeze(1) * pt_cop).sum(2)
    #         propogate2pt_span = (span_m.unsqueeze(1) * span_pst).sum(2)
    #         tmp_pt_0 = pt[..., 0] + propogate2pt_p[..., 0] + propogate2pt_p[..., 1] + propogate2pt_span[..., 0] + propogate2pt_span[..., 2]
    #         tmp_pt_1 = ph[..., 1] + propogate2pt_p[..., 2] + propogate2pt_p[..., 3] + propogate2pt_span[..., 1] + propogate2pt_span[..., 3]
    #         tmp_pt = torch.stack([tmp_pt_0, tmp_pt_1], dim = -1)

            
    #         q_span = tmp_span
    #         q_ph = tmp_ph
    #         q_pt = tmp_pt
            

    #     return q_span, q_ph, q_pt


    # def propagate(self, span, ph, pt, q_span, q_ph, q_pt, span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop):
    #     tmp_span = span + (q_ph.unsqueeze(3) * span_psh + q_pt.unsqueeze(2) * span_pst).sum(1)
    #     tmp_ph = ph + (q_span.unsqueeze(1) * span_psh + q_ph.unsqueeze(2) * ph_sib).sum(3) + (q_ph.unsqueeze(1) * ph_cop).sum(2)
    #     tmp_pt = pt + (q_pt.unsqueeze(2) * pt_sib).sum(3) + (q_span.unsqueeze(1) * span_pst + q_pt.unsqueeze(1) * pt_cop).sum(2)

    #     return tmp_span, tmp_ph, tmp_pt    
            