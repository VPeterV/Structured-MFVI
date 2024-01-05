from typing import List, Dict
import torch
from torch import nn
from ..structs.cyk import cyk, treecrf_entropy
from ...utils import potential_norm, structure_smoothing

class TreecrfLossSRL(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, ent_lamb = 0.0, entropy_all = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_allc
        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, mask2o_head, mask2o_tail, mask2o_, span_mask = masks
        
        span_logits, ph, pt = logits
        
        # TreeCRF: global norm
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        spans_ind = spans_ind.to(torch.bool)
        
        if self.norm:
            span_logits = potential_norm(span_logits)
        
        span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        if self.smoothing < 1.0:
            span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        else:
            span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
            span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
            span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        span_logits_ob = span_logits_ob.logsumexp(-1)

        span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = True)
        logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = True)
        
        loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum()
        
        ent_loss = 0
        if self.ent_lamb > 0.:
            ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
            entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
            ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())

        return loss_spans + ent_loss
        
class TreecrfLoss(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, pos_weight = 1., neg_weight = 1., ent_lamb = 0.0, entropy_all = False, mf_split = False, nonh2t = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_all
        self.mf_split = mf_split
        self.nonh2t = nonh2t
        if self.mf_split:
            self.cseloss = nn.CrossEntropyLoss()
        else:
            if pos_weight == 1. and neg_weight == 1.:
                self.bceloss = nn.BCEWithLogitsLoss()
            else:
                self.bceloss = nn.BCEWithLogitsLoss(reduction = "none")
        

        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, mask2o_head, mask2o_tail, mask2o_, span_mask = masks
        
        span_logits, ph, pt = logits
        
        # TreeCRF: global norm
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        spans_ind = spans_ind.to(torch.bool)

        if self.norm:
            span_logits = potential_norm(span_logits)
        
        span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        if self.smoothing < 1.0:
            span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        else:
            span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
            span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
            span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        span_logits_ob = span_logits_ob.logsumexp(-1)

        span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = True)
        logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = True)
        
        ent_loss = 0
        if self.ent_lamb > 0.:
            ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
            entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
            ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())

        loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum() + ent_loss
        
        # local norm
        if self.mf_split:
            ph = ph.masked_select(maskarc[...,None])
            ph = ph.reshape(-1, 2)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc[..., None])
            pt = pt.reshape(-1, 2)
            pt_ind = pt_ind.masked_select(maskarc)

            loss_ph = self.cseloss(ph, ph_ind)
            loss_pt = self.cseloss(pt, pt_ind)

            if self.pos_weight != 1. or self.neg_weight != 1.:
                raise NotImplementedError

        else:
            ph = ph.masked_select(maskarc)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc)
            pt_ind = pt_ind.masked_select(maskarc)
            
            loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
            loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        if self.pos_weight != 1.:

            ph_weights = torch.where(ph_ind > 0, self.pos_weight, 1.)
            pt_weights = torch.where(pt_ind > 0, self.pos_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.neg_weight != 1.:
            # breakpoint()
            ph_weights = torch.where(ph_ind == 0, self.neg_weight, 1.)
            pt_weights = torch.where(pt_ind == 0, self.neg_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.pos_weight != 1. or self.neg_weight != 1.:
            loss_ph = loss_ph.mean()
            loss_pt = loss_pt.mean()

        if not self.nonh2t:
            if self.span_lamb > 0:
                loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
            else:
                loss = (loss_spans + loss_ph + loss_pt)        
        else:
                loss = loss_ph + loss_pt

        return loss

class TreecrfLoss_spanarc(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, pos_weight = 1., neg_weight = 1., ent_lamb = 0.0, entropy_all = False, mf_split = False, nonh2t = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_all
        self.mf_split = mf_split
        self.nonh2t = nonh2t
        if self.mf_split:
            self.cseloss = nn.CrossEntropyLoss()
        else:
            if pos_weight == 1. and neg_weight == 1.:
                self.bceloss = nn.BCEWithLogitsLoss()
            else:
                self.bceloss = nn.BCEWithLogitsLoss(reduction = "none")
        

        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, mask2o_head, mask2o_tail, mask2o_, span_mask = masks
        
        span_logits, ph, pt = logits
        
        # TreeCRF: global norm
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        spans_ind = spans_ind.to(torch.bool)

        if self.norm:
            span_logits = potential_norm(span_logits)
        
        span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        if self.smoothing < 1.0:
            span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        else:
            span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
            span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
            span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        span_logits_ob = span_logits_ob.logsumexp(-1)

        span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = True)
        logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = True)
        
        ent_loss = 0
        if self.ent_lamb > 0.:
            ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
            entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
            ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())

        loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum() + ent_loss
        
        # local norm
        if self.mf_split:
            ph = ph.masked_select(maskarc[...,None])
            ph = ph.reshape(-1, 2)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc[..., None])
            pt = pt.reshape(-1, 2)
            pt_ind = pt_ind.masked_select(maskarc)

            loss_ph = self.cseloss(ph, ph_ind)
            loss_pt = self.cseloss(pt, pt_ind)

            if self.pos_weight != 1. or self.neg_weight != 1.:
                raise NotImplementedError

        else:
            ph = ph.masked_select(maskarc)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc)
            pt_ind = pt_ind.masked_select(maskarc)
            
            loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
            loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        if self.pos_weight != 1.:

            ph_weights = torch.where(ph_ind > 0, self.pos_weight, 1.)
            pt_weights = torch.where(pt_ind > 0, self.pos_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.neg_weight != 1.:
            # breakpoint()
            ph_weights = torch.where(ph_ind == 0, self.neg_weight, 1.)
            pt_weights = torch.where(pt_ind == 0, self.neg_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.pos_weight != 1. or self.neg_weight != 1.:
            loss_ph = loss_ph.mean()
            loss_pt = loss_pt.mean()

        # if not self.nonh2t:
        #     if self.span_lamb > 0:
        #         loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
        #     else:
        #         loss = (loss_spans + loss_ph + loss_pt)        
        # else:
        #         loss = loss_ph + loss_pt

        return loss_spans

class TreecrfLoss_mloss(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, pos_weight = 1., neg_weight = 1., ent_lamb = 0.0, entropy_all = False, mf_split = False, nonh2t = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_all
        self.mf_split = mf_split
        self.nonh2t = nonh2t
        if self.mf_split:
            self.cseloss = nn.CrossEntropyLoss()
        else:
            if pos_weight == 1. and neg_weight == 1.:
                self.bceloss = nn.BCEWithLogitsLoss()
            else:
                self.bceloss = nn.BCEWithLogitsLoss(reduction = "none")
        

        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor], ph_arc: torch.Tensor) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, mask2o_head, mask2o_tail, mask2o_, span_mask = masks
        
        span_logits, ph, pt = logits
        
        # TreeCRF: global norm
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        spans_ind = spans_ind.to(torch.bool)

        if self.norm:
            span_logits = potential_norm(span_logits)
        
        # span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        # if self.smoothing < 1.0:
        #     span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        # else:
        #     span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
        #     span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
        #     span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        # span_logits_ob = span_logits_ob.logsumexp(-1)

        # span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits, maskspan[:, 0].sum(-1), r_closed = True, marginals = span_logits)
        # marginals = cyk(span_logits.logsumexp(-1), maskspan[:, 0].sum(-1), r_closed = True, marginals = span_logits)

        # ph_arc_m = ph_arc.sigmoid()

        # selected_m = marginals.masked_select(maskspan[..., None]).view(-1, 2)
        # selected_span_ind = spans_ind.masked_select(maskspan)
        # selected_pharc = ph_arc_m.masked_select(maskspan)
        # pos_m = selected_m[..., 1][selected_span_ind]
        # neg_m = selected_m[..., 0][~selected_span_ind]

        # pos_ph_arc_m = selected_pharc[selected_span_ind]
        # pos_m = pos_ph_arc_m * pos_m

        # neg_m = selected_m[~selected_span_ind]
        # neg_ph_arc_m = selected_pharc[~selected_span_ind]
        
        # pos_loss = pos_m.log().sum()
        # neg_loss = neg_m.log().sum()        

        # loss_spans = -(pos_loss + neg_loss) / maskspan[:, 0].sum()

        # loss_spans = -(pos_loss) / maskspan[:, 0].sum()

        ph_arc_m = ph_arc.sigmoid()

        selected_m = marginals.masked_select(maskspan)
        selected_span_ind = spans_ind.masked_select(maskspan).bool()
        selected_pharc = ph_arc_m.masked_select(maskspan)
        pos_m = selected_m[selected_span_ind]
        pos_ph_arc_m = selected_pharc[selected_span_ind]
        pos_m = pos_ph_arc_m * pos_m

        neg_m = selected_m[~selected_span_ind]
        neg_ph_arc_m = selected_pharc[~selected_span_ind]
        
        pos_loss = pos_m.log().sum()
        neg_loss = ( (1 - neg_m) + (1 - neg_ph_arc_m) * neg_m ).log().sum()

        loss_spans = -(pos_loss + neg_loss) / maskspan[:, 0].sum()
        # breakpoint()
        # breakpoint()
        # marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = True)
        # logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = True)
        
        # ent_loss = 0
        # if self.ent_lamb > 0.:
        #     ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
        #     entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
        #     ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())

        # loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum() + ent_loss


        
        # local norm
        if self.mf_split:
            ph = ph.masked_select(maskarc[...,None])
            ph = ph.reshape(-1, 2)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc[..., None])
            pt = pt.reshape(-1, 2)
            pt_ind = pt_ind.masked_select(maskarc)

            loss_ph = self.cseloss(ph, ph_ind)
            loss_pt = self.cseloss(pt, pt_ind)

            if self.pos_weight != 1. or self.neg_weight != 1.:
                raise NotImplementedError

        else:
            ph = ph.masked_select(maskarc)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc)
            pt_ind = pt_ind.masked_select(maskarc)
            
            loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
            loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        if self.pos_weight != 1.:

            ph_weights = torch.where(ph_ind > 0, self.pos_weight, 1.)
            pt_weights = torch.where(pt_ind > 0, self.pos_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.neg_weight != 1.:
            # breakpoint()
            ph_weights = torch.where(ph_ind == 0, self.neg_weight, 1.)
            pt_weights = torch.where(pt_ind == 0, self.neg_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.pos_weight != 1. or self.neg_weight != 1.:
            loss_ph = loss_ph.mean()
            loss_pt = loss_pt.mean()

        if not self.nonh2t:
            if self.span_lamb > 0:
                loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
            else:
                loss = (loss_spans + loss_ph + loss_pt)        
        else:
                loss = loss_ph + loss_pt

        return loss

class TreecrfLoss_mcse(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, pos_weight = 1., neg_weight = 1., ent_lamb = 0.0, entropy_all = False, mf_split = False, nonh2t = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_all
        self.mf_split = mf_split
        self.nonh2t = nonh2t
        if self.mf_split:
            self.cseloss = nn.CrossEntropyLoss()
        else:
            if pos_weight == 1. and neg_weight == 1.:
                self.bceloss = nn.BCEWithLogitsLoss()
            else:
                self.bceloss = nn.BCEWithLogitsLoss(reduction = "none")
        

        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor], ph_arc: torch.Tensor) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, mask2o_head, mask2o_tail, mask2o_, span_mask = masks
        
        span_logits, ph, pt = logits
        
        # TreeCRF: global norm
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        spans_ind = spans_ind.to(torch.bool)

        if self.norm:
            span_logits = potential_norm(span_logits)
        
        # span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        # if self.smoothing < 1.0:
        #     span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        # else:
        #     span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
        #     span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
        #     span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        # span_logits_ob = span_logits_ob.logsumexp(-1)

        # span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits.logsumexp(-1), maskspan[:, 0].sum(-1), r_closed = True, marginals = span_logits)

        # ph_arc_m = ph_arc.sigmoid()
        # breakpoint()
        selected_m = marginals.masked_select(maskspan[..., None]).view(-1, 2)
        selected_span_ind = spans_ind.masked_select(maskspan)
        # selected_pharc = ph_arc_m.masked_select(maskspan)
        pos_m = selected_m[..., 1][selected_span_ind]
        neg_m = selected_m[..., 0][~selected_span_ind]
        # pos_ph_arc_m = selected_pharc[selected_span_ind]
        # pos_m = pos_ph_arc_m * pos_m

        # neg_m = selected_m[~selected_span_ind]
        # neg_ph_arc_m = selected_pharc[~selected_span_ind]
        
        pos_loss = pos_m.log().sum()
        neg_loss = neg_m.log().sum()        

        # loss_spans = -(pos_loss + neg_loss) / maskspan[:, 0].sum()

        # loss_spans = -(pos_loss) / maskspan[:, 0].sum()

        # breakpoint()

        loss_spans = -(pos_loss + neg_loss) / maskspan[:, 0].sum()
        # breakpoint()
        # breakpoint()
        # marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = True)
        # logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = True)
        
        # ent_loss = 0
        # if self.ent_lamb > 0.:
        #     ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
        #     entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
        #     ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())

        # loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum() + ent_loss


        
        # local norm
        if self.mf_split:
            ph = ph.masked_select(maskarc[...,None])
            ph = ph.reshape(-1, 2)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc[..., None])
            pt = pt.reshape(-1, 2)
            pt_ind = pt_ind.masked_select(maskarc)

            loss_ph = self.cseloss(ph, ph_ind)
            loss_pt = self.cseloss(pt, pt_ind)

            if self.pos_weight != 1. or self.neg_weight != 1.:
                raise NotImplementedError

        else:
            ph = ph.masked_select(maskarc)
            ph_ind = ph_ind.masked_select(maskarc)
            
            pt = pt.masked_select(maskarc)
            pt_ind = pt_ind.masked_select(maskarc)
            
            loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
            loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        if self.pos_weight != 1.:

            ph_weights = torch.where(ph_ind > 0, self.pos_weight, 1.)
            pt_weights = torch.where(pt_ind > 0, self.pos_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.neg_weight != 1.:
            # breakpoint()
            ph_weights = torch.where(ph_ind == 0, self.neg_weight, 1.)
            pt_weights = torch.where(pt_ind == 0, self.neg_weight, 1.)

            loss_ph = loss_ph * ph_weights
            loss_pt = loss_pt * pt_weights

        if self.pos_weight != 1. or self.neg_weight != 1.:
            loss_ph = loss_ph.mean()
            loss_pt = loss_pt.mean()

        if not self.nonh2t:
            if self.span_lamb > 0:
                loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
            else:
                loss = (loss_spans + loss_ph + loss_pt)        
        else:
                loss = loss_ph + loss_pt

        return loss
        
class TreecrfLossORL(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, fencepost = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.r_closed = ~fencepost
        
    def forward(self, span_logits: torch.Tensor, spans_ind: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        
        maskspan, span_mask = masks["maskspan"], masks["span_mask"]

        # TreeCRF: global norm
        
        spans_ind = spans_ind.to(torch.bool)
        
        if self.norm:
            span_logits = potential_norm(span_logits)
        
        span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        if self.smoothing < 1.0:
            span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        else:
            span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
            span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
            span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        span_logits_ob = span_logits_ob.logsumexp(-1)

        span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = self.r_closed)
        logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = self.r_closed)
        
        loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum()

        # if self.span_lamb > 0:
        #     loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
        # else:
        #     loss = (loss_spans + loss_ph + loss_pt)        
        return loss_spans
        
class TreeCRFLossDict(nn.Module):
    def __init__(self, span_lamb, norm, smoothing = 1.0, fencepost = False, ent_lamb = 0., entropy_all = False) -> None:
        super().__init__()
        self.span_lamb = span_lamb
        self.norm = norm
        self.smoothing = smoothing      # structure smoothing
        self.r_closed = ~fencepost
        self.ent_lamb = ent_lamb
        self.entropy_all = entropy_all
        
    def forward(self, span_logits: torch.Tensor, spans_ind: torch.Tensor, masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        maskspan, span_mask = masks["maskspan"], masks["span_mask"]

        # TreeCRF: global norm
        
        spans_ind = spans_ind.to(torch.bool)
        
        if self.norm:
            span_logits = potential_norm(span_logits)
        
        span_logits_ob = span_logits.clone()
        
        # pdb.set_trace()
        
        if self.smoothing < 1.0:
            span_logits_ob = structure_smoothing(span_logits_ob, span_mask, spans_ind, self.smoothing)
        else:
            span_logits_ob.masked_fill_(span_mask[...,None], -1e9)
            span_logits_ob[..., 0].masked_fill_(spans_ind, -1e9)
            span_logits_ob[..., 1].masked_fill_( ~span_mask & ~spans_ind, -1e9)
        
        span_logits_ob = span_logits_ob.logsumexp(-1)

        span_logits_all = span_logits.logsumexp(-1)
        
        marginals = cyk(span_logits_ob, maskspan[:, 0].sum(-1), r_closed = self.r_closed)
        logz = cyk(span_logits_all, maskspan[:, 0].sum(-1), r_closed = self.r_closed)
        
        loss_spans = (logz - marginals).sum() / maskspan[:, 0].sum()
        
        ent_loss = 0
        if self.ent_lamb > 0.:
            ent_span_logits = span_logits_all if self.entropy_all else span_logits_ob
            entropy = treecrf_entropy(ent_span_logits, maskspan[:, 0].sum(-1), r_closed = True)
            ent_loss = self.ent_lamb * ( -(entropy.sum()) / maskspan[:, 0].sum())
            # if ent_loss < 0.:
            # breakpoint()
        # if self.span_lamb > 0:
        #     loss = self.span_lamb * loss_spans + (1 - self.span_lamb) * (loss_ph + loss_pt)
        # else:
        #     loss = (loss_spans + loss_ph + loss_pt)        
        return loss_spans + ent_loss