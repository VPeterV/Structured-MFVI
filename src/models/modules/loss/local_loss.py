from typing import Dict, List
import torch
from torch import nn

class MultiLabelCategoricalCrossentropy(nn.Module):
    def __init__(self):
        super(MultiLabelCategoricalCrossentropy, self).__init__()
        
    def forward(self, entity_score, targets):
        """
        https://kexue.fm/archives/7359
        """
        entity_score = (1 - 2 * targets) * entity_score  # -1 -> pos classes, 1 -> neg classes
        entity_score_neg = entity_score - targets * 1e12  # mask the pred outputs of pos classes
        entity_score_pos = (entity_score - (1 - targets) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(entity_score[..., :1])
        entity_score_neg = torch.cat([entity_score_neg, zeros], dim=-1)
        entity_score_pos = torch.cat([entity_score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(entity_score_neg, dim=-1)
        pos_loss = torch.logsumexp(entity_score_pos, dim=-1)
        # breakpoint()
        return (neg_loss + pos_loss).mean()

class LocalLossNew(nn.Module):
    def __init__(self, ml_loss = False) -> None:
        super().__init__()
        
        self.ml_loss = ml_loss
        
        if ml_loss:
            self.local_loss = MultiLabelCategoricalCrossentropy()
        else:
            self.local_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator
        mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop, span_mask = masks
        
        span, ph, pt = logits
        
        t = ph.size(-1)
        
        # local norm
        if self.ml_loss:
            ph.masked_fill(~maskarc, -1e12)
            pt.masked_fill(~maskarc, -1e12)
            
            ph = ph.reshape(-1, t * t)
            pt = pt.reshape(-1, t * t)
            
            ph_ind = ph_ind.reshape(-1, t * t)
            pt_ind = pt_ind.reshape(-1, t * t)
            
            
        else:
            ph = ph.masked_select(maskarc)
            ph_ind = ph_ind.masked_select(maskarc)
        
            pt = pt.masked_select(maskarc)
            pt_ind = pt_ind.masked_select(maskarc)
        
        loss_ph = self.local_loss(ph, ph_ind.to(torch.float))
        loss_pt = self.local_loss(pt, pt_ind.to(torch.float))

        loss = loss_ph + loss_pt
        
        return loss

class LocalLoss(nn.Module):
    def __init__(self, ml_loss = False, pos_weight = 1.) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        if pos_weight == 1.:
            self.bceloss = nn.BCEWithLogitsLoss()
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction = "none")
        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator
        mask, maskarc, maskspan, _, _, _ = masks
        
        span, ph, pt = logits
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        span_logits = span.masked_select(maskspan)
        spans_ind = spans_ind.masked_select(maskspan)
        
        # if self.pos_weight == 1.:
        loss_spans = self.bceloss(span_logits, spans_ind.to(torch.float))
        # else:
            # breakpoint()
        if self.pos_weight != 1:
            breakpoint()
            span_weights = torch.where(spans_ind > 0, self.pos_weight, 1.)
            loss_spans = loss_spans * span_weights
            loss_spans = loss_spans.mean()

        # local norm
        ph = ph.masked_select(maskarc)
        ph_ind = ph_ind.masked_select(maskarc)
        
        pt = pt.masked_select(maskarc)
        pt_ind = pt_ind.masked_select(maskarc)
        
        loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
        loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        loss = loss_spans + loss_ph + loss_pt
        
        return loss

class LocalLossDict(nn.Module):
    def __init__(self, ml_loss = False) -> None: 
        super().__init__()
        self.ml_loss = ml_loss
        if self.ml_loss:
            self.bceloss = MultiLabelCategoricalCrossentropy()
        else:
            self.bceloss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: Dict, indicators: Dict, masks: Dict) -> torch.Tensor:
        
        all_loss = 0.
        
        for name in logits:
            mask = masks[name]
            if not self.ml_loss:
                score = logits[name].masked_select(mask)
                ind = indicators[name].masked_select(mask)
            else:
                t = logits[name].shape[1]
                score = logits[name].masked_fill(~mask, -1e12)
                score = score.reshape(-1, t*t)
                
                ind = indicators[name]
                ind = ind.reshape(-1, t*t)

            
            all_loss = all_loss + self.bceloss(score, ind.to(torch.float))
        
        return all_loss
        
class LocalLossORL(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: Dict, indicators: Dict, masks: Dict) -> torch.Tensor:
        
        all_loss = 0.
        
        for name in logits:

            mask = masks[name]
            score = logits[name].masked_select(mask)
            ind = indicators[name].masked_select(mask)

            
            all_loss = all_loss + self.bceloss(score, ind.to(torch.float))
        
        return all_loss
    
class LocalLoss_AB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: torch.Tensor, fo_indicator: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
        spans_ind, ph_ind, pt_ind= fo_indicator
        mask, maskarc, maskspan, _, _, _ = masks
        
        span, ph, pt = logits
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        span_logits = span.masked_select(maskspan)
        spans_ind = spans_ind.masked_select(maskspan)
        
        # loss_spans = self.bceloss(span_logits, spans_ind.to(torch.float))

        # local norm
        ph = ph.masked_select(maskarc)
        ph_ind = ph_ind.masked_select(maskarc)
        
        pt = pt.masked_select(maskarc)
        pt_ind = pt_ind.masked_select(maskarc)
        
        loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
        loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        # loss = loss_spans + loss_ph + loss_pt
        loss = loss_ph + loss_pt
        
        return loss