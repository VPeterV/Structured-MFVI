import torch
from torch import nn
from typing import Dict, List

class PhLabelLoss(nn.Module):
    def __init__(self, num_span_label) -> None:
        super().__init__()
        self.num_span_label = num_span_label
        self.cse = nn.CrossEntropyLoss()
        
    def forward(self, s_labels: torch.Tensor, arc_labels: torch.Tensor, ph_ind: torch.Tensor, debug = False) -> torch.Tensor:
        ph_scores = s_labels
        ph_labels = arc_labels
        
        if ph_ind.dtype is not torch.bool:
            selected_ph = ph_ind.gt(0)
        else:
            selected_ph = ph_ind
        ph = ph_scores.masked_select(selected_ph[...,None]).view(-1, self.num_span_label)
        ph_gold = ph_labels.masked_select(selected_ph)
        loss_ph = self.cse(ph, ph_gold)
        
        loss = loss_ph
        
        return loss
        
class BCELabelLossDict(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss()        
        
    def forward(self, logits: Dict, gold_labels: Dict, masks: Dict) -> torch.Tensor:
        
        all_loss = 0.
        
        for name in logits:
            mask = masks[name]
            score = logits[name].masked_select(mask[..., None])
            ind = gold_labels[name].masked_select(mask[..., None])
            
            all_loss = all_loss + self.bceloss(score, ind.to(torch.float))
        
        return all_loss
        
class CSELabelLossDict(nn.Module):
    def __init__(self, weight = None) -> None: 
        super().__init__()
        if weight is not None:
            self.cseloss = nn.CrossEntropyLoss(weight = weight)
        else:
            self.cseloss = nn.CrossEntropyLoss()
        
    def forward(self, logits: Dict, gold_labels: Dict, masks: Dict) -> torch.Tensor:
        
        all_loss = 0.
        
        for name in logits:
            label_dim = logits[name].size(-1)
            
            mask = masks[name]
            score = logits[name].masked_select(mask[..., None]).view(-1, label_dim)
            ind = gold_labels[name].masked_select(mask)
            
            all_loss = all_loss + self.cseloss(score, ind)
        
        return all_loss
        

    
