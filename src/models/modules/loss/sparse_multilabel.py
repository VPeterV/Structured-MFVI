import pdb
from turtle import forward
import torch
from torch import nn
import numpy as np

class multilabel_categorical_crossentropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    '''
    多标签交叉熵损失的torch实现
    '''
    # shape = y_pred.shape
    # y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    # y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    # zeros = torch.zeros_like(y_pred[...,:1])
    # y_pred = torch.cat([y_pred, zeros], dim=-1)
    # if mask_zero:
    #     infs = zeros + 1e12
    #     y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    # y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    # y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    # if mask_zero:
    #     y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
    #     y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    # pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    # all_loss = torch.logsumexp(y_pred, dim=-1)
    # aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    # aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    # neg_loss = all_loss + torch.log(aux_loss)
    # loss = torch.mean(torch.sum(pos_loss + neg_loss))
    
    # return loss
    
    # def forward(self, y_pred, y_true, label_mask, mask_zero=False):
    #     # pdb.set_trace()
    #     zeros = torch.zeros_like(y_pred[...,:1])
    #     # pos_pred = y_pred.masked_select(y_true, dim = -1)
    #     pos_pred = y_pred.masked_fill(~(y_true.bool()), 1e9)
    #     pos_pred_aux = y_pred.masked_fill(~(y_true.bool()), -1e9)
        
    #     pos_pred_1 = torch.concat([pos_pred, zeros], dim = -1)
    #     pos_loss = torch.logsumexp(-pos_pred_1, dim = -1)
        
    #     all_pred = torch.concat([y_pred, zeros], dim = -1)
    #     all_loss = torch.logsumexp(all_pred, dim = -1)
        
    #     aux_loss = torch.logsumexp(pos_pred_aux, dim = -1) - all_loss
    #     aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    #     neg_loss = all_loss + torch.log(aux_loss)
        
    #     loss = torch.mean(torch.sum(pos_loss + neg_loss))
        
    #     return loss
    def forward(self, y_pred, y_true, label_mask, mask_zero=False):
        # pdb.set_trace()
        zeros = torch.zeros_like(y_pred[...,:1])
        
        pos_pred = y_pred.masked_fill(~(y_true.bool()), 1e9)
        pos_pred_1 = torch.concat([pos_pred, zeros], dim = -1)
        pos_loss = torch.logsumexp(-pos_pred_1, dim = -1)
        
        
        neg_pred = y_pred.masked_fill((y_true.bool()), -1e9)
        neg_pred_1 = torch.concat([neg_pred, zeros], dim = -1)
        neg_loss = torch.logsumexp(neg_pred_1, dim = -1)
        
        loss = torch.mean(torch.sum(pos_loss + neg_loss))
        
        return loss