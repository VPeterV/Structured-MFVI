import pdb
import logging
import torch
from torch import nn
from opt_einsum import contract

log = logging.getLogger(__name__)
class SpanRelationLBP(nn.Module):
    r"""
    ref: https://github.com/yzhangcs/parser/blob/16ad39534957bc4ee7af6ca8874de79332e8e8a2/supar/structs/vi.py#L367
    Loopy Belief Propagation for approximately calculating marginals
    """

    def __init__(self, conf):
        super().__init__()

        self.max_iter = conf.max_lbp_iter
        self.edge_m_norm_coef = conf.edge_m_norm_coef
        self.edge_pred_softmax_temp = conf.edge_pred_softmax_temp
        self.edge_appearance_prob = conf.edge_appearance_prob

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, maskspan, mask2o_head, mask2o_tail, target=None):
        r"""
        Args:
            scores (~torch.Tensor * 6):
                Quintet of five tensors first order: `s_ner`, `s_shoh`, `s_stot` and `s_shohot`, `s_stohot`.
                `s_shst` (``[batch_size, seq_len, seq_len, 2]``) holds scores of all possible spans.
                `s_shoh` (``[batch_size, seq_len, seq_len, 2]``) holds scores of all possible relations between subject heads and object heads.
                `s_shot` (``[batch_size, seq_len, seq_len, 2]``) holds scores of all possible relations between subject heads and object tails.
                `s_{shst,shoh}` (``[batch_size, seq_len, seq_len, seq_len, 2]``) holds the scores of subject spans and object spans.
                `s_{shst,shot}` (``[batch_size, seq_len, seq_len, seq_len, 2]``) holds the scores of subject spans and object tails.
                `s_{shoh,shot}` (``[batch_size, seq_len, seq_len, seq_len, 2]``) holds the scores of subject heads 
                                                                                with object heads and subject heads with object tails.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor * 3): 
            `t_spans`(``[batch_size, seq_len, seq_len]``) holds the graph of all ner spans.
            `t_shoh`(``[batch_size, seq_len, seq_len]``) holds the golden chart of all relations between subject heads and object heads.
            `t_shot`(``[batch_size, seq_len, seq_len]``) holds the golden chart of all relations between subject heads and object tails.
                Default = None
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        shst_m, shoh_m, stot_m = self.lbp(*scores, mask, maskspan, mask2o_head, mask2o_tail)
        # marginals = logits.softmax(-1)

        # if target is None:
        #     return marginals
        # loss = self.cross_entropy(logits[mask], target[mask])

        # return shst_m.softmax(-1), shoh_m.softmax(-1), stot_m.softmax(-1)
        return shst_m, shoh_m, stot_m
        

    def lbp(self, shst, shoh, stot, shstoh, shstot, mask, maskspan, mask2o_head, mask2o_tail, refine = False):
    
        bsz, t, t = mask.shape
        
        #TODO all pesudo-marginals have been log_softmax
        #TODO will shst change? But since it does not matter
        
        eps = torch.finfo(shst.dtype).eps
        shstoh = torch.softmax(self.edge_pred_softmax_temp * shstoh, dim = -1) + eps
        shstot = torch.softmax(self.edge_pred_softmax_temp * shstot, dim = -1) + eps
        
        s_shst = shst.log_softmax(-1)
        s_shoh = shoh.log_softmax(-1)
        s_stot = stot.log_softmax(-1)
        
        shstoh = shstoh.reshape(bsz, t, t, t, 2, 2)
        shstot = shstot.reshape(bsz, t, t, t, 2, 2)
        
        # shstoh_dem = s_shst[:,:,:, None].unsqueeze(-1) + s_shoh[:,:,None,:].unsqueeze(-2)
        # shstot_dem = s_shst[:,:,:, None].unsqueeze(-1) + s_stot[:,None,:,:].unsqueeze(-2)
        # shstoh_dem = shst[:,:,:, None].unsqueeze(-2) + shoh[:,:,None,:].unsqueeze(-1)
        
        # factor (Heads, Entity)
        shstoh_dem = (shstoh.sum(-1).unsqueeze(-1) + eps).log() + (shstoh.sum(-2).unsqueeze(-2) + eps).log()
        # shoh_dem = shstoh.logsumexp(-1).unsqueeze(-1) + shstoh.logsumexp(-2).unsqueeze(-2)
        # f_nh = shstoh.log() - self.edge_m_norm_coef * shoh_dem
        f_nh = shstoh.log() - self.edge_m_norm_coef * shstoh_dem
        
        # factor (Tails, Entity)
        shstot_dem = (shstot.sum(-1).unsqueeze(-1) + eps).log() + (shstot.sum(-2).unsqueeze(-2) + eps).log()
        # f_nt = shstot.log() - self.edge_m_norm_coef * stot_dem
        f_nt = shstot.log() - self.edge_m_norm_coef * shstot_dem
        
        # f_nh.masked_fill_(~mask2o_head[...,None,None], -1e9)
        # f_nt.masked_fill_(~mask2o_tail[...,None,None], -1e9)
        
        # s_shst.masked_fill_(~maskspan[...,None], -1e9)
        # s_shoh.masked_fill_(~mask[...,None], -1e9)
        # s_stot.masked_fill_(~mask[...,None], -1e9)
        
        # log.info(s_shstot)
        # [bsz, sh, x, 2] => [2, sh, x, bsz]
        # s_shst = s_shst.permute(3, 1, 2, 0)
        # s_shoh = s_shoh.permute(3, 1, 2, 0)
        # s_stot = s_stot.permute(3, 1, 2, 0)
        
        # [bsz, sh, x, y, 2, 2] => [2, 2 ,sh, x, y, bsz]
        # f_nh = f_nh.permute(4, 5, 1, 2, 3, 0)
        # f_nt = f_nt.permute(4, 5, 1, 2, 3, 0)

        # mask2o_head = mask2o_head.permute(1, 2, 3, 0)
        # mask2o_tail = mask2o_tail.permute(1, 2, 3, 0)
        
        
        m_fnh_n = f_nh.new_zeros(*f_nh.shape[:-1]).log_softmax(-1)
        m_fnt_n = f_nh.new_zeros(*f_nh.shape[:-1]).log_softmax(-1)
        m_fnh_h = f_nh.new_zeros(*f_nh.shape[:-1]).log_softmax(-1)
        m_fnt_t = f_nh.new_zeros(*f_nh.shape[:-1]).log_softmax(-1)
        
        # pdb.set_trace()
        
        mn_init = (m_fnh_n * mask2o_head[...,None]).sum(3) + (m_fnt_n * mask2o_tail[...,None]).sum(3)
        mh_init = (m_fnh_h * mask2o_head[...,None]).sum(2)
        mt_init = (m_fnt_t * mask2o_tail[...,None]).sum(1)
        
        qs = torch.stack([mn_init, mh_init , mt_init])

        # pdb.set_trace()
        for i in range(self.max_iter):
            
            m_h_fnh = self.edge_appearance_prob * qs[1].unsqueeze(2) - m_fnh_h
            m_fnh_n = (m_h_fnh[...,None,:] + f_nh).logsumexp(-1).log_softmax(-1)
            
            m_t_fnt = self.edge_appearance_prob * qs[2].unsqueeze(1) - m_fnt_t
            m_fnt_n = (m_t_fnt[...,None,:] + f_nt).logsumexp(-1).log_softmax(-1)
            
            m_n_fnh = self.edge_appearance_prob * qs[0].unsqueeze(3) - m_fnh_n
            m_fnh_h = (m_n_fnh[...,:,None] + f_nh).logsumexp(-2).log_softmax(-1)
            
            m_n_fnt = self.edge_appearance_prob * qs[0].unsqueeze(3) - m_fnt_n
            m_fnt_t = (m_n_fnt[...,:,None] + f_nt).logsumexp(-2).log_softmax(-1)
            
            qs[0] = s_shst + (m_fnh_n * mask2o_head[...,None]).sum(3) + (m_fnt_n * mask2o_tail[...,None]).sum(3)
            qs[1] = s_shoh + (m_fnh_h * mask2o_head[...,None]).sum(2) 
            qs[2] = s_stot + (m_fnt_t * mask2o_tail[...,None]).sum(1)
            
            if i != (self.max_iter - 1):
                qs = qs.log_softmax(-1)
            
            
        # qs = qs.permute(0, 4, 2, 3, 1)
        
        # if not refine:
        if self.max_iter == 0:
            return shst, shoh, stot
        else:
            return qs[0], qs[1], qs[2]
        # else:
            

