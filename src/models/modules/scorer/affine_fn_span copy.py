import torch
from torch import nn
# from .triaffine import Triaffine
from supar.modules import MLP
from .affinelayer import Biaffine, Triaffine, apply_fencepost
from typing import List


def define_affine(input_dim, biaffine_dim, n_out, init):
    l_mlp = MLP(input_dim, biaffine_dim)
    r_mlp = MLP(input_dim, biaffine_dim)    
    biaffine = Biaffine(biaffine_dim, n_out = n_out, init = init)

    return l_mlp, r_mlp, biaffine
    
class FirstOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, n_span_labels: List):
        """
        Discarding biaffine labels replaced by span encoder
        """
        super(FirstOrderScorer, self).__init__()
        
        self.conf = conf
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        biaffine_dim = conf.biaffine_dim
        biaffine_label_dim = conf.biaffine_label_dim
        
        num_frame, num_role_labels = n_span_labels
        
        # spans scorers
        self.sl_mlp, self.sr_mlp, self.span_att = define_affine(input_dim, biaffine_dim, span_n_out, conf.arc_init)
        
        # arcs scorers
        self.ph_mlp, self.rh_mlp, self.h2h_att = define_affine(input_dim, biaffine_dim, arc_n_out, conf.arc_init)
        self.pt_mlp, self.rt_mlp, self.t2t_att = define_affine(input_dim, biaffine_dim, arc_n_out, conf.arc_init)
        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
        
    def affine_pipeline(self, x, l_mlp, r_mlp, dropout, affine):
        l = l_mlp(x)
        r = r_mlp(x)
        
        l = dropout(l)
        r = dropout(r)
        
        res = affine(l, r)
        
        return res
                
    def forward(self, word, fencepost = False):
        '''
            x: hidden states [bsz, seq_len, input_dim (hidden_dim)]
            =>
            sh: span head [bsz, seq_len, biaffine_dim]
            st: span tail [bsz, seq_len, biaffine_dim]
            p: relations subject heads [bsz, seq_len, biaffine_dim]
            h: relations object heads [bsz, seq_len, biaffine_dim]
            t:relations object tails [bsz, seq_len, biaffine_dim]
            =>
            shst: span head - span tails [bsz, seq_len, seq_len]
            ph: predicates -> span heads [bsz, seq_len, seq_len]
            pt: predicates -> span tails [bsz, seq_len, seq_len]
        '''
        # import pdb
        # pdb.set_trace()
        
        # spans
        boundary = word
        if fencepost:
            boundary = apply_fencepost(word)
        
        span = self.affine_pipeline(boundary, self.sl_mlp, self.sr_mlp, self.arc_dropout,self.span_att)
        span = span.triu() + span.triu(1).transpose(-1,-2)
        if self.conf.span_n_out > 1:
            span = span.permute(0,2,3,1)
                
        # arcs
        h2h = self.affine_pipeline(word, self.ph_mlp, self.rh_mlp, self.arc_dropout, self.h2h_att)
        t2t = self.affine_pipeline(word, self.pt_mlp, self.rt_mlp, self.arc_dropout, self.t2t_att)
        
        if self.conf.arc_n_out > 1:
            h2h = h2h.permute(0,2,3,1)
            t2t = t2t.permute(0,2,3,1)

        # labels
        h2h_label = self.affine_pipeline(word, self.ph_label_mlp, self.rh_label_mlp, self.label_dropout, self.h2h_label_att)
        t2t_label = self.affine_pipeline(word, self.pt_label_mlp, self.rt_label_mlp, self.label_dropout, self.t2t_label_att)
        
        h2h_label = h2h_label.permute(0,2,3,1)
        t2t_label = t2t_label.permute(0,2,3,1)
        
        # span_type = self.affine_pipeline(word, self.sl_type_mlp, self.sr_type_mlp, self.label_dropout, self.span_type_att)
        # span_type = span_type.permute(0,2,3,1)
        
        frames = self.affine_pipeline(word, self.sl_frame_mlp, self.sr_frame_mlp, self.label_dropout, self.frame_att)
        frames = frames.permute(0,2,3,1)
        
        return span, h2h, t2t
        
class SecondOrderScorer(nn.Module):
    def __init__(self, conf, input_dim):
        super(SecondOrderScorer, self).__init__()
        self.conf = conf
        n_out = conf.n_out
        decompose = conf.decompose
        ind_prd = conf.ind_prd if hasattr(conf, 'ind_prd') else False
        shared_span = conf.shared_span
        triaffine_dim = conf.triaffine_dim
        
        # self.split = True
        
        self.ph_mlp = MLP(input_dim, triaffine_dim)
        self.pt_mlp =  MLP(input_dim, triaffine_dim)
            
        self.h_mlp = MLP(input_dim, triaffine_dim)
        self.t_mlp = MLP(input_dim, triaffine_dim)
        
        # if shared_span:
        #     self.sh_mlp = self.h_mlp
        #     self.st_mlp = self.t_mlp
        # else:
        self.rh_mlp = MLP(input_dim, triaffine_dim)
        self.rt_mlp = MLP(input_dim, triaffine_dim)
        
        self.fh_mlp = MLP(input_dim, triaffine_dim)
        self.ft_mlp = MLP(input_dim, triaffine_dim)
        
        # s_hs_t, ps_t
        self.rspan_pt_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # s_hs_t, ps_h
        self.rspan_ph_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_h,ps_t, r_t (gradparent)
        self.fspan_pt_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_h,ps_t, r_h  (sibling)
        self.fspan_ph_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_h, ps'_h (ph sibling)
        self.ph_sib = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_t, ps'_t (pt sibling)
        self.pt_sib = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_h, p's_h (ph co-parent)
        self.ph_cop = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # ps_t, p's_t (pt co-parent)
        self.pt_cop = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        
        self.dropout = nn.Dropout(conf.mlp_dropout)
        
    def forward(self, x):
        '''
            x: hidden states [bsz, seq_len, input_dim (hidden_dim)]
            =>
            sh: span head [bsz, seq_len, biaffine_dim]  default = h
            st: span tail [bsz, seq_len, biaffine_dim]  default = t
            p: relations subject heads [bsz, seq_len, biaffine_dim]
            h: relations object heads [bsz, seq_len, biaffine_dim]
            t:relations object tails [bsz, seq_len, biaffine_dim]
            =>
            s_hs_t, ps_t: [bsz, seq_len, seq_len, seq_len, n_out]
            s_hs_t, ps_h: [bsz, seq_len, seq_len, seq_len, n_out]
            ps_h, ps'_h (ph sibling): [bsz, seq_len, seq_len, seq_len, n_out]
            ps_t, ps'_t (pt sibling): [bsz, seq_len, seq_len, seq_len, n_out]
            ps_h, p's_h (ph co-parent): [bsz, seq_len, seq_len, seq_len, n_out]
            ps_t, p's_t (pt co-parent): [bsz, seq_len, seq_len, seq_len, n_out]

        '''
        # import pdb
        # pdb.set_trace()
        
        # spans
        rh = self.rh_mlp(x)
        rt = self.rt_mlp(x)
        
        rh = self.dropout(rh)
        rt = self.dropout(rt)
        
        fh = self.fh_mlp(x)
        ft = self.ft_mlp(x)
        
        fh = self.dropout(fh)
        ft = self.dropout(ft)

        # prds
        ph = self.ph_mlp(x)
        ph = self.dropout(ph)
        
        pt = self.pt_mlp(x)
        pt = self.dropout(pt)
        
        # arguments heads and arguments tails
        h = self.h_mlp(x)
        h = self.dropout(h)
        
        t = self.t_mlp(x)
        t = self.dropout(t)
        
        # second-order scorers
        
        # pt,rh,rt (co-parent)
        rspan_pst = self.rspan_pt_att(rh, rt, pt)
        # rspan_pst = (rspan_pst.triu() + rspan_pst.triu(1).transpose(-1,-2))
        if len(rspan_pst.shape) > 4:
            rspan_pst = rspan_pst.movedim(1,-1)   # p, s_h, s_t 
        
        # ph,sh,st (grandparent)
        rspan_psh = self.rspan_ph_att(rh, rt, ph)
        # span_psh = (span_psh.triu() + span_psh.triu(1).transpose(-1,-2))
        if len(rspan_psh.shape) > 4:
            rspan_psh = rspan_psh.movedim(1,-1)   # p, s_h, s_t
            
        # ph,pt,rt (gradparent)
        fspan_pst = self.fspan_pt_att(pt, rt, ph)
        # fspan_pst = (fspan_pst.triu() + fspan_pst.triu(1).transpose(-1,-2))
        if len(fspan_pst.shape) > 4:
            fspan_pst = fspan_pst.movedim(1,-1)   # p, s_h, s_t 
        
        # ph,pt,sh (sibling)
        fspan_psh = self.fspan_ph_att(pt, rh, ph)
        # span_psh = (span_psh.triu() + span_psh.triu(1).transpose(-1,-2))
        if len(fspan_psh.shape) > 4:
            fspan_psh = fspan_psh.movedim(1,-1)   # p, s_h, s_t
            
        # sibling ps_h,ps'_h & ps_t,ps'_t suppose s_h < s'_h and s_t < s'_t
        ph_sib = self.ph_sib(rh, rh, ph)
        ph_sib = (ph_sib.triu() + ph_sib.triu(1).transpose(-1,-2))
        if len(ph_sib.shape) > 4:
            ph_sib = ph_sib.movedim(1,-1)   # p, s_h, s_h
            
        pt_sib = self.pt_sib(rt, rt, pt)
        pt_sib = (pt_sib.triu() + pt_sib.triu(1).transpose(-1,-2))
        if len(pt_sib.shape) > 4:
            pt_sib = pt_sib.movedim(1,-1)   # p, s_t, s_t
            
        # co-parent ps_h, p's_h suppose p < p'
        ph_cop = self.ph_cop(ph, ph, rh)  # sh, p, p'
        ph_cop = (ph_cop.triu() + ph_cop.triu(1).transpose(-1,-2))
        if len(ph_cop.shape) > 4:
            ph_cop = ph_cop.movedim(1,-1)
            ph_cop = ph_cop.permute(0, 2, 3, 1, 4) # p, p', s_h
        else:
            ph_cop = ph_cop.permute(0, 2, 3, 1) # p, p', s_h
            
        # co-parent ps_t, p's_t suppose p < p'
        pt_cop = self.pt_cop(pt, pt, rt)
        pt_cop = (pt_cop.triu() + pt_cop.triu(1).transpose(-1,-2))
        if len(pt_cop.shape) > 4:
            pt_cop = pt_cop.movedim(1,-1)
            pt_cop = pt_cop.permute(0, 2, 3, 1, 4) # p, p', s_t
        else:
            pt_cop = pt_cop.permute(0, 2, 3, 1) # p, p', s_t

        return rspan_psh, rspan_pst, fspan_psh, fspan_pst, ph_sib, pt_sib, ph_cop, pt_cop