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

def define_concat(input_dim, linear_dim, dropout, n_out):
    # l_mlp = MLP(input_dim, biaffine_dim)
    # r_mlp = MLP(input_dim, biaffine_dim)    
    # biaffine = Biaffine(biaffine_dim, n_out = n_out, init = init)
    hidden_dim = input_dim * 2
    linear = nn.Sequential(
        nn.Linear(hidden_dim, linear_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(linear_dim, n_out)
        )

    return linear

class FirstOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, n_span_labels: List, score_label: bool = True):
        super(FirstOrderScorer, self).__init__()
        
        self.conf = conf
        # breakpoint()
        self.scoring_type = conf.scoring_type if hasattr(conf, "scoring_type") else "affine"
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        biaffine_dim = conf.biaffine_dim
        biaffine_label_dim = conf.biaffine_label_dim
        
        self.score_label = score_label
        
        num_frame, num_role_labels = n_span_labels
        
        if self.scoring_type == 'affine':
            # spans scorers
            self.sl_mlp, self.sr_mlp, self.span_att = define_affine(input_dim, biaffine_dim, span_n_out, conf.arc_init)
            
            # arcs scorers
            self.ph_mlp, self.rh_mlp, self.h2h_att = define_affine(input_dim, biaffine_dim, arc_n_out, conf.arc_init)
            self.pt_mlp, self.rt_mlp, self.t2t_att = define_affine(input_dim, biaffine_dim, arc_n_out, conf.arc_init)
        elif self.scoring_type == 'concat':
            # spans scorers
            self.span_linear = define_concat(input_dim, conf.linear_dim, conf.linear_dropout, span_n_out)
            
            # arcs scorers
            self.h2h_linear = define_concat(input_dim, conf.linear_dim, conf.linear_dropout, arc_n_out)
            self.t2t_linear = define_concat(input_dim, conf.linear_dim, conf.linear_dropout, arc_n_out)
        else:
            raise NotImplementedError
        # label scorers
        # since 2-class multi-label classification is not too difficult, I guess :)
        # self.sl_type_mlp, self.sr_type_mlp, self.span_type_att = define_affine(input_dim, biaffine_label_dim, 2, conf.label_init)
        
        if score_label:
            # frames num
            self.sl_frame_mlp, self.sr_frame_mlp, self.frame_att = define_affine(input_dim, biaffine_label_dim, num_frame, conf.label_init)
            # roles num
            self.ph_label_mlp, self.rh_label_mlp, self.h2h_label_att = define_affine(input_dim, biaffine_label_dim, num_role_labels, conf.label_init)
            self.pt_label_mlp, self.rt_label_mlp, self.t2t_label_att = define_affine(input_dim, biaffine_label_dim, num_role_labels, conf.label_init)
        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
        
    def affine_pipeline(self, x, l_mlp, r_mlp, dropout, affine):
        l = l_mlp(x)
        r = r_mlp(x)
        
        l = dropout(l)
        r = dropout(r)
        
        res = affine(l, r)
        
        return res

    def concat_pipeline(self, x, linear, dropout):

        seqlen = x.size(-2)
        x_left = x.unsqueeze(-2).expand(-1, -1, seqlen, -1)
        x_right = x.unsqueeze(1).expand(-1, seqlen, -1, -1)

        x_ = torch.cat([x_left, x_right], dim = -1)

        res = linear(x_)
        
        return res.squeeze(-1)
                
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
        
        if self.scoring_type == 'affine':
            span = self.affine_pipeline(boundary, self.sl_mlp, self.sr_mlp, self.arc_dropout,self.span_att)
            span = span.triu() + span.triu(1).transpose(-1,-2)
            if self.conf.span_n_out > 1:
                span = span.permute(0,2,3,1)
        elif self.scoring_type == 'concat':
            # breakpoint()
            span = self.concat_pipeline(boundary, self.span_linear, self.arc_dropout)
            if self.conf.span_n_out > 1:
                span = span.permute(0, 3, 1, 2)
                span = span.triu() + span.triu(1).transpose(-1,-2)
                span = span.permute(0, 2, 3, 1)
            else:
                span = span.triu() + span.triu(1).transpose(-1,-2)

        if self.scoring_type == 'affine':
            # arcs
            h2h = self.affine_pipeline(word, self.ph_mlp, self.rh_mlp, self.arc_dropout, self.h2h_att)
            t2t = self.affine_pipeline(word, self.pt_mlp, self.rt_mlp, self.arc_dropout, self.t2t_att)
            
            if self.conf.arc_n_out > 1:
                h2h = h2h.permute(0,2,3,1)
                t2t = t2t.permute(0,2,3,1)
        elif self.scoring_type == 'concat':
            h2h = self.concat_pipeline(boundary, self.h2h_linear, self.arc_dropout)
            t2t = self.concat_pipeline(boundary, self.t2t_linear, self.arc_dropout)
        
        if self.score_label:
            # labels
            h2h_label = self.affine_pipeline(word, self.ph_label_mlp, self.rh_label_mlp, self.label_dropout, self.h2h_label_att)
            t2t_label = self.affine_pipeline(word, self.pt_label_mlp, self.rt_label_mlp, self.label_dropout, self.t2t_label_att)
            
            h2h_label = h2h_label.permute(0,2,3,1)
            t2t_label = t2t_label.permute(0,2,3,1)
            
            # span_type = self.affine_pipeline(word, self.sl_type_mlp, self.sr_type_mlp, self.label_dropout, self.span_type_att)
            # span_type = span_type.permute(0,2,3,1)
            
            frames = self.affine_pipeline(word, self.sl_frame_mlp, self.sr_frame_mlp, self.label_dropout, self.frame_att)
            frames = frames.permute(0,2,3,1)
        
            return span, h2h, t2t, frames, h2h_label, t2t_label
        else:
            return span, h2h, t2t
        
class SRLSecondOrderScorer(nn.Module):
    def __init__(self, conf, input_dim):
        super(SRLSecondOrderScorer, self).__init__()
        self.conf = conf
        n_out = conf.n_out
        decompose = conf.decompose
        ind_prd = conf.ind_prd if hasattr(conf, 'ind_prd') else False
        shared_span = conf.shared_span
        triaffine_dim = conf.triaffine_dim
        
        if ind_prd:
            self.p_mlp = MLP(input_dim, triaffine_dim)
            self.p2_mlp =  MLP(input_dim, triaffine_dim)
        else:
            self.p_mlp = MLP(input_dim, triaffine_dim)
            self.p2_mlp = self.p_mlp
            
        self.h_mlp = MLP(input_dim, triaffine_dim)
        self.t_mlp = MLP(input_dim, triaffine_dim)
        
        if shared_span:
            self.sh_mlp = self.h_mlp
            self.st_mlp = self.t_mlp
        else:
            self.sh_mlp = MLP(input_dim, triaffine_dim)
            self.st_mlp = MLP(input_dim, triaffine_dim)
        
        # s_hs_t, ps_t
        self.span_pt_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
        # s_hs_t, ps_h
        self.span_ph_att = Triaffine(triaffine_dim, n_out, decompose=decompose, init=conf.init)
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
        sh = self.sh_mlp(x)
        st = self.st_mlp(x)
        
        sh = self.dropout(sh)
        st = self.dropout(st)

        # prds
        p = self.p_mlp(x)
        p = self.dropout(p)
        
        p2 = self.p2_mlp(x)
        p2 = self.dropout(p2)
        
        # arguments heads and arguments tails
        h = self.h_mlp(x)
        h = self.dropout(h)
        
        t = self.t_mlp(x)
        t = self.dropout(t)
        
        # second-order scorers
        
        # p,sh,st (co-parent)
        span_pst = self.span_pt_att(sh, st, p2)
        span_pst = (span_pst.triu() + span_pst.triu(1).transpose(-1,-2))
        if len(span_pst.shape) > 4:
            span_pst = span_pst.movedim(1,-1)   # p, s_h, s_t 
        
        # p,sh,st (grandparent)
        span_psh = self.span_ph_att(sh, st, p)
        # span_psh = (span_psh.triu() + span_psh.triu(1).transpose(-1,-2))
        if len(span_psh.shape) > 4:
            span_psh = span_psh.movedim(1,-1)   # p, s_h, s_t
            
        # sibling ps_h,ps'_h & ps_t,ps'_t suppose s_h < s'_h and s_t < s'_t
        ph_sib = self.ph_sib(sh, sh, p)
        ph_sib = (ph_sib.triu() + ph_sib.triu(1).transpose(-1,-2))
        if len(ph_sib.shape) > 4:
            ph_sib = ph_sib.movedim(1,-1)   # p, s_h, s_h
            
        pt_sib = self.pt_sib(st, st, p2)
        pt_sib = (pt_sib.triu() + pt_sib.triu(1).transpose(-1,-2))
        if len(pt_sib.shape) > 4:
            pt_sib = pt_sib.movedim(1,-1)   # p, s_t, s_t
            
        # co-parent ps_h, p's_h suppose p < p'
        ph_cop = self.ph_cop(p, p, sh)  # sh, p, p'
        ph_cop = (ph_cop.triu() + ph_cop.triu(1).transpose(-1,-2))
        if len(ph_cop.shape) > 4:
            ph_cop = ph_cop.movedim(1,-1)
        ph_cop = ph_cop.permute(0, 2, 3, 1) # p, p', s_h
            
        # co-parent ps_t, p's_t suppose p < p'
        pt_cop = self.pt_cop(p2, p2, st)
        pt_cop = (pt_cop.triu() + pt_cop.triu(1).transpose(-1,-2))
        if len(pt_cop.shape) > 4:
            pt_cop = pt_cop.movedim(1,-1)
        pt_cop = pt_cop.permute(0, 2, 3, 1) # p, p', s_t

        return span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop