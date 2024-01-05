import torch
from torch import nn
# from .triaffine import Triaffine
from supar.modules import MLP
from .affinelayer import Biaffine, Triaffine, apply_fencepost

class FirstOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, n_span_labels):
        super(FirstOrderScorer, self).__init__()
        
        self.conf = conf
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        biaffine_dim = conf.biaffine_dim
        biaffine_label_dim = conf.biaffine_label_dim
        # spans scorers
        self.sl_mlp = MLP(input_dim, biaffine_dim)  # span heads
        self.sr_mlp = MLP(input_dim, biaffine_dim) # span tails 
        
        # arc scorers
        self.eh_mlp = MLP(input_dim, biaffine_dim) # expressiong/predicate head mlp
        self.et_mlp = MLP(input_dim, biaffine_dim) # expressiong/predicate tail mlp    
        
        self.sh_mlp = MLP(input_dim, biaffine_dim) # roles head mlp
        self.st_mlp = MLP(input_dim, biaffine_dim) # roles tail mlp
        
        # label scorers
        
        self.eh_label_mlp = MLP(input_dim, biaffine_label_dim)  # predicate head mlp for labeling
        self.et_label_mlp = MLP(input_dim, biaffine_label_dim)  # predicate tail mlp for labeling
        
        self.sh_label_mlp = MLP(input_dim, biaffine_label_dim)  # roles tail mlp for labeling
        self.st_label_mlp = MLP(input_dim, biaffine_label_dim)  # roles tail mlp for labeling
        
        self.span_att = Biaffine(biaffine_dim, n_out = span_n_out, init = conf.arc_init) # bce loss
        self.h2h_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        self.t2t_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        
        self.h2h_labels_att = Biaffine(biaffine_label_dim, n_out = n_span_labels, init = conf.label_init)
        self.t2t_labels_att = Biaffine(biaffine_label_dim, n_out = n_span_labels, init = conf.label_init)
        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
                
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
        
        sl = self.sl_mlp(boundary)
        sr = self.sr_mlp(boundary)
        
        sl = self.arc_dropout(sl)
        sr = self.arc_dropout(sr)
        
        span = self.span_att(sl, sr)
        span = span.triu() + span.triu(1).transpose(-1,-2)
        if self.conf.span_n_out > 1:
            span = span.permute(0,2,3,1)
                
        # arcs
        eh = self.eh_mlp(word)
        et = self.et_mlp(word)
        
        sh = self.sh_mlp(word)
        st = self.st_mlp(word)
        
        eh = self.arc_dropout(eh)
        et = self.arc_dropout(et)
        
        sh = self.arc_dropout(sh)
        st = self.arc_dropout(st)
        
        h2h = self.h2h_att(eh, sh)
        t2t = self.t2t_att(et, st)
        
        if self.conf.arc_n_out > 1:
            h2h = h2h.permute(0,2,3,1)
            t2t = t2t.permute(0,2,3,1)

        # labels
        eh_label = self.eh_label_mlp(word)
        et_label = self.et_label_mlp(word)
        
        sh_label = self.sh_label_mlp(word)
        st_label = self.st_label_mlp(word)
        
        eh_label = self.label_dropout(eh_label)
        et_label = self.label_dropout(et_label)
        
        sh_label = self.label_dropout(sh_label)
        st_label = self.label_dropout(st_label)
        
        h2h_label = self.h2h_labels_att(eh_label, sh_label)
        t2t_label = self.t2t_labels_att(et_label, st_label)
        
        h2h_label = h2h_label.permute(0,2,3,1)
        t2t_label = t2t_label.permute(0,2,3,1)

        return span, h2h, t2t, h2h_label, t2t_label
        
        
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