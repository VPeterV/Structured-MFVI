import torch
from torch import nn
# from .triaffine import Triaffine
from supar.modules import MLP
from .affinelayer import Biaffine, Triaffine
import logging

log = logging.getLogger(__name__)
class SRLFirstOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, n_span_labels, ensemble_label = False, ph_arc = False):
        super(SRLFirstOrderScorer, self).__init__()
        
        self.conf = conf
        self.split = [] # a: arc split l: label split
        if hasattr(conf, "split"):
            self.split = conf.split.split("I")
            log.info(f"Leverageing split {conf.split}")
            
        self.ensemble_label = ensemble_label
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        biaffine_dim = conf.biaffine_dim
        biaffine_label_dim = conf.biaffine_label_dim
        
        self.cat_rel_arc = False
        self.cat_rel_label = False

        self.ph_arc = ph_arc
        
        # if hasattr(conf, 'cat_rel_arc'):
        #     self.cat_rel_arc = conf.cat_rel_arc
        # if hasattr(conf, 'cat_rel_label'):
        #     self.cat_rel_label = conf.cat_rel_label
        
        # spans scorers
        self.sl_mlp = MLP(input_dim, biaffine_dim)  # span heads
        self.sr_mlp = MLP(input_dim, biaffine_dim) # span tails 
        
        # arc scorers
        self.p_mlp = MLP(input_dim, biaffine_dim) # predicates mlp
        self.h_mlp = MLP(input_dim, biaffine_dim) # span heads mlp
        self.t_mlp = MLP(input_dim, biaffine_dim) # tail heads mlp
        
        if 'a' in self.split:
            self.p2_mlp = MLP(input_dim, biaffine_dim)
        
        # label scorers
        self.p_label_mlp = MLP(input_dim, biaffine_label_dim)  # prds labels
        self.h_label_mlp = MLP(input_dim, biaffine_label_dim)  # heads labels
        
        if 'l' in self.split:
            self.p2_label_mlp = MLP(input_dim, biaffine_dim)
        
        self.span_att = Biaffine(biaffine_dim, n_out = span_n_out, init = conf.arc_init)
        self.ph_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        self.pt_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        
        self.ph_labels_att = Biaffine(biaffine_label_dim, n_out = n_span_labels, init = conf.label_init)

        if ensemble_label:
            self.t_label_mlp = MLP(input_dim, biaffine_label_dim)  # heads labels
            self.pt_labels_att = Biaffine(biaffine_label_dim, n_out = n_span_labels, init = conf.label_init)

        if ph_arc:
            self.p_arc_mlp = MLP(input_dim, biaffine_dim)
            self.h_arc_mlp = MLP(input_dim, biaffine_dim)
            self.ph_arc_att = Biaffine(biaffine_dim, n_out = 1, init = conf.arc_init)

        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
                
    def forward(self, word):
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
        sl = self.sl_mlp(word)
        sr = self.sr_mlp(word)
        
        sl = self.arc_dropout(sl)
        sr = self.arc_dropout(sr)
        
        span = self.span_att(sl, sr)
        span = span.triu() + span.triu(1).transpose(-1,-2)
        if self.conf.span_n_out > 1:
            span = span.permute(0,2,3,1)
                
        # arcs
        p = self.p_mlp(word)
        h = self.h_mlp(word)
        t = self.t_mlp(word)
        
        p = self.arc_dropout(p)
        h = self.arc_dropout(h)
        t = self.arc_dropout(t)
        
        if 'a' in self.split:
            p2 = self.p2_mlp(word)
            p2 = self.arc_dropout(p2)
        
        ph = self.ph_att(p, h)
        
        if 'a' in self.split:
            pt = self.pt_att(p2, t)
        else:
            pt = self.pt_att(p, t)
        
        if self.conf.arc_n_out > 1:
            ph = ph.permute(0,2,3,1)
            pt = pt.permute(0,2,3,1)

        # labels
        p_label = self.p_label_mlp(word)
        h_label = self.h_label_mlp(word)
        
        p_label = self.label_dropout(p_label)
        h_label = self.label_dropout(h_label)
        
        ph_label = self.ph_labels_att(p_label, h_label)
        ph_label = ph_label.permute(0,2,3,1)

        if self.ensemble_label:
            t_label = self.t_label_mlp(word)
            t_label = self.label_dropout(t_label)
            if 'l' in self.split:
                p2_label = self.p2_label_mlp(word)
                p2_label = self.label_dropout(p2_label)
                pt_label = self.pt_labels_att(p2_label, t_label)
            else:
                pt_label = self.pt_labels_att(p_label, t_label)
            
            pt_label = pt_label.permute(0,2,3,1)

        if self.ph_arc:
            p_arc = self.p_arc_mlp(word)
            h_arc = self.h_arc_mlp(word)

            p_arc = self.arc_dropout(p_arc)
            h_arc = self.arc_dropout(h_arc)

            ph_arc = self.ph_arc_att(p_arc, h_arc)

            return span, ph, pt, ph_label, ph_arc
    
        if self.ensemble_label:
            # breakpoint()
            return span, ph, pt, ph_label, pt_label
        else:
            return span, ph, pt, ph_label
        
def define_concat(input_dim, linear_dim, dropout, n_out, cse_label = False):
    # l_mlp = MLP(input_dim, biaffine_dim)
    # r_mlp = MLP(input_dim, biaffine_dim)    
    # biaffine = Biaffine(biaffine_dim, n_out = n_out, init = init)
    hidden_dim = input_dim * 2
    
    if cse_label:
        linear = nn.Sequential(
            nn.Linear(hidden_dim, linear_dim),
            nn.Tanh()
            )
    else:
        linear = nn.Sequential(
            nn.Linear(hidden_dim, linear_dim),
            nn.Tanh(),
            nn.Linear(linear_dim, n_out)
            )

    return linear

class FirstOrderConcat(nn.Module):
    def __init__(self, conf, input_dim, n_rel_labels, cse_label = False):
        super(FirstOrderConcat, self).__init__()
        self.conf = conf
        linear_dim = conf.linear_dim
        linear_label_dim = conf.linear_label_dim
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        self.cse_label = cse_label
        # spans scorers
        # self.sh_mlp = MLP(input_dim, biaffine_dim)  # span heads
        # self.st_mlp = MLP(input_dim, biaffine_dim) # span tails 
        # self.sh_mlp = nn.Linear(input_dim, biaffine_dim)  # span heads
        # self.st_mlp = nn.Linear(input_dim, biaffine_dim) # span tails 
        
        # relation scorers
        # self.rel_sh_mlp = MLP(input_dim, biaffine_dim) # relations subject heads
        # self.rel_st_mlp = MLP(input_dim, biaffine_dim) # relations subject heads
        # self.rel_oh_mlp = MLP(input_dim, biaffine_dim) # relations object heads
        # self.rel_ot_mlp = MLP(input_dim, biaffine_dim) # relations object tails
        # self.rel_sh_mlp = nn.Linear(input_dim, biaffine_dim) # relations subject heads
        # self.rel_st_mlp = nn.Linear(input_dim, biaffine_dim) # relations subject heads
        # self.rel_oh_mlp = nn.Linear(input_dim, biaffine_dim) # relations object heads
        # self.rel_ot_mlp = nn.Linear(input_dim, biaffine_dim) # relations object tails
        
        # self.sh_st = Biaffine(biaffine_dim, n_out = n_out, init=conf.arc_init)    # sh_st Existence for Spans
        # self.sh_oh = Biaffine(biaffine_dim, n_out = n_out, init=conf.arc_init)    # Existence for relations sh -> oh (directional)
        # self.st_ot = Biaffine(biaffine_dim, n_out = n_out, init=conf.arc_init) # Existence for relations  sh -> ot (directional)
        
        self.span_linear = define_concat(input_dim, linear_dim, conf.arc_mlp_dropout, span_n_out)
        self.h2h_linear = define_concat(input_dim, linear_dim, conf.arc_mlp_dropout, arc_n_out)
        self.t2t_linear = define_concat(input_dim, linear_dim, conf.arc_mlp_dropout, arc_n_out)
        
        # if cse_label:
        #     self.h2h_label_w = nn.Parameter(torch.randn(linear_dim, n_rel_labels, 3))
        #     self.t2t_label_w = nn.Parameter(torch.randn(linear_dim, n_rel_labels, 3))
            
        #     self.h2h_label_b = nn.Parameter(torch.zeros(n_rel_labels, 3))
        #     self.t2t_label_b = nn.Parameter(torch.zeros(n_rel_labels, 3))

        self.h2h_label_linear = define_concat(input_dim, linear_dim, conf.arc_mlp_dropout, n_rel_labels, cse_label)
        # self.w_h2h = nn.Parameter(torch.randn(linear_dim, n_rel_labels))
        self.t2t_label_linear = define_concat(input_dim, linear_dim, conf.arc_mlp_dropout, n_rel_labels, cse_label)
        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
                
    def forward(self, x):
        '''
            x: hidden states [bsz, seq_len, input_dim (hidden_dim)]
            =>
            sh: span head [bsz, seq_len, biaffine_dim]
            st: span tail [bsz, seq_len, biaffine_dim]
            rel_sh: relations subject heads [bsz, seq_len, biaffine_dim]
            rel_st: relations object heads [bsz, seq_len, biaffine_dim]
            rel_ot:relations object tails [bsz, seq_len, biaffine_dim]
            =>
            shst: span head - span tails [bsz, seq_len, seq_len]
            shoh: span head -> object heads [bsz,seq_len, seq_len]
            shot: span tail -> object tails [bsz,seq_len, seq_len]
        '''
        # import pdb
        # pdb.set_trace()
        
        # spans
        seqlen = x.size(1)
        x_left = x.unsqueeze(-2).expand(-1, -1, seqlen, -1)
        x_right = x.unsqueeze(1).expand(-1, seqlen, -1, -1)
        x_ = torch.cat([x_left, x_right], dim = -1)
        
        shst = self.span_linear(x_)
        shst = shst.squeeze(-1)
        shst = shst.movedim(-1, 1)
        shst = shst.triu() + shst.triu(1).transpose(-1, -2)
        shst = shst.movedim(1, -1)
        shoh = self.h2h_linear(x_)
        shoh = shoh.squeeze(-1)
        stot = self.t2t_linear(x_)
        stot = stot.squeeze(-1)
        
        # spans
        # sh = self.sh_mlp(x)
        # st = self.st_mlp(x)
        
        # sh = self.arc_dropout(sh)
        # st = self.arc_dropout(st)
        
        # shst = self.sh_st(sh, st)
        # shst = shst.triu() + shst.triu(1).transpose(-1,-2)
        # # shst = shst.permute(0,2,3,1)
                
        # # relations
        # rel_sh = self.rel_sh_mlp(x)
        # rel_st = self.rel_st_mlp(x)
        # rel_oh = self.rel_oh_mlp(x)
        # rel_ot = self.rel_ot_mlp(x)
        
        # rel_sh = self.arc_dropout(rel_sh)
        # rel_st = self.arc_dropout(rel_st)
        # rel_oh = self.arc_dropout(rel_oh)
        # rel_ot = self.arc_dropout(rel_ot)
        
        # shoh = self.sh_oh(rel_sh, rel_oh)
        # stot = self.st_ot(rel_st, rel_ot)
        
        shoh_labels = self.h2h_label_linear(x_)
        # stot_labels = self.t2t_label_linear(x_)
        
        # if self.cse_label:
        #     shoh_labels = contract('bijh, hrk -> brijk', shoh_labels, self.h2h_label_w)
        #     shoh_labels += self.h2h_label_b[None, :, None, None]
            
        #     stot_labels = contract('bijh, hrk -> brijk', stot_labels, self.t2t_label_w)
        #     stot_labels += self.t2t_label_b[None, :, None, None]

        return shst, shoh, stot, shoh_labels


class SRLSecondOrderScorer(nn.Module):
    def __init__(self, conf, input_dim):
        super(SRLSecondOrderScorer, self).__init__()
        self.conf = conf
        n_out = conf.n_out
        decompose = conf.decompose
        ind_prd = conf.ind_prd if hasattr(conf, 'ind_prd') else False
        shared_span = conf.shared_span
        triaffine_dim = conf.triaffine_dim
        
        self.split = True
        if hasattr(conf, "split"):
            self.split = conf.split
        
        if self.split:
            self.p_mlp = MLP(input_dim, triaffine_dim)
            self.p2_mlp =  MLP(input_dim, triaffine_dim)
        else:
            self.p_mlp = MLP(input_dim, triaffine_dim)
            # self.p2_mlp = self.p_mlp
            
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
        
        if self.split:
            p2 = self.p2_mlp(x)
            p2 = self.dropout(p2)
        else:
            p2 = p
        
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
            ph_cop = ph_cop.permute(0, 2, 3, 1, 4) # p, p', s_h
        else:
            ph_cop = ph_cop.permute(0, 2, 3, 1) # p, p', s_h
            
        # co-parent ps_t, p's_t suppose p < p'
        pt_cop = self.pt_cop(p2, p2, st)
        pt_cop = (pt_cop.triu() + pt_cop.triu(1).transpose(-1,-2))
        if len(pt_cop.shape) > 4:
            pt_cop = pt_cop.movedim(1,-1)
            pt_cop = pt_cop.permute(0, 2, 3, 1, 4) # p, p', s_t
        else:
            pt_cop = pt_cop.permute(0, 2, 3, 1) # p, p', s_t

        return span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop