import torch
from torch import nn
# from .triaffine import Triaffine
from supar.modules import MLP
from .affinelayer import Biaffine, Triaffine

class SRLFirstOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, n_span_labels):
        super(SRLFirstOrderScorer, self).__init__()
        
        self.conf = conf
        span_n_out = conf.span_n_out
        arc_n_out = conf.arc_n_out
        biaffine_dim = conf.biaffine_dim
        biaffine_label_dim = conf.biaffine_label_dim
        # spans scorers
        self.sl_mlp = MLP(input_dim, biaffine_dim)  # span heads
        self.sr_mlp = MLP(input_dim, biaffine_dim) # span tails 
        
        # arc scorers
        self.p_mlp = MLP(input_dim, biaffine_dim) # predicates mlp
        self.h_mlp = MLP(input_dim, biaffine_dim) # span heads mlp
        self.t_mlp = MLP(input_dim, biaffine_dim) # tail heads mlp
        
        # label scorers
        self.sl_label_mlp = MLP(input_dim, biaffine_label_dim)  # span heads labels
        self.sr_label_mlp = MLP(input_dim, biaffine_label_dim)  # span tails labels
        
        self.span_att = Biaffine(biaffine_dim, n_out = span_n_out, init = conf.arc_init)
        self.ph_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        self.pt_att = Biaffine(biaffine_dim, n_out = arc_n_out, init = conf.arc_init)  # bce loss
        
        self.span_labels_att = Biaffine(biaffine_label_dim, n_out = n_span_labels, init = conf.label_init)
        
        self.arc_dropout = nn.Dropout(conf.arc_mlp_dropout)
        self.label_dropout = nn.Dropout(conf.label_mlp_dropout)
                
    def forward(self, x):
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
        sl = self.sl_mlp(x)
        sr = self.sr_mlp(x)
        
        sl = self.arc_dropout(sl)
        sr = self.arc_dropout(sr)
        
        span = self.span_att(sl, sr)
        span = span.triu() + span.triu(1).transpose(-1,-2)
        if self.conf.span_n_out > 1:
            span = span.permute(0,2,3,1)
                
        # arcs
        p = self.sl_mlp(x)
        h = self.sl_mlp(x)
        t = self.sr_mlp(x)
        
        p = self.arc_dropout(p)
        h = self.arc_dropout(h)
        t = self.arc_dropout(t)
        
        ph = self.ph_att(p, h)
        pt = self.pt_att(p, t)
        
        if self.conf.arc_n_out > 1:
            ph = ph.permute(0,2,3,1)
            pt = pt.permute(0,2,3,1)

        # labels
        sl_label = self.sl_label_mlp(x)
        sr_label = self.sr_label_mlp(x)
        
        sl_label = self.label_dropout(sl_label)
        sr_label = self.label_dropout(sr_label)
        
        span_label = self.span_labels_att(sl_label, sr_label)
        span_label = span_label.permute(0,2,3,1)

        return span, ph, pt, span_label
        
        
class SecondOrderScorer(nn.Module):
    def __init__(self, conf, input_dim, shared_mlp = None):
        super(SecondOrderScorer, self).__init__()
        self.conf = conf
        n_out = conf.n_out
        triaffine_dim = conf.triaffine_dim
        shared = shared_mlp is not None
        if shared and 'sh_mlp' in shared_mlp:
            self.sh_mlp = shared_mlp['sh_mlp']
        else:
            # self.sh_mlp = MLP(input_dim, triaffine_dim)
            self.sh_mlp = nn.Linear(input_dim, triaffine_dim)
            
        
        if shared and 'st_mlp' in shared_mlp:
            self.sh_mlp = shared_mlp['st_mlp']
        else:
            # self.st_mlp = MLP(input_dim, triaffine_dim)
            self.st_mlp = nn.Linear(input_dim, triaffine_dim)
            
        if shared and 'rel_sh_mlp' in shared_mlp:
            self.rel_sh_mlp = shared_mlp['rel_sh_mlp']
        else:
            # self.rel_sh_mlp = MLP(input_dim, triaffine_dim)
            self.rel_sh_mlp = nn.Linear(input_dim, triaffine_dim)
            
        if shared and 'rel_st_mlp' in shared_mlp:
            self.rel_st_mlp = shared_mlp['rel_st_mlp']
        else:
            # self.rel_st_mlp = MLP(input_dim, triaffine_dim)
            self.rel_st_mlp = nn.Linear(input_dim, triaffine_dim)
            
        if shared and 'rel_oh_mlp' in shared_mlp:
            self.rel_oh_mlp = shared_mlp['rel_oh_mlp']
        else:
            # self.rel_oh_mlp = MLP(input_dim, triaffine_dim)
            self.rel_oh_mlp = nn.Linear(input_dim, triaffine_dim)
        
        if shared and 'rel_ot_mlp' in shared_mlp:
            self.rel_ot_mlp = shared_mlp['rel_ot_mlp']
        else:
            # self.rel_ot_mlp = MLP(input_dim, triaffine_dim)
            self.rel_ot_mlp = nn.Linear(input_dim, triaffine_dim)

        self.shstoh = Triaffine(triaffine_dim, n_out, init=conf.init)
        self.shstot = Triaffine(triaffine_dim, n_out, init=conf.init)
        
        self.dropout = nn.Dropout(conf.mlp_dropout)
        
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
            shstoh: span head - span tails + span head -> object heads [bsz, seq_len, seq_len, seq_len, n_out]
            shstot: span head -> span tails + span head -> object tails [bsz, seq_len, seq_len, seq_len, n_out]
            # shohot: span tail -> object heads + span head -> object tails [bsz, seq_len, seq_len, seq_len, n_out]
        '''
        # import pdb
        # pdb.set_trace()
        
        # spans
        sh = self.sh_mlp(x)
        st = self.st_mlp(x)
        
        sh = self.dropout(sh)
        st = self.dropout(st)
        # relations
        # rel_sh = self.rel_sh_mlp(x)
        # rel_st = self.rel_st_mlp(x)
        rel_oh = self.rel_oh_mlp(x)
        rel_ot = self.rel_ot_mlp(x)
        
        rel_oh = self.dropout(rel_oh)
        rel_ot = self.dropout(rel_ot)
        
        # second-order scorers
        #TODO Try to share spans' represntation with relation representations here
        shstoh = self.shstoh(st,rel_oh,sh)  # sh, st, rel_oh
        shstoh = shstoh.permute(0, 4, 3, 1, 2).triu()
        shstoh = shstoh.permute(0, 3, 4, 2 , 1)
        shstot = self.shstot(rel_ot,st, sh) # sh rel_ot st
        shstot = shstot.permute(0, 4, 2, 1, 3).triu()
        shstot = shstot.permute(0, 3, 4, 2, 1)
        
        return shstoh.log_softmax(-1), shstot.log_softmax(-1)