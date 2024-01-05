import torch
from torch import nn
from supar.modules import MLP
from .affinelayer import Biaffine

class SpanScorer(nn.Module):
    def __init__(self, conf, hidden_dim, span_dim, n_out):
        
        super().__init__()

        self.n_out = n_out
        self.span_affine_dim = conf.span_affine_dim
        
        self.span_scorer_mlp = True if self.span_affine_dim > 0 else False
        if self.span_scorer_mlp:
            self.p_mlp = MLP(hidden_dim, self.span_affine_dim)
            self.span_mlp = MLP(span_dim, self.span_affine_dim)
        else:
            self.span_affine_dim = hidden_dim
        
        self.scorer = Biaffine(self.span_affine_dim, n_out)
        
        self.dropout = nn.Dropout(conf.span_dropout)
        
    def forward(self, x, y):
        """
        x: token representations (predicates)
        y: span representations (spans)
        """
        
        if self.span_scorer_mlp:
            # breakpoint()

            p = self.p_mlp(x)
            span = self.span_mlp(y)
            
            p = self.dropout(p)
            span = self.dropout(span)
        else:
            p = x
            span = y
            
        repr = self.scorer(p, span)
        
        if self.n_out > 1:
            repr = repr.permute(0,2,3,1)
            
        return repr
        
class FrameNetSpanScorer(nn.Module):
    def __init__(self, conf, hidden_dim, n_out):
        super(FrameNetSpanScorer, self).__init__()
        
        self.label_linear = nn.Sequential(
        nn.Linear(hidden_dim, conf.linear_dim),
        nn.ReLU(),
        nn.Dropout(conf.dropout),
        nn.Linear(conf.linear_dim, n_out)
        )
        
    def forward(self, x):
        return self.label_linear(x)
        
class FrameNetRelationScorer(nn.Module):
    def __init__(self, conf, hidden_dim, n_out):
        super(FrameNetRelationScorer, self).__init__()
        
        self.label_linear = nn.Sequential(
        nn.Linear(hidden_dim * 3, conf.linear_dim),
        nn.ReLU(),
        nn.Dropout(conf.dropout),
        nn.Linear(conf.linear_dim, n_out)
        )        
        
    def forward(self, x, y):
        assert x.size() == y.size()
        
        relation_repr = torch.cat([x, y, x * y], dim = -1)
        
        return self.label_linear(relation_repr)
        
        