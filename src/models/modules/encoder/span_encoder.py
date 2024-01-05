import torch
from torch import nn
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor

class SpanEncoder(nn.Module):
    """
    Support three types of span encoding:
        1. Span encoder using coherent encoding (SEO et al. 2019)
        2. Concat
        3. span encoder coded by allennlp (Lin et al. 2021)
    """
    def __init__(self, conf, hidden_dim, emb_dim, concat_emb = True):
        super(SpanEncoder, self).__init__()
        self.conf = conf
        self.encoding = conf.encoding
        self.x_mlp = conf.x_mlp
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.concat_emb = concat_emb
        assert self.x_mlp or self.encoding == 'concat' or self.encoding == 'endpoint'
        
        if self.encoding == 'coherent':
            self.a_dim = conf.a_dim
            self.b_dim = conf.b_dim
            self.d_dim = 2 * (self.a_dim +self.b_dim)
        elif self.encoding == 'concat':
            self.d_dim  = conf.d_dim
        elif self.encoding == 'endpoint':
            self.d_dim  = conf.d_dim
            self.width_emb_dim = conf.width_emb_dim
            self.endpoint_extractor = EndpointSpanExtractor(input_dim = hidden_dim, 
                                        combination="x,y", 
                                        num_width_embeddings=300,
                                        span_width_embedding_dim=self.width_emb_dim,
                                        bucket_widths=False)
            self.attentive_extractor = SelfAttentiveSpanExtractor(input_dim = emb_dim)
        else:
            raise NotImplementedError
        
        self.left_mlp = nn.Linear(hidden_dim, self.d_dim)
        self.right_mlp = nn.Linear(hidden_dim, self.d_dim)
        
    def get_span_dim(self):
        if self.encoding == 'coherent':
            return 2 * self.a_dim + 1
        elif self.encoding == 'concat':
            return 2 * self.d_dim
        else:
            if self.concat_emb:
                return 2 * self.hidden_dim + self.emb_dim + self.width_emb_dim
            else:
                return 2 * self.hidden_dim + self.width_emb_dim
        
    def forward(self, inputs, span, emb_inputs = None):
        
        bsz = inputs.size(0)
        
        xl = inputs[torch.arange(bsz)[:, None], span[..., 0]]
        xr = inputs[torch.arange(bsz)[:, None], span[..., 1]]
        
        if self.x_mlp:
            l = self.left_mlp(xl)
            r = self.right_mlp(xr)
        else:
            l, r = xl, xr
        
        if self.encoding == 'coherent':
            a1 = l[..., :self.a_dim]
            a2 = r[..., self.a_dim: 2 * self.a_dim]
            
            a3 = l[..., 2 * self.a_dim : -self.b_dim]
            a4 = r[..., -self.b_dim : ]
            
            # a34 = torch.matmul(a3, a4)
            # breakpoint()
            a34 = torch.einsum('bmh, bmh -> bm', a3, a4)
            a34 = a34.unsqueeze(-1)
            
            a = torch.cat([a1, a2, a34], dim = -1)
        elif self.encoding == 'concat':
            a = torch.concat([l, r], dim = -1)
        else:
            t = inputs.size(1)
            input_span = torch.where(span == -1, t - 1, span)
            endpoint = self.endpoint_extractor(inputs, input_span)
            if self.concat_emb and emb_inputs is not None:
                attentiive = self.attentive_extractor(emb_inputs, input_span)
                a = torch.cat([endpoint, attentiive], dim = -1)
            else:
                a = endpoint
        
        return a
            
        
        
        