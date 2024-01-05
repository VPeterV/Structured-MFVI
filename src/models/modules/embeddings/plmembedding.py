from torch import nn
from supar.modules.pretrained import TransformerEmbedding

class Embedding(nn.Module):
    def __init__(self, conf, fields):
        super(Embedding, self).__init__()
        self.conf = conf
    
        if 'bert' in fields.inputs:
            self.embedding = TransformerEmbedding(model=fields.get_bert_name(),
                                                    n_layers=conf.n_bert_layers,
                                                    n_out = conf.n_bert_out,
                                                    pad_index = fields.get_pad_index("bert"),
                                                    mix_dropout=conf.mix_dropout,
                                                    finetune=conf.finetune,
                                                    pooling=conf.pooling)
            print("PLM model:", end='\t')
            print(fields.get_bert_name())
            
    def forward(self, x):
        
        return self.embedding(x)