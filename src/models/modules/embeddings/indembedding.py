import pdb
import torch
from torch import nn
from ..encoder.char_lstm_encoder import CharLSTM

class IndEmbedding(nn.Module):
    def __init__(self, emb_conf) -> None:
        super().__init__()
        self.emb_conf = emb_conf
        
        self.ind_emb = nn.Embedding(2, emb_conf.ind_emb)
        
    def forward(self, x_table) -> torch.Tensor:
    
        ind_embs = self.ind_emb(x_table['ind'])
        
        return ind_embs