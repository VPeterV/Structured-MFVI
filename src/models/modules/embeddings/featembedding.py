import pdb
import torch
from torch import nn
from ..encoder.char_lstm_encoder import CharLSTM

class FeatEmbedding(nn.Module):
    def __init__(self, emb_conf, fields) -> None:
        super().__init__()
        n_feats = fields.get_n_feats
        char_pad = fields.get_char_pad
        self.emb_conf = emb_conf
        assert 'char' in emb_conf.feats or 'lemma' in emb_conf.feats
        
        n_input = 0
        if 'char' in emb_conf.feats:
            self.char_embed = CharLSTM(n_chars = n_feats['char'],
                                        n_embed = emb_conf.n_char_emb,
                                        n_hidden = emb_conf.n_char_hidden,
                                        n_out = emb_conf.n_char_feat_embed,
                                        pad_index = char_pad,
                                        dropout = emb_conf.char_dropout)
            n_input += emb_conf.n_char_feat_embed
        
        if 'lemma' in emb_conf.feats:
            self.lemma_emb = nn.Embedding(num_embeddings = n_feats['lemma'],
                                            embedding_dim = emb_conf.lemma_emb)
            n_input += emb_conf.lemma_emb
            
        self.n_input = n_input
        
    @property
    def n_out_emb(self):
        return self.n_input
        
    def load_pretrained(self, embed = None):
        if embed is not None:
            self.pretrianed = nn.Embedding.from_pretrained(embed.to)
        
    def forward(self, x_table) -> torch.Tensor:
        
        feat_embs = []
        if 'char' in self.emb_conf.feats:
            feat_embs.append(self.char_embed(x_table['char']))
        if 'lemma' in self.emb_conf.feats:
            feat_embs.append(self.lemma_emb(x_table['lemma']))
        
        feat_embs = torch.cat(feat_embs, -1)
        embs = torch.cat(feat_embs, dim = -1)
        
        return embs