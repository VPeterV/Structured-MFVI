import torch
from torch import nn
from supar.modules import LSTM, SharedDropout, IndependentDropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEncoder(nn.Module):
    def __init__(self, conf, input_dim):
        self.conf = conf
        super(LSTMEncoder, self).__init__()
        
        if conf.emb_dropout_type == 'independent':
            self.emb_dropout = IndependentDropout(p=conf.before_lstm_dropout)
            
        elif conf.emb_dropout_type == 'shared':
            self.emb_dropout = SharedDropout(p=conf.before_lstm_dropout)
            
        elif conf.emb_dropout_type == 'vanilla':
            self.emb_dropout = nn.Dropout(p=conf.before_lstm_dropout)
            
        else:
            self.emb_dropout = nn.Dropout(p=0.)
            
        self.lstm = LSTM(input_size=input_dim,
                 hidden_size=conf.n_lstm_hidden,
                 num_layers=conf.n_lstm_layers,
                 bidirectional=conf.bilstm,
                 dropout=conf.lstm_dropout)
                 
        if conf.lstm_dropout_type == 'independent':
            self.lstm_dropout = IndependentDropout(p=conf.lstm_dropout)
            
        elif conf.lstm_dropout_type == 'shared':
            self.lstm_dropout = SharedDropout(p=conf.lstm_dropout)
            
        elif conf.lstm_dropout_type == 'vanilla':
            self.lstm_dropout = nn.Dropout(p=conf.lstm_dropout)
            
        else:
            self.lstm_dropout = nn.Dropout(p=0.)
        
    def forward(self, x, seq_lens):
        
        embs = x
        
        if self.conf.emb_dropout_type == 'independent':
            embs = [emb for emb in x.values()]
            embs = torch.cat(embs, dim=-1)
        else:
            embs = self.emb_dropout(embs)
        
        seq_lens = seq_lens.cpu()
        
        x = pack_padded_sequence(embs, seq_lens, True, False)   # why seq_len + pad_len for pointer?
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=embs.shape[1])
        x = self.lstm_dropout(x)

        return x