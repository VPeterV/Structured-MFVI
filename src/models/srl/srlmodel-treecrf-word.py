import copy
import logging
import pdb
from functools import partial
from typing import List
import torch
from torch import nn
from .utils import unfold_arc_label, potential_norm, unfold_graphs
from .modules.loss.treecrf_loss import TreecrfLoss
from .modules.loss.label_loss import PhLabelLoss
from .modules.structs.cyk import cyk
from .modules.encoder.lstm_encoder import LSTMEncoder
from .modules.encoder.var_lstm_encoder import VariationalLSTM
from .modules.embeddings.indembedding import IndEmbedding
from .modules.embeddings.plmembedding import Embedding
from .modules.embeddings.w2vembedding import W2VEmbedding
from .modules.scorer.affine_fp_word import SRLFirstOrderScorer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import logging


log = logging.getLogger(__name__)

class SRLModel(nn.Module):
    def __init__(self, conf, fields):
        super(SRLModel, self).__init__()
        
        self.conf = conf
        self.fields = fields
        self.num_span_label = fields.get_span_label_num()
        self.is_lstm = True if conf.encoder.n_lstm_layers > 0 else False

        if hasattr(conf.embeddings, 'static_emb') and conf.embeddings.static_emb:
            self.embedding = W2VEmbedding(conf.embeddings, fields)
            input_for_lstm = self.embedding.n_out_emb
        else:
            self.embedding = Embedding(conf.embeddings, fields)
            input_for_lstm = conf.embeddings.n_bert_out
        
        if conf.task.wprd and conf.embeddings.ind_emb > 0:
            self.ind_embedding = IndEmbedding(conf.embeddings)
            input_for_lstm += conf.embeddings.ind_emb
                        
        if self.is_lstm:
            if hasattr(conf.encoder, 'var_lstm') and conf.encoder.var_lstm:
                self.encoder = VariationalLSTM(conf.encoder, input_for_lstm)
            else:
                self.encoder = LSTMEncoder(conf.encoder, input_for_lstm)
        
            self.affine_input = conf.encoder.n_lstm_hidden * 2 * conf.encoder.bilstm
        else:
            self.affine_input = conf.embeddings.n_bert_out
            
            if conf.task.wprd and conf.embeddings.ind_emb > 0:
                self.affine_input += conf.embeddings.ind_emb
            
            if hasattr(conf.embeddings, 'static_emb') and conf.embeddings.static_emb:
                raise NotImplementedError

        self.bert_dropout = nn.Dropout(conf.encoder.bert_dropout)
        self.firstorder = SRLFirstOrderScorer(conf.foscorer, self.affine_input, self.num_span_label)
        
        self.cyk = partial(cyk, r_closed = True)
        
        self.treecrf_loss = TreecrfLoss(conf.loss.span_lamb, conf.treecrf.potential_norm, 
                                        conf.treecrf.structure_smoothing)
                                        
        self.label_loss = PhLabelLoss(self.num_span_label)
        
    def forward(self, x_table, y_table,  inference = False):
    
        seq_lens = x_table['seq_len']
        # pdb.set_trace()
        if 'bert' in x_table:
            x = x_table['bert']
            seq_lens += 2   # not fencepost but including [SEP] temporarily
            emb = self.embedding(x)
        else:
            x = x_table['words']
            emb = self.embedding(x_table)
        
        if self.conf.task.wprd:
            indemb = self.ind_embedding(x_table)
            emb = torch.concat([emb, indemb], dim = -1)
        
        t = x.size(1)
        span_mask = x_table['span_mask']        # True: crossing False: accept
        
        gold_spans = y_table['gold_spans']      # idx starts from 1, but [cls] has been added
        span_labels = y_table['span_labels']    # idx starts from 1, but [cls] has been added
        predicates = y_table['predicates']      # idx starts from 1, but [cls] has been added
        
        span_mask = span_mask.to(torch.bool)
        
        ph_labels_ind = unfold_arc_label(predicates, gold_spans[...,0], span_labels, t)
        
        fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop = \
                        unfold_graphs(predicates, gold_spans, seq_lens, t, self.conf.task.wprd)
        
        x = emb
        
        if self.is_lstm:
            if hasattr(self.conf.encoder, 'var_lstm') and self.conf.encoder.var_lstm:
                emb_pack = pack_padded_sequence(emb, (seq_lens).cpu(), True, False)
                x, _ = self.encoder(emb_pack)
                x, _ = pad_packed_sequence(x, True, total_length = emb.shape[1])
                
            else:
                x = self.encoder(emb, seq_lens)
        else:
            x = self.bert_dropout(x)
            
        h = x
        # pdb.set_trace()
        span, ph, pt, ph_labels = self.firstorder(h)
        
        # shstoh, shstot = self.secondorder(h)
        shstoh, shstot = None, None
        
        firstorder = [span, ph, pt]
        secondorder = [shstoh, shstot]
        label_scores = ph_labels

        masks = [mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop, span_mask]
        
        if inference:
            loss = None
            # pdb.set_trace()
            span_preds, span_labels, span_m, ph_m, pt_m = self.inference(firstorder, maskspan, self.conf.decoder.mbr)
            span_socres = span_m[...,1]
            
            #TODO -1e9?
            span_socres.masked_fill_(~maskspan, -1e9)
            ph_m.masked_fill_(~maskarc, -1e9)
            pt_m.masked_fill_(~maskarc, -1e9)
            
            label_scores = label_scores.softmax(-1)
        else:
            span_preds, span_socres, ph_m, pt_m = None, None, None, None
            arc_loss = self.treecrf_loss(firstorder, fo_ind, masks)
            label_loss = self.label_loss(label_scores, ph_labels_ind, fo_ind[1])
            
            loss = (1- self.conf.loss.lamb) * arc_loss +  self.conf.loss.lamb * label_loss
            
        return {'loss':loss, 'arcs':[span_preds, ph_m, pt_m],
                    'labels':label_scores, 'scores':span_socres, 'span_labels':span_labels}
    
    @torch.no_grad()
    def inference(self, firstorder, maskspan, mbr = True):
        """
            Implementing mbr to compute global optimization score for every p->{spans}
        """
        scores = firstorder
        
        span_m, ph_m, pt_m = scores[:3]
        
        span_preds = None
        span_marginals = span_m
        
        span_max, span_labels = span_m.max(-1)
        if 'argmax' in self.conf.decoder.mode:
            span_preds = cyk(span_max, lens=maskspan[:, 0].sum(-1), r_closed=True, decode=True)
        
        if 'marginal' in self.conf.decoder.mode:
            span_marginals = cyk(span_m, lens=maskspan[:, 0].sum(-1), r_closed=True, marginals = span_marginals)
        
        return span_preds, span_labels, span_marginals, ph_m, pt_m
        
        
        
        
        
        