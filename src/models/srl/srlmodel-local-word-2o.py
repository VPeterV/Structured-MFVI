import copy
import logging
import pdb
import torch
from torch import nn
from ..utils import unfold_arc_label, unfold_graphs
from ..modules.loss.local_loss import LocalLoss
from ..modules.loss.label_loss import PhLabelLoss
from ..modules.encoder.lstm_encoder import LSTMEncoder
from ..modules.encoder.var_lstm_encoder import VariationalLSTM
from ..modules.embeddings.indembedding import IndEmbedding
from ..modules.embeddings.plmembedding import Embedding
from ..modules.embeddings.w2vembedding import W2VEmbedding
from ..modules.structs.mf2o import SecondOrderMF
from ..modules.scorer.affine_fp_word import SRLFirstOrderScorer, SRLSecondOrderScorer
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
        self.secondorder = SRLSecondOrderScorer(conf.soscorer, self.affine_input)
        
        self.mfvi = SecondOrderMF(conf.mfvi)
        
        self.local_loss = LocalLoss()
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
        
        gold_spans = y_table['gold_spans']      # idx starts from 1, but [cls] has been added
        span_labels = y_table['span_labels']    # idx starts from 1, but [cls] has been added
        predicates = y_table['predicates']      # idx starts from 1, but [cls] has been added
                        
        ph_labels_ind = unfold_arc_label(predicates, gold_spans[...,0], span_labels, t)

        fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop = \
                        unfold_graphs(predicates, gold_spans, seq_lens, t, self.conf.task.wprd, so = True)
        
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

        # 1o score
        span, ph, pt, ph_labels = self.firstorder(h)
        
        # 2o score
        span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop = self.secondorder(h)
        
        firstorder = [span, ph, pt]
        secondorder = [span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop]
        label_scores = ph_labels

        masks = [mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop]
        
        mf = self.mfvi(firstorder, secondorder, maskspan, maskarc, mask2o_pspan, mask2o_psib, mask2o_pcop, inference = inference)
        
        if inference:
            loss = None
            # pdb.set_trace()
            span_m, ph_logits, pt_logits = self.inference(mf, self.conf.decoder.mbr)
            
            #TODO -1e9?
            span_m.masked_fill_(~maskspan, -1e9)
            ph_logits.masked_fill_(~maskarc, -1e9)
            pt_logits.masked_fill_(~maskarc, -1e9)
            
            label_scores = label_scores.softmax(-1)
        else:
            span_m, ph_logits, pt_logits = None, None, None
            arc_loss = self.local_loss(mf['logits'], fo_ind, masks)
            label_loss = self.label_loss(label_scores, ph_labels_ind, fo_ind[1])
            
            loss = (1- self.conf.loss.lamb) * arc_loss +  self.conf.loss.lamb * label_loss
            
        return {'loss':loss, 'arcs':[span_m, ph_logits, pt_logits],
                    'labels':label_scores}
        
    def get_arc_loss(self, mf_logits, fo_indicator, masks):
        spans_ind, ph_ind, pt_ind= fo_indicator

        mask, maskarc, maskspan, _, _ = masks
        
        span, ph, pt = mf_logits
        
        spans_ind = torch.where(spans_ind > 0, spans_ind - 1, spans_ind) 
        span_logits = span.masked_select(maskspan)
        spans_ind = spans_ind.masked_select(maskspan)
        
        loss_spans = self.bceloss(span_logits, spans_ind.to(torch.float))

        # local norm
        ph = ph.masked_select(maskarc)
        ph_ind = ph_ind.masked_select(maskarc)
        
        pt = pt.masked_select(maskarc)
        pt_ind = pt_ind.masked_select(maskarc)
        
        loss_ph = self.bceloss(ph, ph_ind.to(torch.float))
        loss_pt = self.bceloss(pt, pt_ind.to(torch.float))

        loss = loss_spans + loss_ph + loss_pt
        
        return loss
    
    def inference(self, mf, mbr = True):
        span_logits, ph_logits, pt_logits = mf['logits']
        
        return span_logits, ph_logits, pt_logits