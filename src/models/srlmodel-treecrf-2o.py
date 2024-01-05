import logging
import pdb
from turtle import shape
import torch
from torch import nn
from .utils import unfold_span_label, potential_norm, unfold_graphs, get_word_mask
from .modules.loss.treecrf_loss import TreecrfLoss
from .modules.loss.label_loss import PhLabelLoss
from .modules.loss.local_loss import LocalLossDict
from .modules.encoder.lstm_encoder import LSTMEncoder
from .modules.encoder.var_lstm_encoder import VariationalLSTM
from .modules.encoder.span_encoder import SpanEncoder
from .modules.scorer.span_scorer import SpanScorer
from .modules.embeddings.indembedding import IndEmbedding
from .modules.embeddings.plmembedding import Embedding
from .modules.embeddings.w2vembedding import W2VEmbedding
from .modules.structs.cyk import cyk
from .modules.structs.mf2o import SecondOrderMF
from .modules.scorer.affine_fp_word import SRLFirstOrderScorer, SRLSecondOrderScorer
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
                self.affine_input = conf.encoder.n_lstm_hidden * 2 * conf.encoder.bilstm
            elif hasattr(self.conf.encoder, 'bihlstm') and self.conf.encoder.bihlstm:
                self.encoder = StackedAlternatingLstmSeq2SeqEncoder(
                            input_size = input_for_lstm,
                            hidden_size = conf.encoder.n_lstm_hidden,
                            num_layers = conf.encoder.n_lstm_layers,
                            recurrent_dropout_probability = conf.encoder.lstm_dropout,
                            use_input_projection_bias = conf.encoder.use_input_projection_bias
                )
                self.affine_input = self.encoder.get_output_dim()
                self.after_lstm_dropout = nn.Dropout(conf.encoder.after_lstm_dropout)
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
        
        self.spanencoder = SpanEncoder(conf.span_encoder, self.affine_input, input_for_lstm)
        self.span_affine_input = self.spanencoder.get_span_dim()
        self.span_label_scorer = SpanScorer(conf.foscorer, self.affine_input, self.span_affine_input, self.num_span_label)
        self.span_arc_scorer = SpanScorer(conf.foscorer, self.affine_input, self.span_affine_input, 1)
        
        self.span_arc_loss = LocalLossDict()
        
        self.mfvi = SecondOrderMF(conf.mfvi)
        
        if hasattr(conf.loss, "pos_weight"):
            pos_weight = conf.loss.pos_weight
        else:
            pos_weight = 1.

        if hasattr(conf.loss, "neg_weight"):
            neg_weight = conf.loss.neg_weight
        else:
            neg_weight = 1.

        log.info(f"Weights of postive labels: {pos_weight}")
        log.info(f"Weights of postive labels: {neg_weight}")
        
        self.treecrf_loss = TreecrfLoss(conf.loss.span_lamb, conf.treecrf.potential_norm, 
                                        conf.treecrf.structure_smoothing, pos_weight = pos_weight, neg_weight = neg_weight,
                                        ent_lamb = conf.loss.ent_lamb, entropy_all = conf.loss.entropy_all)
                                        
        self.label_loss = PhLabelLoss(self.num_span_label)
        
    def forward(self, x_table, y_table,  inference = False):
        
        # pdb.set_trace()
        # breakpoint()
        seq_lens = x_table['seq_len']
        # pdb.set_trace()
        if 'bert' in x_table:
            x = x_table['bert']
            seq_lens += 2   # not fencepost but including [SEP] temporarily
            emb = self.embedding(x)
        else:
            x = x_table['words']
            emb = self.embedding(x_table)
        
        if self.conf.task.wprd and self.conf.embeddings.ind_emb > 0:
            indemb = self.ind_embedding(x_table)
            emb = torch.concat([emb, indemb], dim = -1)

        # if x.size(-1) >= 7:
        # breakpoint()
        
        t = x.size(1)
        span_mask = x_table['span_mask']
        span_mask = span_mask.bool()
        
        gold_spans = y_table['gold_spans']      # idx starts from 1, but [cls] has been added
        span_labels = y_table['span_labels']    # idx starts from 1, but [cls] has been added
        predicates = y_table['predicates']      # idx starts from 1, but [cls] has been added
        
        label_graph, span_label_ind, span_label_mask, gold_prds = unfold_span_label(predicates, span_labels, t)

        wordmask = get_word_mask(seq_lens, t, fencepost = False)
        fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop = \
                        unfold_graphs(predicates, gold_spans, seq_lens, t, self.conf.task.wprd, so = True)
        # print('bert')
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        x = emb
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        
        if self.is_lstm:
            if hasattr(self.conf.encoder, 'var_lstm') and self.conf.encoder.var_lstm:
                emb_pack = pack_padded_sequence(emb, (seq_lens).cpu(), True, False)
                x, _ = self.encoder(emb_pack)
                x, _ = pad_packed_sequence(x, True, total_length = emb.shape[1])
            elif hasattr(self.conf.encoder, 'bihlstm') and self.conf.encoder.bihlstm:
                # emb_pack = pack_padded_sequence(emb, (bert_seq_lens).cpu(), True, False)
                # breakpoint()
                x = self.encoder(x, wordmask)
                x = self.after_lstm_dropout(x)
                # x, _ = pad_packed_sequence(x, True, total_length = emb.shape[1])
            else:
                x = self.encoder(emb, seq_lens)
        else:
            x = self.bert_dropout(x)
        h = x

        # 1o score
        # print('biaffine')
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        span, ph, pt, _ = self.firstorder(h)
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        
        # 2o score
        # print('triaffine')
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop = self.secondorder(h)
        # print(torch.cuda.memory_allocated() // 2 ** 30)
        
        firstorder = [span, ph, pt]
        secondorder = [span_psh, span_pst, ph_sib, pt_sib, ph_cop, pt_cop]

        masks = [mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop, span_mask]
        
        mf = self.mfvi(firstorder, secondorder, maskspan, maskarc, mask2o_pspan, mask2o_psib, mask2o_pcop, inference = inference)
        
        if inference:
            loss = None
            # pdb.set_trace()
            span_preds, span_label_scores, span_arc_scores,\
                span_m, ph_m, pt_m = self.inference(h, mf, maskspan, self.conf.decoder.mbr, emb)
            
            span_socres = span_m[...,1]
            
            #TODO -1e9?
            span_socres.masked_fill_(~maskspan, 0)
            ph_m.masked_fill_(~maskarc, 0.)
            pt_m.masked_fill_(~maskarc, 0.)
            
            if self.conf.task.wprd:
                gold_prds = gold_prds.bool()
                span_arc_scores.masked_fill_((~gold_prds)[:,:,None], 0.)
            
            # label_scores = label_scores.softmax(-1)
        else:
            span_repr = self.spanencoder(h, gold_spans, emb)
            span_label_scores = self.span_label_scorer(h, span_repr)
            span_arc_scores = self.span_arc_scorer(h, span_repr)
            
            span_preds, span_socres, ph_m, pt_m = None, None, None, None
            arc_loss = self.treecrf_loss(mf['logits'], fo_ind, masks)
            label_loss = self.label_loss(span_label_scores, label_graph, span_label_ind)
            
            span_label_mask_prd = span_label_mask
            
            if self.conf.task.wprd:
                gold_prds = gold_prds.bool()
                span_label_mask_prd = gold_prds[..., None] & span_label_mask_prd
            
            span_arc_loss = self.span_arc_loss({"span_arc": span_arc_scores}, 
                                                {"span_arc": span_label_ind.float()},
                                                {"span_arc": span_label_mask_prd})
            
            loss = (1- self.conf.loss.lamb) * (arc_loss + span_arc_loss) +  self.conf.loss.lamb * label_loss
            # loss = arc_loss + label_loss
            
        return {'loss':loss, 'arcs':[span_preds, ph_m, pt_m],
                'labels':span_label_scores, "span_arcs":span_arc_scores, 
                'scores':span_socres}
    
    def inference(self, hiddens, mf, maskspan, mbr = True, emb = None):

        span_logits, ph_logits, pt_logits = mf['logits']
        
        span_preds = None
        span_marginals = span_logits
        span_max, span_labels = span_marginals.max(-1)
        
        # pdb.set_trace()
        # breakpoint()
        # if 'argmax' in self.conf.decoder.mode:
        span_preds = cyk(span_max, lens=maskspan[:, 0].sum(-1), r_closed=True, decode=True)
        
        bsz = span_logits.size(0)
        
        spans_batch = [[]] * bsz
        
        spans_idx = span_preds.tolist()
        
        # log.info(f'span_labels:{span_labels.sum()}')
        for spans in spans_idx:
            b, i, j = spans
            if span_labels[b,i,j] == 1:
                spans_batch[b] = spans_batch[b] + [[i, j]]
                
        max_num_spans = max(len(i) for i in spans_batch)
        
        if max_num_spans == 0:
            max_num_spans = 1

        pred_spans = [i + (max_num_spans - len(i)) * [[-1,-1]] for i in spans_batch]
        
        pred_spans = torch.tensor(pred_spans, requires_grad = False).to(span_logits.device)
        # pred_mask = torch.where(pred_spans[..., 0] == -1, False, True)
        # breakpoint()
        span_repr = self.spanencoder(hiddens, pred_spans, emb)
        label_scores = self.span_label_scorer(hiddens, span_repr)
        span_arc_scores = self.span_arc_scorer(hiddens, span_repr)
        
        span_marginals = cyk(span_marginals, lens=maskspan[:, 0].sum(-1), r_closed=True, marginals = span_marginals)
        
        
        return spans_batch, label_scores, span_arc_scores.sigmoid(), span_marginals, ph_logits.sigmoid(), pt_logits.sigmoid()
