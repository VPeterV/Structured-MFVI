import torch
import logging
from collections import Counter

from torchmetrics import Metric
from functools import cmp_to_key

log = logging.getLogger(__name__)

class NERMetric(Metric):
    def __init__(self, conf, fields):
        super(NERMetric, self).__init__()
        self.cfg = conf
        self.fields = fields
        self.vocab = fields.get_vocab('words')
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx = "sum")    # gold
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx = "sum")    # pred
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx = "sum")   # total samples
        self.add_state("n_multiple", default=torch.tensor(0.), dist_reduce_fx = "sum")  # number with multiple
        self.add_state("c_multiple", default=torch.tensor(0.), dist_reduce_fx = "sum")  # correct number with multiple
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx = "sum")   # number of total correct unlabeled span
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx = "sum")   # number of total correct labeled ner
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx = "sum")     # number of correct unlabeled span
        self.add_state("ltp", default=torch.tensor(0.), dist_reduce_fx = "sum")     # number of correct labeled ner
        
        self.eps = 1e-12
        
        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "ner"
        
    def update(self, info):
        preds = info["ner"]
        golds = info['gold_ner']

        pr = []
        for b_pred_list in preds:
            bpr = []
            for head in b_pred_list.values():
                for pred in head:
                    bpr.append((pred['spans'][0], pred['spans'][1], pred['entity_type']))
            pr.append(bpr)
        preds = pr
        
        if self.cfg.write_result_to_file:
            output = {}
            output['words'] = info['words']
            output['gold_ner'] = golds
            output['preds'] = pr
            self.outputs.append(output)
        
        _n_ucm, _n_lcm, _utp, _ltp, _pred, _gold = 0, 0, 0, 0, 0, 0
        self.n += len(preds)
        
        # import pdb
        # pdb.set_trace()
        
        for pred, gold in zip(preds, golds):
            
            upred = Counter([(i, j) for i,j,_ in pred])
            
            try:
                ugold = Counter([(i,j) for i,j, _ in gold])
            except:
                assert len(gold) == 0
                ugold = Counter()
                
            utp = list((upred & ugold).elements())
            
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            
            _n_ucm += float(len(utp) == len(pred) == (gold))
            _n_lcm += float(len(ltp) == len(pred) == (gold))
            
            _utp += len(utp)
            _ltp += len(ltp)
            _pred += len(pred)
            _gold += len(gold)
            
        self.n_ucm += _n_ucm
        self.n_lcm += _n_lcm
        self.utp += _utp
        self.ltp += _ltp
        self.pred += _pred
        self.gold += _gold
        
    def compute(self, test=True, **kwargs):
        super(NERMetric, self).compute()
        
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test)
        
        return self.result
        
    @property
    def result(self):
        return {
            'c_ucm': self.ucm(),
            'c_lcm': self.lcm(),
            'up': self.up(),
            'ur': self.ur(),
            'uf': self.uf(),
            'lp': self.lp(),
            'lr': self.lr(),
            'lf': self.lf(),
            'score': self.lf(),
        }
        
    def ucm(self):  # unlabled total correct ratio
        return ( self.n_ucm / (self.n + self.eps)).item()

    def lcm(self):  # labled total correct ratio
        return (self.n_lcm / (self.n + self.eps)).item()

    def up(self):   # unlabeled-precision
        return (self.utp / (self.pred + self.eps)).item()

    def ur(self):   # unlabeled recall
        return (self.utp / (self.gold + self.eps)).item()

    def uf(self):   # unlabeled f1
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    def lp(self):   # unlabled  precision
        return (self.ltp / (self.pred + self.eps)).item()

    def lr(self):   # labeled recall
        return (self.ltp / (self.gold + self.eps)).item()

    def lf(self):   # labeled f1 score
        return (2 * self.ltp / (self.pred + self.gold + self.eps)).item()
        
    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs

        words = [output['words'] for output in outputs]
        pred_spans = [output['preds'] for output in outputs]
        gold_spans = [output['gold_ner'] for output in outputs]

        final_results = []

        for batch in zip(words, pred_spans, gold_spans):
            batch_words, batch_pred_span, batch_gold_span = batch

            for i in range(batch_words.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_words[i])
                a.append(batch_pred_span[i])
                a.append(batch_gold_span[i])
                final_results.append(a)


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (sen, pred_span, gold_span) in final_results:
                f.write(f'{sen}')
                f.write('\n')
                f.write(f'pred_spans:{pred_span}')
                f.write('\n')
                f.write(f'gold_spans:{gold_span}')
                f.write('\n')
                
class RelationMetric(Metric):
    def __init__(self, conf, fields):
        super(RelationMetric, self).__init__()
        self.cfg = conf
        self.fields = fields
        self.vocab = fields.get_vocab('words')
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx = "sum")    # gold
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx = "sum")    # pred
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx = "sum")   # total samples
        self.add_state("rtp", default=torch.tensor(0.), dist_reduce_fx = "sum") # relation for tp
        self.add_state("stp", default=torch.tensor(0.), dist_reduce_fx = "sum") # strict relation for tp
        self.add_state("n_scm", default=torch.tensor(0.),  dist_reduce_fx = "sum")  # totally correct strict relations in one sentence
        self.add_state("n_rcm", default=torch.tensor(0.),  dist_reduce_fx = "sum")  # totally correct relations in one sentence
        self.add_state("urtp", default=torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("urpred", default=torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("urgold", default=torch.tensor(0.), dist_reduce_fx = "sum")

        
        self.eps = 1e-12
        
        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "relation"
        
    def update(self, info):
        ner_preds = info["ner"]
        rel_preds = info['relations']
        ner_golds = info['gold_ner']
        rel_golds = info['gold_relations']
        
        ner_pr = []
        for b_pred_list in ner_preds:
            b_ner_pr = {}
            for h_dict in b_pred_list.values():
                for pred in h_dict:
                    # ner_pr.append((pred['spans'][0], pred['spans'][1], pred['entity_type']))
                    head, tail = pred['spans'][0], pred['spans'][1]
                    # if (head, tail) not in ner_pr:
                    #     ner_pr[(head, tail)] = []
                    b_ner_pr[(head, tail)] = [head, tail, pred['entity_type']]
            ner_pr.append(b_ner_pr)
        
        if self.cfg.write_result_to_file:
            output = {}
            output['words'] = info['words']
            output['gold_ner'] = ner_golds
            output['ner_preds'] = ner_pr
            output['gold_relations'] = rel_golds
            output['rel_preds'] = rel_preds
            self.outputs.append(output)
        
        _n_scm, _n_rcm, _stp, _rtp, _urtp, _pred, _gold, _urpred, _urgold = 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.n += len(rel_preds)
        
        # import pdb
        # pdb.set_trace()
        
        for i, (rel_pred, rel_gold) in enumerate(zip(rel_preds, rel_golds)):
            
            urpred = Counter([(i, j, x, y) for i, j, x, y, _ in rel_pred])
            
            try:
                urgold = Counter([(i,j, x, y) for i, j, x, y, _ in rel_gold])
            except:
                assert len(rel_gold) == 0
                urgold = Counter()
                
            spred = []
            for rel in rel_pred:
                try:
                    if tuple(ner_pr[i][rel[0], rel[1]]) in ner_golds[i] \
                        and tuple(ner_pr[i][rel[2], rel[3]]) in ner_golds[i]:
                        spred.append(rel)
                except KeyError:
                    pass
            
            urtp = list((urpred & urgold).elements())
            
            rpred = Counter(rel_pred)
            rgold = Counter(rel_gold)
            rtp = list((rpred & rgold).elements())
            
            spred = Counter(spred)
            stp = list((spred & rgold).elements())
            
            _n_rcm += float(len(rtp) == len(rel_pred) == (rel_gold))
            _n_scm += float(len(stp) == len(rel_pred) == (rel_gold))
            
            _urtp += len(urtp)
            _rtp += len(rtp)
            _stp += len(stp)
            _pred += len(rel_pred)
            _gold += len(rel_gold)
            
            _urpred += len(urpred)
            _urgold += len(urgold)
        
        self.urtp += _urtp
        self.n_rcm += _n_rcm
        self.n_scm += _n_scm
        self.rtp += _rtp
        self.stp += _stp
        self.pred += _pred
        self.gold += _gold
        self.urpred += _urpred
        self.urgold += _urgold
        
    def compute(self, test=True, **kwargs):
        super(RelationMetric, self).compute()
        
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test)
        
        # import pdb
        # pdb.set_trace()
        
        return self.result
        
    @property
    def result(self):
        return {
            # 'c_rcm': self.rcm(),
            # 'c_scm': self.scm(),
            'urp': self.urp(),
            'urr': self.urr(),
            'urf': self.urf(),
            'rp': self.rp(),
            'rr': self.rr(),
            'rf': self.rf(),
            'sp': self.sp(),
            'sr': self.sr(),
            'sf': self.sf(),
            'score': self.rf(),
        }
                
    def rcm(self):  # relation total correct ratio
        return ( self.n_rcm / (self.n + self.eps)).item()

    def scm(self):  # strict relation total correct ratio
        return (self.n_scm / (self.n + self.eps)).item()

    def urp(self):
        return (self.urtp / (self.urpred + self.eps)).item()
        
    def urr(self):
        return (self.urtp / (self.urgold + self.eps)).item()
        
    def urf(self):
        return (self.urtp / (self.urpred + self.urgold + self.eps)).item()
    
    def rp(self):   # relation-precision
        return (self.rtp / (self.pred + self.eps)).item()

    def rr(self):   # relation recall
        return (self.rtp / (self.gold + self.eps)).item()

    def rf(self):   # relation f1
        return (2 * self.rtp / (self.pred + self.gold + self.eps)).item()

    def sp(self):   # strict relation  precision
        return (self.stp / (self.pred + self.eps)).item()

    def sr(self):   # strict relation recall
        return (self.stp / (self.gold + self.eps)).item()

    def sf(self):   # strict relation f1 score
        return (2 * self.stp / (self.pred + self.gold + self.eps)).item()
        
    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs

        words = [output['words'] for output in outputs]
        pred_rel = [output['rel_preds'] for output in outputs]
        gold_rel = [output['gold_relations'] for output in outputs]

        final_results = []

        for batch in zip(words, pred_rel, gold_rel):
            batch_words, batch_pred_rel, batch_gold_rel = batch

            for i in range(batch_words.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_words[i])
                a.append(batch_pred_rel[i])
                a.append(batch_gold_rel[i])
                final_results.append(a)


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (sen, pred_rel, gold_rel) in final_results:
                f.write(f'{sen}')
                f.write('\n')
                f.write(f'pred_relations:{pred_rel}')
                f.write('\n')
                f.write(f'gold_relations:{gold_rel}')
                f.write('\n')