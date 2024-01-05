import os
import pdb
import json
import torch
import shutil
import tempfile
import subprocess
import logging
from collections import Counter

from torchmetrics import Metric

from functools import cmp_to_key

from wandb import set_trace

log = logging.getLogger(__name__)

class SpanSRLMetric(Metric):

    def __init__(self, conf, fields, eps=1e-12):
        """
        ref https://github.com/yzhangcs/crfsrl/blob/main/crfsrl/metric.py
        Implementing official metric script to compute f1
        URL = 'http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz'
        """
        super().__init__()
        
        self.add_state("tp", default = torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("pred", default = torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("gold", default = torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("prd_tp", default = torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("prd_pred", default = torch.tensor(0.), dist_reduce_fx = "sum")
        self.add_state("prd_gold", default = torch.tensor(0.), dist_reduce_fx = "sum")

        self.eps = eps
        
        self.conf = conf
        self.fields = fields
        self.data_path = '/nas-alinlp/peter.lw/code/span-rl/data'
        self.script = os.path.join(self.data_path, 'srlconll-1.1/bin/srl-eval.pl')
        
        if self.conf.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "relation"

    def update(self, info):
        preds, golds = info['srl'], info['gold_srl']
        if self.conf.write_result_to_file:
            output = {}
            output['words'] = info['words']
            output['srl'] = preds
            output['gold_srl'] = golds
            if info['p_spans'] is not None:
                output['p_spans'] = info['p_spans']
            self.outputs.append(output)
            
        lens = [max(max([*i, *j], key=lambda x: max(x[:3]))[:3]) if i or j else 1 for i, j in zip(preds, golds)]
        ftemp = tempfile.mkdtemp()
        fpred, fgold = os.path.join(ftemp, 'pred'), os.path.join(ftemp, 'gold')
        with open(fpred, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(preds)]))
        with open(fgold, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(golds)]))
        # pdb.set_trace()
        os.environ['PERL5LIB'] = os.path.join(self.data_path, 'srlconll-1.1', 'lib:$PERL5LIB')
        try:
            p_out = subprocess.check_output(['perl', f'{self.script}', f'{fpred}', f'{fgold}'], stderr=subprocess.STDOUT).decode()
            r_out = subprocess.check_output(['perl', f'{self.script}', f'{fgold}', f'{fpred}'], stderr=subprocess.STDOUT).decode()
        except:
            log.info(f"preds error please see preds {fpred} and golds {fgold}")
            raise ValueError
        p_out = [i for i in p_out.split('\n') if 'Overall' in i][0].split()
        r_out = [i for i in r_out.split('\n') if 'Overall' in i][0].split()
        shutil.rmtree(ftemp)

        self.tp += int(p_out[1])
        self.pred += int(p_out[3]) + int(p_out[1])
        self.gold += int(r_out[3]) + int(p_out[1])
        for pred, gold in zip(preds, golds):
            prd_pred, prd_gold = {span[0] for span in pred}, {span[0] for span in gold}
            self.prd_tp += len(prd_pred & prd_gold)
            self.prd_pred += len(prd_pred)
            self.prd_gold += len(prd_gold)
            
    def compute(self, test = True, **kwargs):
        super(SpanSRLMetric, self).compute()
        
        if self.conf.write_result_to_file:
            self._write_result_to_json(test)
        
        return self.result

    def __repr__(self):
        return f"PRD: {self.prd_p:6.2%} {self.prd_r:6.2%} {self.prd_f:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @classmethod
    def span2prop(cls, spans, length):
        prds, args = ['-'] * length, {}
        for prd, start, end, role in spans:
            prds[prd-1] = str(prd)
            if prd not in args:
                args[prd] = ['*'] * length
                args[prd][prd-1] = '(V*)'
            args[prd][start-1] = f'({role}*'
            args[prd][end-1] += ')'
        args = [args[key] for key in sorted(args)]
        return '\n'.join(['\t'.join(i) for i in zip(*[prds, *args])])
        
    @property
    def result(self):
        return {
            'prd_p': self.prd_p,
            'prd_r': self.prd_r,
            'prd_f': self.prd_f,
            'p': self.p,
            'r': self.r,
            'f': self.f,
            'score': self.score,
        }
        
    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

    @property
    def prd_p(self):
        return self.prd_tp / (self.prd_pred + self.eps)

    @property
    def prd_r(self):
        return self.prd_tp / (self.prd_gold + self.eps)

    @property
    def prd_f(self):
        return 2 * self.prd_tp / (self.prd_pred + self.prd_gold + self.eps)
            
    def _write_result_to_json(self, test=False):
        '''
        get_name_from_id
        '''
        mode = 'test' if test else 'valid'
        outputs = self.outputs
        
        # import pdb
        # pdb.set_trace()
        words = [output['words'] for output in outputs]
        pred_srl = [output['srl'] for output in outputs]
        gold_srl = [output['gold_srl'] for output in outputs]
        p_spans = [output['p_spans'] if 'p_spans' in output else None for output in outputs]
    
        final_results = []
    
        for batch in zip(words, pred_srl, gold_srl, p_spans):
            batch_words, batch_pred_srl, batch_gold_srl, batch_ps  = batch
    
            for i in range(batch_words.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = {}
                words_list = batch_words[i].tolist()
                batch_words_origin = []
                for idx in words_list:
                    if idx == self.fields.get_pad_index('words'):
                        break
                    batch_words_origin.append(self.fields.get_name_from_id('words',idx))
                a["sentences"] = [batch_words_origin]
                a["pred_srl"] = [batch_pred_srl[i]]
                a["gold_srl"] = [batch_gold_srl[i]]
                if batch_ps is not None:
                    a["p_spans"] = [batch_ps[i]]
                    
                final_results.append(a)
        
        with open(f"{self.prefix}_output_{mode}.json", 'w', encoding='utf8') as f:
            for item in final_results:
                json_item = json.dumps(item)
                f.write(json_item+'\n')