# Code to do evaluation of predictions for a loaded dataset.

import pdb
import torch
import logging
from modulefinder import Module
import os
from collections import Counter
import numpy as np
import torch
from fastNLP.core.dataset import DataSet, Instance
from fastNLP.core.field import AutoPadder, EngChar2DPadder

from supar.utils import Embedding
from ..base import DataInterface
from ..utils.fields import Field, SpanLabelField, SubwordField, IndField
from supar.utils.common import *
from ..utils.fn import generate_spanmask

from functools import partial

log = logging.getLogger(__name__)


class CONLLDataLoader(DataInterface):
    def __init__(self, conf):
        super(CONLLDataLoader, self).__init__(conf)
        
    def get_inputs(self):
        # return ['seq_len', 'words','span_mask', 'ind', 'char', 'lemma']
        return ['seq_len', 'words','span_mask', 'ind']
        
    def get_targets(self):
        return ['gold_spans', 'span_labels', 'predicates']
    
    def build_datasets(self):
        # pdb.set_trace()
        conf = self.conf
        datasets = {}
        datasets['train'] = self._load(conll_file = conf.train)
        datasets['dev'] = self._load(conll_file = conf.dev)
        datasets['test'] = self._load(conll_file = conf.test)
        if 'conll05' in self.conf.name:
            datasets['test_ood'] = self._load(conll_file = conf.test_ood)
        
        return datasets
        
    def _load(self, conll_file):
        log.info(f"Loading: {conll_file}")
        
        examples = self.read_conll(conll_file)
        
        instances = [Instance(**example) for example in examples]
        dataset = DataSet(instances)
                                    
        # dataset.add_field('words', words)
        # dataset.add_field('raw_words', words)
        # dataset.add_field('gold_spans', gold_spans)     # golden spans
        # dataset.add_field('span_labels', span_labels)   # span labels
        # dataset.add_field('subjects', subjects)
        # dataset.add_field('objects', objects)
        # dataset.add_field('relation_labels', rel_labels)
        
        dataset.copy_field('words', 'raw_words')
        dataset.add_seq_len('words', 'seq_len')
        
        generate_spanmask_partial = partial(generate_spanmask, bert = self.conf.use_bert)
        dataset.apply_more(generate_spanmask_partial)
        # dataset.add_field('bert',words)
        
        # padder
        dataset.set_padder('span_mask', EngChar2DPadder(0))
        dataset.set_pad_val('span_labels', self.conf.target_pad)
        dataset.set_pad_val('predicates', self.conf.target_pad)
        dataset.set_pad_val('gold_spans', self.conf.target_pad)
        
        
        log.info(f"loading: {conll_file} finished")
        
        return dataset

    def read_conll(self, in_file, lowercase=False, max_example=None):
        examples = []
        with open(in_file) as f:
            word, pos, predicates, spans, span_labels  = [], [], [], [], []
            lemma = []
            ind = []
            for line in f.readlines():
                sp = line.strip().split('\t')
                if len(sp) == 10:
                    if '_' not in sp[0]:
                        word.append(sp[1].lower() if lowercase else sp[1])
                        pos.append(sp[4])
                        lemma.append(sp[9])
                        if sp[9] != '_':
                            ind.append(1)
                        else:
                            ind.append(0)
                        if sp[8] != '_':
                            for span_annotation in sp[8].split('|'):
                                span_item = span_annotation.split(':')
                                predicates.append(int(span_item[0]))
                                span_w_labels = span_item[1]
                                span_idx = eval(span_w_labels.split('-')[0])
                                spans.append([span_idx[0], span_idx[1]])
                                sl = '-'.join(span_w_labels.split('-')[1:])
                                span_labels.append(sl)
                        # head.append(int(sp[6]))
                        # label.append(sp[7])
                elif len(word) > 0:
                    if len(spans) == 0:
                        spans.append([self.conf.target_pad,self.conf.target_pad])
                    if len(span_labels) == 0:
                        span_labels.append(self.conf.target_pad)
                    if len(predicates) == 0:
                        predicates.append(int(self.conf.target_pad))
                    if self.conf.use_bert:
                        examples.append({'words': word, 'pos': pos, 'predicates': predicates, 'gold_spans': spans, 'span_labels': span_labels, 
                        "ind":ind, 'char': word, 'lemma': lemma})
                    else:
                        examples.append({'words': word, 'pos': pos, 'predicates': predicates, 'gold_spans': spans, 'span_labels': span_labels,
                        'lemma':lemma, 'char': word, "ind":ind})
                    word, pos, predicates, spans, span_labels = [], [], [], [], []
                    lemma = []
                    ind = []
                    if (max_example is not None) and (len(examples) == max_example):
                        break
            
            if len(word) > 0:
                if self.conf.use_bert:
                    examples.append({'words': word, 'pos': pos, 'predicates': predicates, 'gold_spans': spans, 'span_labels': span_labels,
                    'ind':ind, 'char': word})
                else:
                    examples.append({'words': word, 'pos': pos, 'predicates': predicates, 'gold_spans': spans, 'span_labels': span_labels, 
                    'lemma':lemma, 'char': word, 'ind':ind})
                    
        return examples 
        
    def build_fields(self, train_data, dev_data = None, test_data = None):
        fields = {}
        add_none = False if not hasattr(self.conf, "add_none") else self.conf.add_none
        fields['span_labels'] = SpanLabelField('span_labels', dataname=self.conf.name, 
                                golden_predicate = self.conf.golden_predicate, target_pad = self.conf.target_pad, add_none = add_none)
        fields['ind'] = IndField('ind')
        
        if not self.conf.use_bert:
            fields['words'] = Field('words', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
            fields['char'] = SubwordField('char', pad = PAD, unk=UNK, bos=BOS, eos=EOS, fix_len = self.conf.fix_len)
            fields['lemma'] = Field('lemma', pad = PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        else:
            fields['words'] = Field('words', pad=PAD, unk=UNK, lower=True, min_freq=self.conf.min_freq)
            fields['char'] = SubwordField('char', pad = PAD, unk=UNK, bos=BOS, eos=EOS, fix_len = self.conf.fix_len)
            fields['lemma'] = Field('lemma', pad = PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)

        for name, field in fields.items():
            if not self.conf.use_bert and self.conf.pretrained and name == 'words':
                # TODO
                field.build(train_data[name], Embedding.load(self.conf.pretrained_path, self.conf.glove_unk))
            else:
                if isinstance(field, SpanLabelField):
                    field.build(train_data[name], dev_data[name], test_data[name])
                else:
                    field.build(train_data[name])
        return fields