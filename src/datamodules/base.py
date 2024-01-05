import pdb
import pytorch_lightning as pl
import os
from supar.utils.vocab import Vocab
from supar.utils.common import *
import pickle
from .utils.fields import SubwordField, Field
from fastNLP.core.dataset import DataSet
from fastNLP.core.sampler import RandomSampler, SequentialSampler, BucketSampler
from supar.utils import Embedding
from .utils.sampler import get_bucket_sampler
from fastNLP.core.batch import DataSetIter
import logging

# import copym
log = logging.getLogger(__name__)
import numpy as np
# from src.model.module.ember.ext_embedding import ExternalEmbeddingSupar

def set_inputs(datasets, inputs):
    for i in inputs:
        for dataset in datasets.values():
            # print(i)
            dataset.set_input(i)
            
def set_targets(datasets, targets):
    for t in targets:
        for dataset in datasets.values():
            # print(t)
            # print(dataset)
            dataset.set_target(t)

class Fields():
    def __init__(self, fields, inputs, config):
        self.fields = fields
        self.inputs = self._get_true_input_fields(inputs)
        # datamodule config..
        self.conf = config
        self.root_dir = os.getcwd()
    
    def get_span_label_num(self):
        if 'span_labels' in self.fields:
            return self.fields['span_labels'].num_span_labels 
        elif 'labels' in self.fields:
            return self.fields['labels'].num_span_labels 
        elif 'span_types' in self.fields.keys() and'frames' in self.fields.keys() and 'p2r_labels' in self.fields.keys():
            return self.fields['span_types'].num_span_labels, self.fields['frames'].num_span_labels, self.fields['p2r_labels'].num_span_labels
        else:
            raise NotImplementedError
            
    def get_span_label_dict(self):
        if 'span_labels' in self.fields.keys():
            return self.fields['span_labels'].label2id, \
            self.fields['span_labels'].id2label
        elif 'labels' in self.fields.keys():
            return self.fields['labels'].label2id, \
            self.fields['labels'].id2label
        elif 'span_types' in self.fields.keys() and'frames' in self.fields.keys() and 'p2r_labels' in self.fields.keys():
            return  self.fields['span_types'].label2id, self.fields['span_types'].id2label,\
                    self.fields['frames'].label2id, self.fields['frames'].id2label,\
                    self.fields['p2r_labels'].label2id, self.fields['p2r_labels'].id2label
        else:
            raise ValueError
            
    def get_bert_name(self):
        if 'bert' in self.fields.keys():
            return self.fields['bert_name']
        else:
            raise ValueError

    def get_ext_emb(self):
        if 'ext_emb' in self.fields:
            return self.fields['ext_emb']
        else:
            return None

    def _get_true_input_fields(self, inputs):
        true_inputs = []
        for i in inputs:
            if i in self.fields:
                true_inputs.append(i)
        return true_inputs
    
    @property
    def get_n_feats(self):
        n_feats = {}
        
        n_feats['words'] = len(self.fields['words'].vocab)
        
        if 'char' in self.fields:
            n_feats['char'] = len(self.fields['char'].vocab)
        
        if 'lemma' in self.fields:
            n_feats['lemma'] = len(self.fields['lemma'].vocab)
            
        return n_feats
    
    @property
    def get_char_pad(self):
        return self.fields['char'].pad_index

    def get_vocab_size(self, name):
        return len(self.fields[name].vocab)

    def get_name_from_id(self, name, id):
        return self.fields[name].vocab[id]

    def get_pad_index(self, name):
        return self.fields[name].pad_index

    def get_vocab(self, name):
        return self.fields[name].vocab
        
    def get_max_span_len(self):
        return self.conf.max_span_len
        
class DataInterface(pl.LightningDataModule):
    '''ref codebase: 
    https://github.com/sustcsonglin/pointer-net-for-nested/blob/main/src/datamodule/base.py#L64
    '''
    def __init__(self, conf):
        super(DataInterface, self).__init__()
        self.conf = conf
        
    def _load(self):
        raise NotImplementedError
    
    def build_datasets(self):
        raise NotImplementedError
        
    def build_fields(self):
        raise NotImplementedError
        
    def get_inputs(self):
        raise NotImplementedError
    
    def get_targets(self):
        raise NotImplementedError
        
    def _set_padder(self, datasets):
        pass
        
    def _make_assertion(self, datasets):
        pass
        
    def _post_process(self, datasets, fields):
        pass
    
    def build_data(self, **kwargs) -> None:
        datasets, fields = self._build_dataset_and_fields()
        inputs = self.get_inputs()
        targets = self.get_targets()
        
        if self.conf.use_bert:
            self._add_bert_to_field(datasets, fields)
            inputs.append('bert')
            # inputs.append('plm_spans')
        else:
            inputs.append('char')
            inputs.append('lemma')
            
            
        # for dataset in datasets.values():
        #     dataset.add_field
        
        # print(datasets)
        set_inputs(datasets, inputs)
        set_targets(datasets, targets)
        
        self.inputs = inputs
        self.datasets = datasets
        self.fields = Fields(fields=fields, inputs=inputs, config=self.conf)
        self._post_process(datasets, fields)
        self._set_padder(datasets)
        
        drop_type = 'seq_len'
        
        log.info(f"max_len:{self.conf.max_len}, before_drop: {len(self.datasets['train']['words'])}")
        self.datasets['train'] = self.datasets['train'].drop(lambda x: x[drop_type] > self.conf.max_len, inplace=True)
        self.datasets['train'] = self.datasets['train'].drop(lambda x: x[drop_type] < 2, inplace=True)
        log.info(f"after drop for len: {len(self.datasets['train']['words'])}")
        if self.conf.masked_inside:
            self.datasets['train'] = self.datasets['train'].drop(lambda x: x['valid'] == False, inplace=True)
            log.info(f"after drop for valid: {len(self.datasets['train']['words'])}")

        log.info(f"max_len:{self.conf.max_len}, before_drop: dev: {len(self.datasets['dev']['words'])}, test:{len(self.datasets['test']['words'])}")

        self.datasets['dev'] = self.datasets['dev'].drop(lambda x: x[drop_type] > self.conf.max_len_test, inplace=True)
        self.datasets['test'] = self.datasets['test'].drop(lambda x: x[drop_type] > self.conf.max_len_test, inplace=True)
        
        log.info(f"after drop: dev: {len(self.datasets['dev']['words'])}, test:{len(self.datasets['test']['words'])}")

        log.info(f"Train: {len(self.datasets['train']['words'])} sentences, valid: {len(self.datasets['dev']['words'])} sentences, test: {len(self.datasets['test']['words'])} sentences")
        log.info(f"Training max tokens: {self.conf.max_tokens}, total_bucket:{self.conf.bucket}")
        log.info(f"Testing max tokens: {self.conf.max_tokens_test}, total_bucket:{self.conf.bucket_test}")
        log.info(f"input: {self.inputs}")


    def train_dataloader(self):
        if self.conf.train_sampler_type == 'token':
            # if self.conf.sort_by_subwords:
            #     length = self.datasets['train'].get_field('subwords_len').content
            # else:
            length = self.datasets['train'].get_field('seq_len').content
            sampler = BucketSampler(num_buckets = self.conf.bucket, batch_size = self.conf.batch_size)
            sampler = get_bucket_sampler(lengths=length, max_tokens=self.conf.max_tokens,
                                n_buckets=self.conf.bucket, distributed=self.conf.distributed, evaluate=False)
            # for item in self.datasets['train'].labels:
                # print(len(item))
                # if len(item) == 0:
                #     breakpoint()
            return DataSetIter(self.datasets['train'], batch_size=self.conf.max_tokens, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler)
            # return DataSetIter(self.datasets['train'], batch_size=self.conf.batch_size, sampler=sampler, as_numpy=False, num_workers=4,
            #                    pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
            #                    batch_sampler=None)
        else:
            sampler = RandomSampler()
            return DataSetIter(self.datasets['train'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens, num_workers=4)        
    
    def val_dataloader(self):
        # if self.conf.train_sampler_type == 'token':
        #     length = self.datasets['dev'].get_field('seq_len').content
        #     sampler = get_bucket_sampler(lengths=length, max_tokens=self.conf.max_tokens_test,
        #                         n_buckets=self.conf.bucket, distributed=self.conf.distributed, evaluate=True)
        #     return DataSetIter(self.datasets['dev'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
        #                        pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
        #                        batch_sampler=sampler)
        # else:
        # sampler = RandomSampler()
        sampler = SequentialSampler()
        # return DataSetIter(self.datasets['dev'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens, num_workers=4)    
        return DataSetIter(self.datasets['dev'], sampler=sampler, batch_size=self.conf.batch_size, num_workers=4)
        
    def test_dataloader(self):
        # if self.conf.train_sampler_type == 'token':
        #     length = self.datasets['test'].get_field('seq_len').content
        #     sampler = get_bucket_sampler(lengths=length, max_tokens=self.conf.max_tokens_test,
        #                         n_buckets=self.conf.bucket, distributed=self.conf.distributed, evaluate=True)
        #     return DataSetIter(self.datasets['test'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
        #                        pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
        #                        batch_sampler=sampler)
        # else:
        # sampler = RandomSampler()
        sampler = SequentialSampler()
        # return DataSetIter(self.datasets['test'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens, num_workers=4)   
        return DataSetIter(self.datasets['test'], sampler=sampler, batch_size=self.conf.batch_size, num_workers=4)
        
        
    def _index_datasets(self, fields, datasets):
        # import pdb
        # pdb.set_trace()
        for _, dataset in datasets.items():
            for name, field in fields.items():
                dataset.apply_field(func=field.transform, field_name=name, new_field_name=name)        
        
    # def _build_external_emb(self, datasets, fields):
    #     assert self.conf.ext_emb_path, ("The external word embedding path does not exsit, please check.")
    #     log.info(f"use external embeddings :{self.conf.ext_emb_path}")
    #     has_cache = False
    #     cache_path = self.conf.ext_emb_path + ".cache.pickle"
    #     if self.conf.use_cache and  os.path.exists(cache_path):
    #         log.info(f"Load cache: {cache_path}")
    #         with open(cache_path, 'rb') as f:
    #             cache = pickle.load(f)
    #         WORD = cache['word']
    #         has_cache = True

    #     # https://github.com/yzhangcs/parser/blob/main/supar/utils/field.py line178
    #     if not has_cache:
    #         log.info("Find no cache, building..")
    #         cache = {}
    #         if 'glove' in self.conf.ext_emb_path:
    #             unk = 'unk'
    #         else:
    #             unk = None

    #         embed = Embedding.load(self.conf.ext_emb_path, unk=unk)
    #         WORD = fields['word']

    #         tokens = WORD.preprocess(embed.tokens)
    #         if embed.unk:
    #             tokens[embed.unk_index] = WORD.unk


    #         cache['origin_len'] = len(WORD.vocab)

    #         WORD.vocab.extend(tokens)
    #         embedding = torch.zeros(len(WORD.vocab), embed.dim)
    #         embedding[WORD.vocab[tokens]] = embed.vectors
    #         embedding /= torch.std(embedding)
    #         cache['embed'] = embedding

    #         for name, d in datasets.items():
    #             cache[name] = [WORD.transform(instance) for instance in d.get_field('raw_word').content]

    #         cache['word'] = WORD

    #         self._dump(cache_path=cache_path, to_cache=cache)

    #     for name, d in datasets.items():
    #         d.add_field('word', cache[name])

    #     fields['word'] = WORD
    #     fields['ext_emb'] = ExternalEmbeddingSupar(cache['embed'], cache['origin_len'], WORD.unk_index)
    #     log.info(f"before extend, vocab_size:{cache['origin_len']}")
    #     log.info(f"extended_vocab_size:{cache['embed'].shape[0]}")
        
    def _build_dataset_and_fields(self):
        # import pdb
        # pdb.set_trace()
        # breakpoint()
        cache_path = self.conf.cache
        postfix = ''
        if self.conf.masked_inside:
            postfix += '_masked_inside'
        if self.conf.use_bert:
            postfix += '_use_bert'
        if hasattr(self.conf, "fencepost") and self.conf.fencepost:
            postfix += '_fencepost'
        if hasattr(self.conf, "with_null") and self.conf.with_null:
            postfix += '_withnull'
        postfix += f'_fixlen{self.conf.fix_len}'

        cache_path += postfix
            
        log.info(f"looking for cache:{cache_path}, use_cache:{self.conf.use_cache}")
        if os.path.exists(cache_path) and self.conf.use_cache:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                datasets = cache['datasets']
                fields = cache['fields']
            log.info(f"load cache:{cache_path}")
        else:
            log.info("creating dataset.")
            # return: {"train", "dev","test"}
            datasets = self.build_datasets()
            fields = self.build_fields(datasets['train'], datasets['dev'], datasets['test'])
            self._index_datasets(fields, datasets)
            if self.conf.use_cache:
                self._dump(cache_path=cache_path, to_cache={'datasets': datasets,
                         'fields': fields})
        return datasets, fields        
        
    def _add_bert_to_field(self, datasets, fields):
        # 魔改 for span modification
        log.info(f"Use bert:{self.conf.bert}")
        
        bert_name = self.conf.bert
        if os.path.isdir(self.conf.bert):
            bert_name = (self.conf.bert).split('/')[-1]

        bert_cache_path = f"{self.conf.cache_bert}.{bert_name}"

        postfix = ''
        if self.conf.masked_inside:
            postfix += '_masked_inside'
        if hasattr(self.conf, "fencepost") and self.conf.fencepost:
            postfix += '_fencepost'
        if hasattr(self.conf, "with_null") and self.conf.with_null:
            postfix += '_withnull'

        postfix += f'_fixlen{self.conf.fix_len}'
            
        bert_cache_path += postfix
        if not os.path.exists(bert_cache_path) or not self.conf.use_bert_cache:
            bert_field = get_bert(self.conf.bert, fix_len = self.conf.fix_len)
            # ===================get_bert_cache=================== #
            def get_bert_cache(datasets):
                cache_bert = {}
                cache_bert['bert'] = bert_field
                
                for name, dataset in datasets.items():
                    if name not in cache_bert:
                        cache_bert[name] = {}
                    tokens= [bert_field.transform(seq) for seq in dataset.get_field('raw_words').content]
                    # for token, seq in zip(tokens,dataset.get_field('raw_words').content):
                        # print(token,end="\t")
                        # print(seq)
                    cache_bert[name]['bert'] = tokens
                    
                    # modify span index since bert uses word pieces here
                    # cache_bert[name]['plm_spans'] = [bert_field.modify_spans(tokens, ss) \
                    #                                     for ss in dataset.get_field('spans').content]                
                return cache_bert
            # ===================get_bert_cache=================== #
            cache_bert = get_bert_cache(datasets)
            if self.conf.use_bert_cache:
                self._dump(bert_cache_path, cache_bert)
        else:
            log.info(f"load cache bert:{bert_cache_path}")
            with open(bert_cache_path, 'rb') as file:
                cache_bert = pickle.load(file)
            bert_field = cache_bert['bert']

        for name, dataset in datasets.items():
            dataset.add_field('bert', cache_bert[name]['bert'])
            dataset.set_pad_val('bert', bert_field.pad_index)
            subwords_len = []
            # taking padding into consideration
            for tokens in cache_bert[name]['bert']:
                tokens_len = 0
                for subtokens in tokens:
                    tokens_len = max(len(subtokens), tokens_len)
                sentence_len = tokens_len * len(tokens)
                subwords_len.append(sentence_len)
            # breakpoint()
            dataset.add_field('subwords_len', subwords_len)
            # dataset.add_field('plm_spans', cache_bert[name]['plm_spans'])
            
        fields['bert'] = bert_field
        fields['bert_name'] = self.conf.bert
        
    def _dump(self, cache_path, to_cache):
        with open(cache_path, 'wb') as f:
            pickle.dump(to_cache, f)
    
    
def get_bert(bert_name, fix_len=20):
    from transformers import AutoTokenizer
    # print('bert name in data')
    # print(bert_name)
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_field = SubwordField(bert_name,
                        pad=tokenizer.pad_token,
                        unk=tokenizer.unk_token,
                        bos=tokenizer.cls_token or tokenizer.cls_token,
                        eos=tokenizer.sep_token or tokenizer.sep_token,
                        fix_len=fix_len,
                        tokenize=tokenizer.tokenize)
    bert_field.vocab = tokenizer.get_vocab()
    return bert_field