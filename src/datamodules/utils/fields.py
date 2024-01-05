import torch
import numpy as np
from collections import Counter
from supar.utils.fn import pad
from supar.utils.vocab import Vocab
from typing import List

'''
    ref: https://github.com/yzhangcs/parser/blob/main/supar/utils/field.py
    Guess it may be based on Torchtext :)
'''

class RawField(object):
    r"""

    Defines a general datatype.
    A :class:`RawField` object does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequence):
        return self.preprocess(sequence)

class Field(RawField):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`~supar.utils.vocab.Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.
    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`~supar.utils.vocab.Vocab` object.
            If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, use_vocab=True, tokenize=None, fn=None, min_freq=1):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn
        self.min_freq = min_freq

        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    def __getstate__(self):
        # serialization
        state = dict(self.__dict__)
        if self.tokenize is None:
            state['tokenize_args'] = None
        elif self.tokenize.__module__.startswith('transformers'):
            state['tokenize_args'] = (self.tokenize.__module__, self.tokenize.__self__.name_or_path)
            state['tokenize'] = None
        return state

    def __setstate__(self, state):
        # deserialization
        tokenize_args = state.pop('tokenize_args', None)
        if tokenize_args is not None and tokenize_args[0].startswith('transformers'):
            from transformers import AutoTokenizer
            state['tokenize'] = AutoTokenizer.from_pretrained(tokenize_args[1]).tokenize
        self.__dict__.update(state)

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence):
        r"""
        Loads a single example using this field, tokenizing if necessary.
        The sequence will be first passed to ``fn`` if available.
        If ``tokenize`` is not None, the input will be tokenized.
        Then the input will be lowercased optionally.
        Args:
            sequence (list):
                The sequence to be preprocessed.
        Returns:
            A list of preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, sequences, embed=None):
        r"""
        Constructs a :class:`~supar.utils.vocab.Vocab` object for this field from the dataset.
        If the vocabulary has already existed, this function will have no effect.
        Args:
            dataset (Dataset):
                A :class:`~supar.utils.data.Dataset` object.
                One of the attributes should be named after the name of this field.
            min_freq (int):
                The minimum frequency needed to include a token in the vocabulary. Default: 1.
            embed (Embedding):
                An Embedding object, words in which will be extended to the vocabulary. Default: ``None``.
        """

        if hasattr(self, 'vocab'):
            return
        # sequences = getattr(dataset, self.name)
        counter = Counter(token
                          for seq in sequences
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequence):
        r"""
        Turns a list of sequences that use this field into tensors.
        Each sequence is first preprocessed and then numericalized if needed.
        Args:
            sequences (list[list[str]]):
                A list of sequences.
        Returns:
            A list of tensors transformed from the input sequences.
        """
        # import pdb
        # pdb.set_trace()
        
        sequence = self.preprocess(sequence)
        if self.use_vocab:
            sequence = self.vocab[sequence]
        if self.bos:
            sequence = [self.bos_index] + sequence
        if self.eos:
            sequence = sequence + [self.eos_index]
        #     sequences = [[self.bos_index] + seq for seq in sequences]
        # if self.eos:
        #     sequences = [seq + [self.eos_index] for seq in sequences]
        # sequences = [torch.tensor(seq) for seq in sequences]

        return sequence

    def compose(self, sequences):
        r"""
        Composes a batch of sequences into a padded tensor.
        Args:
            sequences (list[~torch.Tensor]):
                A list of tensors.
        Returns:
            A padded tensor converted to proper device.
        """

        return pad(sequences, self.pad_index).to(self.device)
        
# class SubwordField(Field):
#     r"""
#     A field that conducts tokenization and numericalization over each token rather the sequence.

#     This is customized for models requiring character/subword-level inputs, e.g., CharLSTM and BERT.

#     Args:
#         fix_len (int):
#             A fixed length that all subword pieces will be padded to.
#             This is used for truncating the subword pieces that exceed the length.
#             To save the memory, the final length will be the smaller value
#             between the max length of subword pieces in a batch and `fix_len`.

#     Examples:
#         >>> from transformers import AutoTokenizer
#         >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
#         >>> field = SubwordField('bert',
#                                  pad=tokenizer.pad_token,
#                                  unk=tokenizer.unk_token,
#                                  bos=tokenizer.cls_token,
#                                  eos=tokenizer.sep_token,
#                                  fix_len=20,
#                                  tokenize=tokenizer.tokenize)
#         >>> field.vocab = tokenizer.get_vocab()  # no need to re-build the vocab
#         >>> field.transform([['This', 'field', 'performs', 'token-level', 'tokenization']])[0]
#         tensor([[  101,     0,     0],
#                 [ 1188,     0,     0],
#                 [ 1768,     0,     0],
#                 [10383,     0,     0],
#                 [22559,   118,  1634],
#                 [22559,  2734,     0],
#                 [  102,     0,     0]])
#     """

#     def __init__(self, *args, **kwargs):
#         self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
#         super().__init__(*args, **kwargs)

#     def build(self, sequences):
#         if hasattr(self, 'vocab'):
#             return
#         counter = Counter(piece
#                           for seq in sequences
#                           for token in seq
#                           for piece in self.preprocess(token))
#         self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)
        
#     def transform(self, seq):
#         seq = [self.preprocess(token) for token in seq]

#         if self.fix_len <= 0:
#             self.fix_len = max(len(token) for token in seq)
            
#         if self.use_vocab:
#             seq =  [  [self.vocab[i] if i in self.vocab else self.unk_index for i in token] if token else [self.unk_index]
#                  for token in seq]

#         if self.bos:
#             seq = [[self.bos_index] ] + seq

#         if self.eos:
#             seq = seq + [[self.eos_index]]

#         l = min(self.fix_len, max(len(ids) for ids in seq))
#         seq = [ids[: l] for ids in seq]
#         return seq

class SubwordField(Field):
    r"""
    A field that conducts tokenization and numericalization over each token rather the sequence.
    This is customized for models requiring character/subword-level inputs, e.g., CharLSTM and BERT.
    Args:
        fix_len (int):
            A fixed length that all subword pieces will be padded to.
            This is used for truncating the subword pieces that exceed the length.
            To save the memory, the final length will be the smaller value
            between the max length of subword pieces in a batch and `fix_len`.
    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> field = SubwordField('bert',
                                 pad=tokenizer.pad_token,
                                 unk=tokenizer.unk_token,
                                 bos=tokenizer.cls_token,
                                 eos=tokenizer.sep_token,
                                 fix_len=20,
                                 tokenize=tokenizer.tokenize)
        >>> field.vocab = tokenizer.get_vocab()  # no need to re-build the vocab
        >>> field.transform([['This', 'field', 'performs', 'token-level', 'tokenization']])[0]
        tensor([[  101,     0,     0],
                [ 1188,     0,     0],
                [ 1768,     0,     0],
                [10383,     0,     0],
                [22559,   118,  1634],
                [22559,  2734,     0],
                [  102,     0,     0]])
    """

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        if self.fix_len == -1:
            self.fix_len = 100000000
        super().__init__(*args, **kwargs)

    def build(self, sequences):
        if hasattr(self, 'vocab'):
            return
            
        counter = Counter(piece
                          for seq in sequences
                          for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)

    def transform(self, seq):
        # import pdb
        # pdb.set_trace()
        seq = [self.preprocess(token) for token in seq]

        if self.use_vocab:
            seq =  [  [self.vocab[i] if i in self.vocab else self.unk_index for i in token] if token else [self.unk_index]
                 for token in seq]

        if self.bos:
            seq = [[self.bos_index] ] + seq

        if self.eos:
            seq = seq + [[self.eos_index]]

        l = min(self.fix_len, max(len(ids) for ids in seq))
        seq = [ids[: l] for ids in seq]
        return seq
        
#     def modify_spans(self, subwords, spans):
#         # ref PURE https://github.com/princeton-nlp/PURE/blob/b1e9cad39bec10eb3c355dc5a8e4e75dd0afebf5/entity/models.py#L194
#         start2idx = []
#         end2idx = []
#         plm_tokens = []
#         plm_tokens.append(subwords[0])
        
#         for sub_token in subwords[1:]:
#             start2idx.append(len(plm_tokens))
#             plm_tokens += sub_token
#             end2idx.append(len(plm_tokens)-1)
            
#         plm_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        
#         return plm_spans
        
class SpanLabelField(Field):
    def __init__(self, *args, **kwargs):
        self.field_name = args[0]
        self.dataname = kwargs.pop('dataname')
        self.golden_predicate = kwargs.pop('golden_predicate', None)
        self.task = kwargs.pop('task', "srl")
        self.target_pad = kwargs.pop('target_pad')
        self.withnull = kwargs.pop('withnull', False)
        self.add_none = kwargs.pop('add_none', False)
        super(SpanLabelField, self).__init__(*args, **kwargs)
        
    def build(self, dataset, dev_dataset = None, test_dataset = None):
        all_labels = []
        for example in dataset:
            for label in example:
                if label != self.target_pad:
                    all_labels.append(label)

        if dev_dataset is not None and not(self.dataname.lower() == 'conll12' and self.add_none):
            for example in dev_dataset:
                for label in example:
                    if label != self.target_pad:
                        all_labels.append(label)
                        
        if test_dataset is not None and not (self.dataname.lower() == 'conll12' and self.add_none):
            for example in test_dataset:
                for label in example:
                    if label != self.target_pad:
                        all_labels.append(label)
        
        orl_dse_starts = 0
        label2id = Counter(all_labels)
        self.label2id = {}
        self.id2label = {}
        for i, (key, _) in enumerate(label2id.most_common()):
            ids = i
            if self.task == 'orl' and key =='DSE':
                orl_dse_starts = i
                continue
            
            if self.task == 'orl' and i > orl_dse_starts:
                ids -= 1

            self.label2id[key] = ids
            self.id2label[ids] = key
        #TODO Temporarily fix
        cnt = len(self.label2id)

        if self.dataname.lower() == 'conll12' and self.add_none:
            self.id2label[cnt] = 'None'
            self.label2id['None'] = cnt
        
        if self.task == 'orl':
            self.id2label[cnt] = 'DSE'
            self.label2id['DSE'] = cnt

        num_label = len(self.label2id)
        # if not self.golden_predicate:
        #     self.label2id['predicate'] = num_label
        #     self.id2label[num_label] = 'predicate'
        
    def transform(self, span_label_seq):
        idx_span_labels = []
        for item in span_label_seq:
            if item != self.target_pad:
                try:
                    idx_span_labels.append(self.label2id[item])
                except KeyError:
                    # breakpoint()
                    idx_span_labels.append(self.label2id['None'])
            else:
                idx_span_labels.append(item)
        
        return idx_span_labels
        
    @property
    def num_span_labels(self):
        return len(self.label2id)
        
class FrameNetLabelField(SpanLabelField):
    def __init__(self, *args, **kwargs):
        super(FrameNetLabelField, self).__init__(*args, **kwargs)
        self.field_name = args[0]
        self.dataname = kwargs.pop('dataname')
        self.target_pad = kwargs.pop('target_pad')
        

        
    def build(self, dataset, dev_dataset = None, test_dataset = None, label2id = None, id2label = None):
    
    
        assert (label2id and id2label) or (label2id is None and id2label is None),\
        "Please offer both of label2id and id2label or ignore both of them"
        
        if label2id is None:
            all_labels = []
            for example in dataset:
                for label in example:
                    if label != self.target_pad:
                        all_labels.append(label)
            
            if dev_dataset is not None:
                for example in dev_dataset:
                    for label in example:
                        if label != self.target_pad:
                            all_labels.append(label)
                            
            if test_dataset is not None:
                for example in test_dataset:
                    for label in example:
                        if label != self.target_pad:
                            all_labels.append(label)
            
            label2id = Counter(all_labels)
            self.label2id = {}
            self.id2label = {}
            for i, (key, _) in enumerate(label2id.most_common()):
                # if "p2r" in self.field_name and self.withnull:
                #     ids = i + 1
                # else:
                ids = i
    
                self.label2id[key] = ids
                self.id2label[ids] = key
    
            num_label = len(self.label2id)
            # breakpoint()
            
            # if "p2r" in self.field_name and self.withnull:
            #     self.label2id["O"] = 0
            #     self.id2label[0] = "O"
        else:
            self.label2id = label2id
            self.id2label = id2label
            
    @property
    def num_span_labels(self):
        return len(self.label2id)
            
class OrignalFrameLabelField(FrameNetLabelField):
    def __init__(self, *args, **kwargs):
        super(OrignalFrameLabelField, self).__init__(*args, **kwargs)
        
    def build(self, frame_label2id, frame_id2label, role_label2id, role_id2label):
        
        self.frame_label2id = frame_label2id
        self.frame_id2label = frame_id2label
        
        self.role_label2id = role_label2id
        self.role_id2label = role_id2label
        
    def transform(self, frame_label_sequence):
        idx_frame_labels = []
        
        for item in frame_label_sequence:
            assert isinstance(item, List), breakpoint()
            if item[0] != self.target_pad and item[1] != self.target_pad:
                idx_frame_labels.append([self.frame_label2id[item[0]], self.role_label2id[item[1]]])
            else:
                idx_frame_labels.append(item)
                
        return idx_frame_labels

        
class IndField(Field):
    def __init__(self, *args, **kwargs):
        super(IndField, self).__init__(*args, **kwargs)
        
    def build(self, dataset):
        self.vocab = {0:0, 1:1}
        
    def transform(self, sequence):
        
        sequence = [0] + sequence
        sequence = sequence + [0]
        
        return sequence