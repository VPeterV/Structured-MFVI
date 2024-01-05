from supar.utils.common import *
from typing import List, Dict
from torch import Tensor

def recover_tokens(tokens, fields, name):
    """
    Recover words or lemmas from idx to tokens
    """
    
    recovered_tokens = []
    pad_idx = fields.get_pad_index(name)
    for w_idx in tokens:
        if w_idx == pad_idx:
            break
        
        recovered_tokens.append(fields.get_name_from_id(name, w_idx))
    
    return recovered_tokens
    
def check_decode_mode(mode : str) -> None:
    supported_mode = ['dp-argmax','dp','dp-argmax-marginals', 'dp-marginals', 'span_repr']
    if mode not in supported_mode:
        raise NotImplementedError

def clean_gold_labels(golds, id2label, target_pad = -1):
    clean_golds = []
    for b_gold in golds:
        b_clean_gold = []
        for item in b_gold:
            if item[0] == -1:
                # if not rel_labes:
                assert item[0] == item[1] == item[2] == item[3] == target_pad
                # else:
                    # assert item[0] == item[1] == -1
                continue
            else:
                # breakpoint()
                if len(item) == 4:
                    b_clean_gold.append((item[0], item[1], item[2], id2label[item[3]]))
                elif len(item) == 5:
                    b_clean_gold.append((item[0], item[1], item[2], item[3], id2label[item[4]]))
                else:
                    raise ValueError
        clean_golds.append(b_clean_gold)
        
    return clean_golds
    
def pad_relation_spans(rel_results, p = -1):
    # import pdb
    # pdb.set_trace()
    max_len = max([len(rel) for rel in rel_results])
    max_len = 1 if max_len == 0 else max_len
    padded_results = []
    for rel in rel_results:
        pad_len = max_len - len(rel)
        padded_results.append(rel + [[p,p,p,p]] * pad_len)
    
    return padded_results
      
def recover_original_frames(frame_idx: Tensor, frame_labels: Tensor, id2label, target_pad: int = -1):
    
    bsz = frame_idx.size(0)
    original_frames = []
    
    frame_idx_list = frame_idx.tolist()
    frame_labels_list = frame_labels.tolist()
    
    for i in range(bsz):
        gold_f = []
        for j in range(len(frame_idx_list[i])):
            gold_iter = []
            span_idx_list = frame_idx_list[i][j]
            span_label = frame_labels_list[i][j]
            if span_idx_list[0] == target_pad:
                assert span_idx_list[1] == target_pad, breakpoint()
                assert span_label == target_pad, breakpoint()
                
                continue
            
            if len(span_idx_list) == 2:
                gold_iter.append((tuple(span_idx_list)))
            elif len(span_idx_list) == 4 and span_idx_list[2] != target_pad:
                gold_iter.append(((span_idx_list[0], span_idx_list[1]), (span_idx_list[2], span_idx_list[3])))
            elif len(span_idx_list) == 4 and span_idx_list[2] == target_pad:
                gold_iter.append(((span_idx_list[0], span_idx_list[1])))
            else:
                raise ValueError(f"Unexpected span index list: {span_idx_list}")

            gold_iter.append(id2label[span_label])
            
            gold_iter = tuple(gold_iter)
    
            gold_f.append(gold_iter)
            
        gold_f = tuple(gold_f)
                
        original_frames.append(gold_f)
    
    return original_frames
    
# def recover_original_frames(frame_idx: Tensor, frame_labels: Tensor, id2label, target_pad: int = -1):
    
#     bsz = frame_idx.size(0)
#     original_frames = []
    
#     frame_idx_list = frame_idx.tolist()
#     frame_labels_list = frame_labels.tolist()
    
#     for i in range(bsz):
#         gold_f = []
#         for j in range(len(frame_idx_list[i])):
#             gold_iter = []
#             span_idx_list = frame_idx_list[i][j]
#             span_label = frame_labels_list[i][j]
#             if span_idx_list[0][0] == target_pad:
#                 assert span_idx_list[0][1] == target_pad, breakpoint()
#                 assert span_label == target_pad, breakpoint()
                
#                 continue
                
#             gold_iter.append(span_idx_list)
#             gold_iter.append(id2label[span_label])
    
#             gold_f.append(gold_iter)
                
#         original_frames.append(gold_f)
    
#     return original_frames
    
def recover_frame_elements(frame_p_idx: Tensor, frame_r_idx: Tensor,
                frame_labels: Tensor, frame_id2label: Dict[int, str],
                role_id2label: Dict[int, str], target_pad: int = -1):
    
    bsz = frame_p_idx.size(0)
    frame_elements = []
    
    frame_p_idx_list = frame_p_idx.tolist()
    frame_r_idx_list = frame_r_idx.tolist()
    frame_labels_list = frame_labels.tolist()
    
    for i in range(bsz):
        gold_f = []
        for j in range(len(frame_p_idx_list[i])):
            
            gold_frame = []
            gold_p2r = []
            
            p_idx = frame_p_idx_list[i][j]
            r_idx = frame_r_idx_list[i][j]
            labels = frame_labels_list[i][j]
            
            if p_idx[0] == target_pad:
                assert p_idx[1] == r_idx[0] == r_idx[1] == target_pad, breakpoint()
                assert labels[0] == labels[1] == target_pad, breakpoint()
                
                continue
            
            if len(p_idx) == 2:
                gold_frame.append((tuple(p_idx)))
            elif len(p_idx) == 4 and p_idx[2] != target_pad:
                gold_frame.append(((p_idx[0], p_idx[1]), (p_idx[2], p_idx[3])))
            elif len(p_idx) == 4 and p_idx[2] == target_pad:
                gold_frame.append(((p_idx[0], p_idx[1])))
            else:
                raise ValueError(f"Unexpected span index list: {p_idx}")
                
            gold_frame.append(frame_id2label[labels[0]])
            
            gold_frame = tuple(gold_frame)
            
            gold_p2r.append(gold_frame)
            gold_p2r.extend(r_idx)
            gold_p2r.append(role_id2label[labels[1]])
            
            gold_p2r = tuple(gold_p2r)
            
            gold_f.append(gold_p2r)
            
        gold_f = tuple(gold_f)
        
        frame_elements.append(gold_f)
        
    return frame_elements

# def recover_frame_elements(frame_p_idx: Tensor, frame_r_idx: Tensor,
#                 frame_labels: Tensor, frame_id2label: Dict[int, str],
#                 role_id2label: Dict[int, str], target_pad: int = -1):
    
#     bsz = frame_p_idx.size(0)
#     frame_elements = []
    
#     frame_p_idx_list = frame_p_idx.tolist()
#     frame_r_idx_list = frame_r_idx.tolist()
#     frame_labels_list = frame_labels.tolist()
    
#     for i in range(bsz):
#         gold_f = []
#         for j in range(len(frame_p_idx_list[i])):
            
#             gold_frame = []
#             gold_p2r = []
            
#             p_idx = frame_p_idx_list[i][j]
#             r_idx = frame_r_idx_list[i][j]
#             labels = frame_labels_list[i][j]
            
#             if p_idx[0][0] == target_pad:
#                 assert p_idx[0][1] == r_idx[0] == r_idx[1] == target_pad, breakpoint()
#                 assert labels[0] == labels[1] == target_pad, breakpoint()
                
#                 continue
            
#             gold_frame.append(p_idx)
#             gold_frame.append(frame_id2label[labels[0]])
            
#             gold_p2r.append(gold_frame)
#             gold_p2r.append(r_idx)
#             gold_p2r.append(role_id2label[labels[1]])
            
#             gold_f.append(gold_p2r)
        
#         frame_elements.append(gold_f)
        
#     return frame_elements

if __name__ == '__main__':
    a = [[[-1,-1,-1],[0,1,0],[1,1,1]],[[-1,-1,-1],[-1,-1,-1]],[[0,0,1],[-1,-1,-1]]]
    c = {0:'a',1:'b'}
    b = clean_gold_labels(a,c)
    print(b)
    
    a = [[[-1,-1,-1, -1, -1],[0,1,0,1,1]],[[-1,-1,-1, -1, -1],[-1,-1,-1, -1, -1]],[[0,0,1,1,1],[-1,-1,-1,-1,-1]]]
    c = {0:'a',1:'b'}
    b = clean_gold_labels(a,c)
    print(b)
