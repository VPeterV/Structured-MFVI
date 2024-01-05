from cProfile import label
import pdb
import torch
from torch import nn
import logging

log = logging.getLogger(__name__)
class Decoder(object):
    def __init__(self, conf, id2label):
        super(Decoder, self).__init__()
        
        self.conf = conf
        self.id2label = id2label
        self.scoring = conf.scoring.split('I')
        
    def graph2tag(self, ph_m, pt_m, span_label_scores):
        # breakpoint()

        # label_preds = ph_labels.argmax(-1)
        label_preds = span_label_scores.argmax(-1)

        
        bsz = ph_m.size(0)
        
        ph_arc = torch.where(ph_m > self.conf.threshold, 1, 0)
        pt_arc = torch.where(pt_m > self.conf.threshold, 1, 0)

        return label_preds, ph_arc, pt_arc
        
    def span_dp(self, span_m, span_arc_m,
                ph_m, pt_m, label_preds, 
                p, p_span_idx_mapping):
        """
        span_m: left closed, right open
        p_spans: left closed, right open
        """
        n = len(span_m)     # seq len, actually words number + 1
        max_spans = []
        tmp = {}
        opt = [0] * n      # i = 0,...,n-1, optimal span socres which ends with i
        
        # pdb.set_trace()
        
        for i in range(1, n):
            max_i = 0      # span ends with i
            for j in range(0, i):
                to_max = True
                if opt[j] >= max_i:
                        max_i = opt[j]
                        opt[i] = max_i
                        tmp[i] = j
                        to_max = False

                if (p, j+1, i) in p_span_idx_mapping:
                    span_idx = p_span_idx_mapping[(p, j+1, i)]
                    scores = 0.
                    if 's' in self.scoring:
                        scores += span_m[j+1][i]
                    if 'h' in self.scoring:
                        scores += ph_m[p][j+1]
                    if 't' in self.scoring:
                        scores += pt_m[p][i]
                    if 'p' in self.scoring:
                        scores += span_arc_m[p][span_idx]
                    if opt[j] + scores >= max_i:
                        max_i = opt[j] + scores
                        opt[i] = max_i
                        tmp[i] = (j+1, i)
                        to_max = False
                if to_max:
                    opt[i] = max_i
        
        idx = n - 1
        while idx > 0:
            if isinstance(tmp[idx], tuple):
                i, j = tmp[idx][0], tmp[idx][1]
                span_idx = p_span_idx_mapping[(p, i, j)]
                max_spans.append([p, i, j, self.id2label[label_preds[p][span_idx]]])
                idx = tmp[idx][0] - 1
            else:
                idx = tmp[idx]
        
        return max_spans
                
    def decode(self, words, span_scores, span_arc_scores, 
                ph_m, pt_m, spans_batch, label_preds, ph_arc, pt_arc):

        srl_results = []
        
        label_preds = label_preds.tolist()

        span_tuples = spans_batch
        
        # connect predicates with arguments
        ph_idx = torch.nonzero(ph_arc)
        ph_tuple = ph_idx.tolist()
        
        pt_idx = torch.nonzero(pt_arc)
        pt_tuple = pt_idx.tolist()
        
        head_span_dict = {}
        for idx, s in enumerate(span_tuples):
            # idx is the index of span (the 1st span, the 2nd span, etc.)
            # which is used to backtrack span_labels and span_arc_scores
            start, end = s[0], s[1]
            if start not in head_span_dict:
                head_span_dict[start] = []
            head_span_dict[start].append(
                    (start, end, idx)
            )
        
        p_spans = {}    # all argmax spans
        overlaping_prds = []
        p_span_idx_mapping = {}
        
        for ph in ph_tuple:
            p = ph[0]
            h = ph[1]
            
            if h not in head_span_dict:
                continue
            span_list = head_span_dict[h]
            
            for span in span_list:
                for pt in pt_tuple:
                    p_1 = pt[0]
                    t = pt[1]
                    if p == p_1 and h == span[0] and t == span[1]:
                        idx = span[2]
                        # left closed right closed
                        if p not in p_spans:
                            p_spans[p] = []
                        p_spans[p].append([p, h, t, self.id2label[label_preds[p][idx]]])
                        
                        assert (p, h, t) not in p_span_idx_mapping
                        p_span_idx_mapping[(p, h, t)] = idx
        # breakpoint()
        srl_results = []
        
        # pdb.set_trace()
        # checking overlapping
        for p in p_spans:
            is_overlapping = False
            pspan = p_spans[p]
            for i in range(len(pspan)-1):
                for j in range(i+1, len(pspan)):
                    hi, ti = pspan[i][1], pspan[i][2]
                    hj, tj = pspan[j][1], pspan[j][2]
                    if (hi >= hj and ti <= tj) or (hi <= hj and ti >= tj):
                        is_overlapping = True
                        overlaping_prds.append(p)
                        break
                if is_overlapping:
                    break
            if not is_overlapping:
                srl_results.extend(pspan)
        
        # only run dp among overlapping spans
        for p in overlaping_prds:
            max_spans = self.span_dp(span_scores.tolist(), 
                span_arc_scores.tolist(), ph_m.tolist(), pt_m.tolist(), 
                label_preds, p, p_span_idx_mapping)
            srl_results.extend(max_spans)

        return srl_results, p_spans

if __name__ == '__main__':
    # mat = torch.randn(5,5).triu_(1).gt(0.5).nonzero()
    # lab = torch.randint(0,5, size=(mat.shape[0],1))
    a = torch.randn(size = (5,5))
    label_preds = torch.arange(5)[None,:].expand(5,-1).tolist()
    
    # span_w_labels = torch.cat([mat, lab], dim = -1)
    span_w_labels = [[0,1,label_preds[0][1]],[1,2,label_preds[1][2]],[1,3,label_preds[1][3]]]
    id2label = {0:0, 1:1, 2:2, 3:3, 4:4}
    
    b = torch.rand(size=(5,5)).gt(0.5)
    c = torch.rand(size=(5,5)).gt(0.5)
    
    decoder = Decoder(None, id2label)
    res = decoder.decode(None, a, span_w_labels, label_preds, b, c)
    print(res)
    
