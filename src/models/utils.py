import pdb
import torch
from src.datamodules.utils.const import FRAMENET_PRD_IDX, FRAMENET_ROLE_IDX

def logspace(x, eps):
    return (x + eps).log()

def potential_norm(span_logits: torch.Tensor, var = 1.):
    s_std, s_mean = torch.std_mean(span_logits, list(range(1, span_logits.ndim)), keepdim=True)
    log_s = span_logits - s_mean
    log_s: torch.Tensor = log_s / ((s_std + 1e-9) / var)
    span_logits = log_s
    return span_logits
    
def structure_smoothing(span_logits: torch.Tensor, span_mask: torch.Tensor, 
                        span_ind: torch.Tensor, ratio: float) -> torch.Tensor:
    # pdb.set_trace()
    
    v = 1 - ratio
    mask_aux = span_mask.new_ones((*span_mask.shape,2)).float()
    mask_aux.masked_fill_(span_mask[...,None], v)
    mask_aux[..., 0].masked_fill_(span_ind, v)
    mask_aux[..., 1].masked_fill_(~span_mask & ~span_ind, v)
    
    span_logits = span_logits + torch.log(mask_aux + 1e-10)
    # mask_aux_cross = span_mask.new_zeros((*span_mask.shape,2)).float()
    # mask_aux_cross = mask_aux_cross + (~span_mask[..., None])
    # span_logits = span_logits + torch.log(v * mask_aux_cross + 1e-10)
    
    # mask_aux_latent = span_ind.new_zeros((*span_ind, 2)).float()
    # mask_aux_latent[..., 0] = mask_aux_latent[..., 0] + span_ind
    # span_logits = span_logits + torch.log(v * mask_aux_latent + 1e-10)
    
    # mask_aux_ob = span_ind.new_zeros((*span_ind, 2)).float()
    # mask_aux_ob[..., 1] = mask_aux_ob[..., 1] + (~span_mask & ~span_ind)
    # span_logits = span_logits + torch.log(v * mask_aux_ob + 1e-10)
    
    return span_logits
    
def unfold_framenet_graphs(gold_spans, predicates, roles, p2r, frames, p2r_labels,
                        bert_seqlens, t, num_role_labels, seq_lens = None, so = False, fencepost = False, span_encoding = False):
    
    """
    fencepost: Span edge is obtained from fencepost representations
    """
    
    bsz = gold_spans.shape[0]
    
    word_mask = get_word_mask(bert_seqlens, t, fencepost = False)
    word_mask_span = get_word_mask(seq_lens, t, fencepost = True) if seq_lens is not None and fencepost else word_mask
    
    mask = word_mask_span.unsqueeze(1) & word_mask_span.unsqueeze(2)
    maskspan = mask.triu(1) if fencepost else mask.triu()     # left closed, right away
    
    maskarc = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
    # maskarc[:, torch.arange(t), torch.arange(t)] = False
    
    # pss_aux, pps_aux = None, None
    # if wprd:
    #     # given golden predicates
    #     maskarc_aux = unfold_golden_predicates(predicates, t)
    #     maskarc = maskarc & maskarc_aux
        
    #     pss_aux = unfold_so_golden_predicates(predicates, t)
    #     pps_aux = unfold_so_golden_predicates(predicates, t, predicates)
    
    mask2o_p2rspan, mask2o_pspan2r, mask2o_psib, mask2o_pcop = None, None, None, None
    if so:
        # three kinds of mask: p,s_h,s_t & p, p, s_h/s_t & p, s_h/s_t, s_h/s_t
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        # p, s_h, s_t
        mask2o_pspan = mask2o
        # mask2o_pspan.diagonal(dim1=1, dim2=2).fill_(False)
        # mask2o_pspan.diagonal(dim1=1, dim2=3).fill_(False)
        # p, s_h/s_t, s_h/s_t or p, p, s_h/s_t
        mask2o_psib = mask2o_pspan.clone()
        mask2o_psib.diagonal(dim1=-1,dim2=-2).fill_(False)
        
        mask2o_pcop = mask2o_pspan.clone()
        mask2o_pcop.diagonal(dim1=1,dim2=2).fill_(False)
        
        # pdb.set_trace()
        mask2o_p2rspan = mask2o_pspan.triu()
        mask2o_pspan2r = mask2o_p2rspan.movedim(1, -1)
        
        
        # mask2o_pspan = mask2o_pspan & pss_aux if pss_aux is not None else mask2o_pspan
        # mask2o_psib = mask2o_psib & pss_aux if pss_aux is not None else mask2o_psib
        # mask2o_pcop = mask2o_pcop & pps_aux if pps_aux is not None else mask2o_pcop

    # gold_spans
    spans_ind = unfold_span_edge(gold_spans[...,0], gold_spans[...,1], t, fencepost)
    spans_ind = spans_ind * maskspan.to(spans_ind.dtype)
    
    # predicates and their frames
    prd_ind = unfold_span_edge(predicates[..., 0], predicates[..., 1], t, fencepost)
    prd_ind = prd_ind * maskspan.to(prd_ind.dtype)
    
    prd_frames = unfold_frame(predicates[..., 0], predicates[..., 1], frames, t, fencepost)
    
    # span types
    span_prd_type = unfold_ml_span_types(predicates[..., 0], predicates[..., 1], t, FRAMENET_PRD_IDX, fencepost)
    span_role_type = unfold_ml_span_types(roles[..., 0], roles[..., 1], t, FRAMENET_ROLE_IDX, fencepost)
    span_types = span_prd_type + span_role_type
    
    # p2r    
    t2t_arc_label = unfold_ml_label(p2r[..., 1], p2r[..., 3], p2r_labels, num_role_labels, t)
    h2h_arc_label = unfold_ml_label(p2r[..., 0], p2r[..., 2], p2r_labels, num_role_labels, t)
        
    h2h_arc = h2h_arc_label.any(-1)
    t2t_arc = t2t_arc_label.any(-1)
    # breakpoint()
    # return fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop
    return {"maskspan": maskspan, "maskarc": maskarc, "word_maskspan": word_mask_span, "word_mask": word_mask,\
            "mask2o_p2rspan": mask2o_p2rspan, "mask2o_pspan2r": mask2o_pspan2r, "mask2o_psib": mask2o_psib, "mask2o_pcop": mask2o_pcop},\
            {"spans_ind": spans_ind, "h2h_arc_label": h2h_arc_label, "t2t_arc_label": t2t_arc_label,\
            "prd_frames": prd_frames, "span_types": span_types, "prd_ind": prd_ind,  "h2h_arc": h2h_arc, "t2t_arc": t2t_arc}
    
def unfold_orl_graphs(gold_spans, expressions, nouns, labels,
                        bert_seqlens, t, wprd, seq_lens = None, num_labels = 2, so = False, fencepost = False):
    
    """
    fencepost: Span edge is obtained from fencepost representations
    """
    
    bsz = gold_spans.shape[0]
    
    word_mask = get_word_mask(bert_seqlens, t, fencepost = False)
    word_mask_span = get_word_mask(seq_lens, t, fencepost = True) if seq_lens is not None and fencepost else word_mask
    
    mask = word_mask_span.unsqueeze(1) & word_mask_span.unsqueeze(2)
    maskspan = mask.triu(1) if fencepost else mask.triu()     # left closed, right away
    
    maskarc = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
    maskarc[:, torch.arange(t), torch.arange(t)] = False

    spans_ind = unfold_span_edge(gold_spans[...,0], gold_spans[...,1], t, fencepost)
    # breakpoint()
    spans_ind = spans_ind * maskspan.to(spans_ind.dtype)
    
    # pdb.set_trace()
    # h2h_arc = unfold_arc(expressions[..., 0], nouns[..., 0], t)
    # t2t_arc = unfold_arc(expressions[..., 1], nouns[..., 1], t)
    
    h2h_arc_label = unfold_ml_label(expressions[..., 0], nouns[..., 0], labels, num_labels, t)
    t2t_arc_label = unfold_ml_label(expressions[..., 1], nouns[..., 1], labels, num_labels, t)
    
    h2h_arc = h2h_arc_label.any(-1)
    t2t_arc = t2t_arc_label.any(-1)
    
    # breakpoint()
    # fo_ind = [spans_ind, ph_ind, pt_ind]   # first-order indicator

    # return fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop
    return {"maskspan": maskspan, "maskarc": maskarc, "word_maskspan": word_mask_span, "word_mask": word_mask},\
            {"spans_ind": spans_ind, "h2h_arc_label": h2h_arc_label, "t2t_arc_label": t2t_arc_label, "h2h_arc": h2h_arc, "t2t_arc": t2t_arc}
    
def unfold_srl_graphs(predicates, gold_spans, seq_lens, t, wprd, so = False, predicted_prd = None):
    
    bsz = predicates.shape[0]
    word_mask = get_word_mask(seq_lens, t, fencepost = False)
    
    mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
    maskspan = mask.triu()     # left closed, right away
    
    maskarc = mask.clone()
    maskarc[:, torch.arange(t), torch.arange(t)] = False
    
    pss_aux, pps_aux = None, None
    if wprd:
        # given golden predicates
        maskarc_aux = unfold_golden_predicates(predicates, t)
        maskarc = maskarc & maskarc_aux
        
        pss_aux = unfold_so_golden_predicates(predicates, t)
        pps_aux = unfold_so_golden_predicates(predicates, t, predicates)
    
    else:
        if predicted_prd is not None:
            maskarc_aux = unfold_predicted_predicates(predicted_prd)
            maskarc = maskarc & maskarc_aux
            
            pss_aux = unfold_so_predicted_predicates(predicted_prd)
            pps_aux = unfold_so_predicted_predicates(predicted_prd, predicted_prd)
    
    mask2o_pspan, mask2o_psib, mask2o_pcop = None, None, None
    if so:
        # three kinds of mask: p,s_h,s_t & p, p, s_h/s_t & p, s_h/s_t, s_h/s_t
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        # p, s_h, s_t
        mask2o_pspan = mask2o
        mask2o_pspan.diagonal(dim1=1, dim2=2).fill_(False)
        mask2o_pspan.diagonal(dim1=1, dim2=3).fill_(False)
        # p, s_h/s_t, s_h/s_t or p, p, s_h/s_t
        mask2o_psib = mask2o_pspan.clone()
        mask2o_psib.diagonal(dim1=-1,dim2=-2).fill_(False)
        
        mask2o_pcop = mask2o_psib.clone()
        
        # pdb.set_trace()
        mask2o_pspan = mask2o_pspan.triu()
        
        mask2o_pspan = mask2o_pspan & pss_aux if pss_aux is not None else mask2o_pspan
        mask2o_psib = mask2o_psib & pss_aux if pss_aux is not None else mask2o_psib
        mask2o_pcop = mask2o_pcop & pps_aux if pps_aux is not None else mask2o_pcop

    spans_ind = unfold_span_edge_fp(gold_spans[...,0], gold_spans[...,1], t, maskspan)
    spans_ind = spans_ind * maskspan.to(spans_ind.dtype)
    
    # pdb.set_trace()
    ph_ind = unfold_arc_fp(predicates, gold_spans[...,0], t)
    pt_ind = unfold_arc_fp(predicates, gold_spans[...,1], t)
    
    # fo_ind = [spans_ind, ph_ind, pt_ind]   # first-order indicator

    return {"mask":mask, "maskarc":maskarc, "maskspan":maskspan, \
            "mask2o_pspan": mask2o_pspan, "mask2o_psib": mask2o_psib, "mask2o_pcop": mask2o_pcop},\
            {"spans_ind": spans_ind, "ph_ind": ph_ind, "pt_ind": pt_ind}
    
def unfold_graphs(predicates, gold_spans, seq_lens, t, wprd, so = False, predicted_prd = None):
    
    bsz = predicates.shape[0]
    word_mask = get_word_mask(seq_lens, t, fencepost = False)
    
    mask = word_mask.unsqueeze(1) & word_mask.unsqueeze(2)
    maskspan = mask.triu()     # left closed, right away
    
    maskarc = mask.clone()
    maskarc[:, torch.arange(t), torch.arange(t)] = False
    
    pss_aux, pps_aux = None, None
    if wprd:
        # given golden predicates
        maskarc_aux = unfold_golden_predicates(predicates, t)
        maskarc = maskarc & maskarc_aux
        
        pss_aux = unfold_so_golden_predicates(predicates, t)
        pps_aux = unfold_so_golden_predicates(predicates, t, predicates)
    
    else:
        if predicted_prd is not None:
            maskarc_aux = unfold_predicted_predicates(predicted_prd)
            maskarc = maskarc & maskarc_aux
            
            pss_aux = unfold_so_predicted_predicates(predicted_prd)
            pps_aux = unfold_so_predicted_predicates(predicted_prd, predicted_prd)
    
    mask2o_pspan, mask2o_psib, mask2o_pcop = None, None, None
    if so:
        # three kinds of mask: p,s_h,s_t & p, p, s_h/s_t & p, s_h/s_t, s_h/s_t
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        # p, s_h, s_t
        mask2o_pspan = mask2o
        mask2o_pspan.diagonal(dim1=1, dim2=2).fill_(False)
        mask2o_pspan.diagonal(dim1=1, dim2=3).fill_(False)
        # p, s_h/s_t, s_h/s_t or p, p, s_h/s_t
        mask2o_psib = mask2o_pspan.clone()
        mask2o_psib.diagonal(dim1=-1,dim2=-2).fill_(False)
        
        mask2o_pcop = mask2o_psib.clone()
        
        # pdb.set_trace()
        mask2o_pspan = mask2o_pspan.triu()
        
        mask2o_pspan = mask2o_pspan & pss_aux if pss_aux is not None else mask2o_pspan
        mask2o_psib = mask2o_psib & pss_aux if pss_aux is not None else mask2o_psib
        mask2o_pcop = mask2o_pcop & pps_aux if pps_aux is not None else mask2o_pcop

    spans_ind = unfold_span_edge_fp(gold_spans[...,0], gold_spans[...,1], t, maskspan)
    spans_ind = spans_ind * maskspan.to(spans_ind.dtype)
    
    # pdb.set_trace()
    ph_ind = unfold_arc_fp(predicates, gold_spans[...,0], t)
    pt_ind = unfold_arc_fp(predicates, gold_spans[...,1], t)
    
    fo_ind = [spans_ind, ph_ind, pt_ind]   # first-order indicator

    return fo_ind, mask, maskarc, maskspan, mask2o_pspan, mask2o_psib, mask2o_pcop

def get_word_mask(seq_lens, t, fencepost = False):
    bsz = seq_lens.size(0)
    # start = 1 if fencepost else 0
    plus = 0 if fencepost else 1
    offset = 1 if fencepost else 0
    word_mask = torch.zeros((bsz, t + plus), device = seq_lens.device)
    word_mask[torch.arange(bsz), seq_lens + offset] = 1
    word_mask = 1 - word_mask.cumsum(dim=-1)
    word_mask = word_mask.bool()
    word_mask = word_mask[:, :-1]
    
    return word_mask
    
def unfold_span(predicates, spans, label, t, n_span_label, p_as_span = False):
    # pdb.set_trace()
    bsz = spans.shape[0]
    unfolded_mat = label.new_zeros(bsz, t+1, t+1)
    unfolded_mat[torch.arange(bsz)[:,None], spans[:,:,0], spans[:,:,1]] = label
    if p_as_span:
        unfolded_mat[torch.arange(bsz)[:,None], predicates, predicates] = n_span_label - 1
    
    return unfolded_mat[:, :-1, :-1]    # remove padded

def unfold_ml_label(x, y, labels, num_labels, t, fencepost = False):
    if fencepost:
        raise NotImplementedError
    
    bsz = x.size(0)
    
    plus = 0 if fencepost else 1
    
    arc_labels = torch.zeros((bsz, t + plus, t + plus, num_labels + 1), device = x.device)
    arc_labels[torch.arange(bsz)[:, None], x, y, labels] = 1
    
    return arc_labels[:, :-1, :-1, :-1]
    
def unfold_span_label(prds, span_labels, t):
    
    bsz = prds.size(0)
    
    num_span = span_labels.size(1)
    span_nums = (prds != -1).sum(-1)
    
    unfolded_mat = prds.new_zeros(bsz, t + 1, num_span)
    
    span_idx = torch.arange(num_span)[None, :].expand_as(prds).to(prds.device)
    
    unfolded_mat[torch.arange(bsz)[:, None], prds, span_idx] = span_labels
    
    unfolded_mat = unfolded_mat[:, :-1]
    
    # mask = unfold_span_label_mask(prds, span_labels, unfolded_mat, t)
    # strictly, it is an indicator rather than mask
    # ind_label = (unfolded_mat != -1)
    ind_span = prds.new_zeros(bsz, t + 1, num_span)
    ind_span[torch.arange(bsz)[:, None], prds, span_idx] = 1
    ind_span = ind_span[:, :-1].bool()
    
    mask = (span_labels != -1)[:, None, :].expand(-1, t, -1)
    
    indicator = mask & ind_span
    
    gold_prds = prds.new_zeros(bsz, t + 1)
    gold_prds[torch.arange(bsz)[:, None], prds] = 1
    # breakpoint()
    
    return unfolded_mat, indicator, mask, gold_prds[:, :-1]
    
def unfold_span_label(prds, span_labels, t):
    
    bsz = prds.size(0)
    
    num_span = span_labels.size(1)
    span_nums = (prds != -1).sum(-1)
    
    unfolded_mat = prds.new_zeros(bsz, t + 1, num_span)
    
    span_idx = torch.arange(num_span)[None, :].expand_as(prds).to(prds.device)
    
    unfolded_mat[torch.arange(bsz)[:, None], prds, span_idx] = span_labels
    
    unfolded_mat = unfolded_mat[:, :-1]
    
    # mask = unfold_span_label_mask(prds, span_labels, unfolded_mat, t)
    # strictly, it is an indicator rather than mask
    # ind_label = (unfolded_mat != -1)
    ind_span = prds.new_zeros(bsz, t + 1, num_span)
    ind_span[torch.arange(bsz)[:, None], prds, span_idx] = 1
    ind_span = ind_span[:, :-1].bool()
    
    mask = (span_labels != -1)[:, None, :].expand(-1, t, -1)
    
    indicator = mask & ind_span
    
    gold_prds = prds.new_zeros(bsz, t + 1)
    gold_prds[torch.arange(bsz)[:, None], prds] = 1
    # breakpoint()
    
    return unfolded_mat, indicator, mask, gold_prds[:, :-1]

def unfold_pspan_graph(prds, gold_spans, t, fencepost = False):
    bsz = prds.shape[0]
    
    plus = 0 if fencepost else 1
    unfolded_mat = prds.new_zeros(bsz, t + plus, t + plus, t + plus)
    unfolded_mat[torch.arange(bsz)[:,None], prds, gold_spans[..., 0], gold_spans[..., 1]] = 1

    return unfolded_mat[:, :-1, :-1, :-1]

def unfold_arc_label(prds, arc_node, label, t, fencepost = False, ensemble_label = False):
    # pdb.set_trace()
    bsz = prds.shape[0]
    # left closed, right away for fencepost
    plus = 0 if fencepost else 1
    unfolded_mat = label.new_zeros(bsz, t + plus, t + plus)
    unfolded_mat[torch.arange(bsz)[:,None], prds, arc_node] = label

    return unfolded_mat[:, :-1, :-1]    # remove padded
    
def unfold_tuple_label(prds, span_node, label, t, fencepost = False, ensemble_label = False):
    # pdb.set_trace()
    bsz = prds.shape[0]
    # left closed, right away for fencepost
    plus = 0 if fencepost else 1
    unfolded_mat = label.new_zeros(bsz, t + plus, t + plus, t + plus)
    unfolded_mat[torch.arange(bsz)[:,None], prds, span_node[..., 0], span_node[..., 1]] = label
    
    unfolded_mat_mask = label.new_zeros(bsz, t + plus, t + plus, t + plus)
    unfolded_mat_mask[torch.arange(bsz)[:,None], prds, span_node[..., 0], span_node[..., 1]] = 1
    

    return unfolded_mat[:, :-1, :-1, :-1], unfolded_mat_mask[:, :-1, :-1, :-1]    # remove padded

def unfold_arc(x, y, t, fencepost = False):
    bsz = x.shape[0]
    
    plus = 0 if fencepost else 1
    unfolded_mat = torch.zeros(size = (bsz, t + plus, t + plus), device=x.device, dtype=x.dtype)
    
    unfolded_mat[torch.arange(bsz)[:,None], x, y] = 1
    
    return unfolded_mat[:, :-1, :-1]    # remove padded

def unfold_arc_with_root(x, y, t):
    """
    add root -> prds
    x: prds
    y: span
    """
    bsz = x.shape[0]
    unfolded_mat = torch.zeros(size = (bsz, t+1, t+1), device=x.device, dtype=x.dtype)
    unfolded_mat[torch.arange(bsz)[:, None], 0, x] = 1
    unfolded_mat[torch.arange(bsz)[:, None], x, y[..., 0]] = 1
    unfolded_mat[torch.arange(bsz)[:, None], x, y[..., 1]] = 1
    
    return unfolded_mat[:, :-1, :-1]    # remove padded
    
def unfold_arc_fp(x, y, t):
    """
    x: prds
    y: span heads
    """
    bsz = x.shape[0]
    unfolded_mat = torch.zeros(size = (bsz, t+1, t+1), device=x.device, dtype=x.dtype)
    # pdb.set_trace()
    # unfolded_mat[torch.arange(bsz)[:, None], 0, x] = 1
    unfolded_mat[torch.arange(bsz)[:, None], x, y] = 1
    
    return unfolded_mat[:, :-1, :-1]    # remove padded

def unfold_golden_predicates_seq(x, t):
    bsz = x.shape[0]
    unfolded_vec = x.new_zeros(size = (bsz, t+1)).float()

    unfolded_vec[torch.arange(bsz)[:, None], x] = 1
    
    return unfolded_vec[:, :-1]
    
def unfold_golden_predicates(x, t):
    bsz = x.size(0)
    unfolded_mat = x.new_zeros(size = (bsz, t+1, t+1)).bool()
    
    unfolded_mat[torch.arange(bsz)[:, None], x] = True
    
    return unfolded_mat[:, :-1, :-1]
    
def unfold_so_golden_predicates(x, t, y = None):
    bsz = x.size(0)
    unfolded_mat = x.new_zeros((bsz, t+1, t+1, t+1)).bool()
    # pdb.set_trace()
    if y is None:
        # for p, s, s
        unfolded_mat[torch.arange(bsz)[:,None], x] = True
    else:
        # for p, p, s
        # Is there any issue? I have checked it. It is okay :)
        unfolded_mat[torch.arange(bsz)[:,None, None], x[:,:,None], y[:, None, :]] = True
    
    return unfolded_mat[:, :-1, :-1, :-1]
    
def unfold_predicted_predicates(x):
    t = x.size(-1)
    return x.unsqueeze(-1).expand(-1, -1, t).bool()
    
def unfold_so_predicted_predicates(x, y = None):
    bsz = x.size(0)
    t = x.size(-1)
    # pdb.set_trace()
    if y is None:
        # for p, s, s
        return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t, t).bool()
    else:
        # for p, p, s
        # Is there any issue? I have checked it. It is okay :)
        x_ = x.unsqueeze(-1).expand(-1, -1, t).bool()
        y_ = y.unsqueeze(-2).expand(-1, t, -1).bool()
        
        return (x_ & y_).unsqueeze(-1).expand(-1, -1, -1, t)

def unfold_span_edge(x, y, t, fencepost = False):
    bsz = x.shape[0]
    plus = 0 if fencepost else 1
    
    unfolded_mat = torch.empty(size= (bsz, t + plus, t + plus), device = x.device, dtype = x.dtype).fill_(0)
    unfolded_mat[torch.arange(bsz)[:,None], x, y] = 1

    return unfolded_mat[:, :-1, :-1]
    
def unfold_frame(x, y, f, t, fencepost = False):
    bsz = x.shape[0]
    plus = 0 if fencepost else 1
    
    unfolded_mat = torch.empty(size= (bsz, t + plus, t + plus), device = x.device, dtype = x.dtype).fill_(0)
    unfolded_mat[torch.arange(bsz)[:,None], x, y] = f

    return unfolded_mat[:, :-1, :-1]

def unfold_span_edge_fp(x, y, t, maskspan):
    bsz = x.shape[0]
    unfolded_mat = torch.empty(size = (bsz, t+1, t+1), device=x.device, dtype=x.dtype).fill_(1)
    
    unfolded_mat[torch.arange(bsz)[:,None], x, y] = 2
        
    return unfolded_mat[:, :-1, :-1]    # remove padded
    
def unfold_ml_span_types(x, y, t, label_idx, fencepost = False):
    bsz = x.shape[0]
    plus = 0 if fencepost else 1
    
    unfolded_mat = torch.empty(size= (bsz, t + plus, t + plus, 2), device = x.device, dtype = x.dtype).fill_(0)
    unfolded_mat[torch.arange(bsz)[:,None], x, y, label_idx] = 1

    return unfolded_mat[:, :-1, :-1]

def unfold_span_edge_fp(x, y, t, maskspan):
    bsz = x.shape[0]
    unfolded_mat = torch.empty(size = (bsz, t+1, t+1), device=x.device, dtype=x.dtype).fill_(1)
    
    unfolded_mat[torch.arange(bsz)[:,None], x, y] = 2
        
    return unfolded_mat[:, :-1, :-1]    # remove padded
    
def spans_to_undir_sib(x):
    '''
    Convert spans arc to undirected in shst, sho_{h,t}
    '''
    undir_sec_mat = x.permute(0,3,1,2)
    undir_sec_mat = undir_sec_mat.triu() + undir_sec_mat.triu(1).transpose(-1,-2)
    
    undir_sec_mat = undir_sec_mat.permute(0,2,3,1)
    
    return undir_sec_mat
    
def get_relation_label_mask(subjects, rel_labels):
    
    lab_mask = torch.where(rel_labels != -1, 1, 0).bool()
    
    lab_mask_pred = torch.where(subjects != -1, 1, 0).bool()
    lab_mask_pred = torch.any(lab_mask_pred, dim = -1)
    
    
    return lab_mask, lab_mask_pred
    
def get_relation_decoding_mask(subjects, seq_lens, t):
    bsz = seq_lens.shape[0]
    # word_mask = span_labels_seq.new_zeros()
    word_mask = torch.zeros((bsz, t+1), device=seq_lens.device)
    word_mask[torch.arange(bsz), seq_lens] = 1
    word_mask = 1 - word_mask.cumsum(dim=-1)
    word_mask = word_mask.bool()
    word_mask = word_mask[:, :-1]
    
    
    # [bsz, seq_len, seq_len]
    mask = word_mask.unsqueeze(-1) & word_mask.unsqueeze(-2)
        
    lab_mask_pred = torch.where(subjects != -1, 1, 0).bool()
    lab_mask_pred = torch.any(lab_mask_pred, dim = -1)
    
    
    return mask, lab_mask_pred