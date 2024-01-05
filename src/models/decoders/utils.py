import torch

def get_constrainted_role_label(p2r_label2id, p2r_id2label, frame_label, role_scores, frame_fe_map):
    valid_frame_elements_mask = torch.zeros(len(p2r_label2id)).detach().cpu()
    if p2r_id2label[0] == "O":
        valid_frame_elements_mask[0] = True
    # breakpoint()
    try:
        valid_frame_elements = torch.LongTensor([p2r_label2id[fe] for fe in frame_fe_map[frame_label] if fe in p2r_label2id])
        valid_frame_elements_mask.index_fill_(-1, valid_frame_elements, True)
    except IndexError:
        breakpoint()
    frame_element_label_id = torch.argmax(sum(role_scores) + torch.log(valid_frame_elements_mask).to(role_scores[0].device)).item()
    
    return p2r_id2label[frame_element_label_id]

def get_constrained_frame_label(tokens, lemmas, frame_ix, frame_scores, lu_frame_map, frame2id, id2frame, num_frame_labels):
    frame_tokens = " ".join([" ".join(tokens[start:end+1]) for start, end in frame_ix])
    frame_lemmas = " ".join([" ".join(lemmas[start:end+1]) for start, end in frame_ix])
    valid_frame_masks = torch.zeros(num_frame_labels).detach().cpu()
    # valid_frame_masks[0] = True

    valid_lemma = None
    if frame_lemmas in lu_frame_map:
        valid_lemma = frame_lemmas
    elif frame_tokens in lu_frame_map:
        valid_lemma = frame_tokens
    if valid_lemma:
        valid_frame_labels = torch.LongTensor([frame2id[frame] for frame in lu_frame_map[valid_lemma] if frame in frame2id])
        valid_frame_masks.index_fill_(-1, valid_frame_labels, True)
    else:
        valid_frame_masks.fill_(True)
    
    frame_label = id2frame[torch.argmax(sum(frame_scores) + torch.log(valid_frame_masks).to(frame_scores[0].device)).item()]
    return frame_label

def is_clique(entity, relations):
    entity = list(entity)

    for idx, fragment1 in enumerate(entity):
        for idy, fragment2 in enumerate(entity):
            if idx < idy:
                if (fragment1, fragment2) not in relations and (fragment2, fragment1) not in relations:
                    return False

    return True