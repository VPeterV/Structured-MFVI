FRAMENET_PRD_IDX = 0
FRAMENET_ROLE_IDX = 1

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label
