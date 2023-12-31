import torch
from torch.autograd import Variable
import random 
import pickle
import numpy as np
from copy import deepcopy


def load_pickle_lines(filename):
    data = []
    with open(filename, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data


def flatten(l):
    return [item for sublist in l for item in sublist]


def build_len_mask_batch(
        # [batch_size], []
        len_batch, max_len=None
):
    if max_len is None:
        max_len = len_batch.max().item()
    # try:
    batch_size, = len_batch.shape
    # [batch_size, max_len]
    idxes_batch = torch.arange(max_len, device=len_batch.device).view(1, -1).repeat(batch_size, 1)
    # [batch_size, max_len] = [batch_size, max_len] < [batch_size, 1]
    return idxes_batch < len_batch.view(-1, 1)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def batch_iter(dataloader, params):
    for epochId in range(params['num_epochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

def list2tensorpad(inp_list, max_seq_len):
    inp_tensor = torch.LongTensor([inp_list])
    inp_tensor_zeros = torch.zeros(1, max_seq_len, dtype=torch.long)
    inp_tensor_zeros[0,:inp_tensor.shape[1]] = inp_tensor  # after preprocess, inp_tensor.shape[1] must < max_seq_len
    inp_tensor = inp_tensor_zeros
    return inp_tensor


def encode_input(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256,max_sep_len=25,mask_prob=0.2):

    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    sep_token_indices = []
    masked_token_list = []

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    masked_token_list.append(0)

    cur_sep_token_index = 0
    
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_masked_index = [1 if random.random() < mask_prob else 0 for _ in range(len(cur_utterance))]
        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment]*len(cur_utterance))

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
        sep_token_indices.append(cur_sep_token_index)            
        cur_segment = cur_segment ^ 1 # cur segment osciallates between 0 and 1
    start_question, end_question = sep_token_indices[-3] + 1, sep_token_indices[-2]
    assert end_question - start_question == len(utterances[-2])

    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1 
    # convert to tensors and pad to maximum seq length
    tokens = list2tensorpad(token_id_list,max_seq_len) # [1, max_len]
    masked_tokens = list2tensorpad(masked_token_list,max_seq_len)
    masked_tokens[0,masked_tokens[0,:]==0] = -1
    mask = masked_tokens[0,:]==1
    masked_tokens[0,mask] = tokens[0,mask]
    tokens[0,mask] = MASK

    segment_id_list = list2tensorpad(segment_id_list,max_seq_len)
    return tokens, segment_id_list, list2tensorpad(sep_token_indices,max_sep_len), masked_tokens, start_question, end_question

def encode_input_with_mask(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256,max_sep_len=25,mask_prob=0.2, get_q_limits=True):

    cur_segment = start_segment
    token_id_list = []
    segment_id_list = []
    sep_token_indices = []
    masked_token_list = []
    input_mask_list = []

    token_id_list.append(CLS)
    segment_id_list.append(cur_segment)
    masked_token_list.append(0)
    input_mask_list.append(1)

    cur_sep_token_index = 0
    
    for cur_utterance in utterances:
        # add the masked token and keep track
        cur_masked_index = [1 if random.random() < mask_prob else 0 for _ in range(len(cur_utterance))]
        masked_token_list.extend(cur_masked_index)
        token_id_list.extend(cur_utterance)
        segment_id_list.extend([cur_segment]*len(cur_utterance))
        input_mask_list.extend([1]*len(cur_utterance))

        token_id_list.append(SEP)
        segment_id_list.append(cur_segment)
        masked_token_list.append(0)
        input_mask_list.append(1)
        cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
        sep_token_indices.append(cur_sep_token_index)
        cur_segment = cur_segment ^ 1 # cur segment osciallates between 0 and 1
    
    if get_q_limits:
        start_question, end_question = sep_token_indices[-3] + 1, sep_token_indices[-2]
        assert end_question - start_question == len(utterances[-2])
    else:
        start_question, end_question = -1, -1
    assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) ==len(input_mask_list) == sep_token_indices[-1] + 1 
    # convert to tensors and pad to maximum seq length
    tokens = list2tensorpad(token_id_list, max_seq_len)
    masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
    input_mask = list2tensorpad(input_mask_list,max_seq_len)
    masked_tokens[0,masked_tokens[0,:]==0] = -1
    mask = masked_tokens[0,:]==1
    masked_tokens[0,mask] = tokens[0,mask]
    tokens[0,mask] = MASK

    segment_id_list = list2tensorpad(segment_id_list,max_seq_len)
    return tokens, segment_id_list, list2tensorpad(sep_token_indices,max_sep_len),masked_tokens, input_mask, start_question, end_question


def encode_image_input(features, num_boxes, boxes, image_target, max_regions=37, mask_prob=0.15):
    output_label = []
    num_boxes = min(int(num_boxes), max_regions)

    mix_boxes_pad = np.zeros((max_regions, boxes.shape[-1]))
    mix_features_pad = np.zeros((max_regions, features.shape[-1]))
    mix_image_target = np.zeros((max_regions, image_target.shape[-1]))

    mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
    mix_features_pad[:num_boxes] = features[:num_boxes]
    mix_image_target[:num_boxes] = image_target[:num_boxes]
 
    boxes = mix_boxes_pad
    features = mix_features_pad
    image_target = mix_image_target
    mask_indexes = []
    for i in range(num_boxes):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.9:
                features[i] = 0
            output_label.append(1)
            mask_indexes.append(i)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    image_mask = [1] * (int(num_boxes))
    while len(image_mask) < max_regions:
        image_mask.append(0)
        output_label.append(-1)
    
    # ensure we have atleast one region being predicted
    output_label[random.randint(1,len(output_label)-1)] = 1
    image_label = torch.LongTensor(output_label)
    image_label[0] = 0 # make sure the <IMG> token doesn't contribute to the masked loss
    image_mask = torch.tensor(image_mask).float()

    features = torch.tensor(features).float()
    spatials = torch.tensor(boxes).float()
    image_target = torch.tensor(image_target).float()

    return features, spatials, image_mask, image_target, image_label


def question_edge_masking(question_edge_indices, question_edge_attributes, mask, question_limits, mask_prob=0.4, max_len=10):
    mask = mask.squeeze().tolist()
    question_limits = question_limits.tolist()
    question_start, question_end = question_limits
    # Get the masking of the question
    mask_question = mask[question_start:question_end]
    masked_idx = np.argwhere(np.array(mask_question) > -1).squeeze().tolist()
    if isinstance(masked_idx, (int)):  # only one question token is masked
        masked_idx = [masked_idx]

    # get rid of all edge indices and attributes that corresond to masked tokens
    edge_attr_gt = []
    edge_idx_gt_gnn = []
    edge_idx_gt_bert = []
    for i, (question_edge_idx, question_edge_attr) in enumerate(zip(question_edge_indices, question_edge_attributes)):
        if not(question_edge_idx[0] in masked_idx or question_edge_idx[1] in masked_idx):
            # Masking
            if random.random() < mask_prob:
                edge_attr_gt.append(np.argwhere(question_edge_attr).item())
                edge_idx_gt_gnn.append(question_edge_idx)
                edge_idx_gt_bert.append([question_edge_idx[0] + question_start, question_edge_idx[1] + question_start])
                question_edge_attr = np.zeros_like(question_edge_attr)
                question_edge_attr[-1] = 1.0  # The [EDGE_MASK] special token is the last one hot vector encoding
                question_edge_attributes[i] = question_edge_attr
        else:
            continue
    # Force masking if the necessary:
    if len(edge_attr_gt) == 0:
        for i, (question_edge_idx, question_edge_attr) in enumerate(zip(question_edge_indices, question_edge_attributes)):
            if not(question_edge_idx[0] in masked_idx or question_edge_idx[1] in masked_idx):
                # Masking
                edge_attr_gt.append(np.argwhere(question_edge_attr).item())
                edge_idx_gt_gnn.append(question_edge_idx)
                edge_idx_gt_bert.append([question_edge_idx[0] + question_start, question_edge_idx[1] + question_start])
                question_edge_attr = np.zeros_like(question_edge_attr)
                question_edge_attr[-1] = 1.0  # The [EDGE_MASK] special token is the last one hot vector encoding
                question_edge_attributes[i] = question_edge_attr
                break

    # For the rare case, where the conditions for masking were not met
    if len(edge_attr_gt) == 0:
        edge_attr_gt.append(-1)
        edge_idx_gt_gnn.append([0, question_end - question_start])
        edge_idx_gt_bert.append(question_limits)

    # Pad to max_len
    while len(edge_attr_gt) < max_len:
        edge_attr_gt.append(-1)
        edge_idx_gt_gnn.append(edge_idx_gt_gnn[-1])
        edge_idx_gt_bert.append(edge_idx_gt_bert[-1])

    # Truncate if longer than max_len
    if len(edge_attr_gt) > max_len:
        edge_idx_gt_gnn = edge_idx_gt_gnn[:max_len]
        edge_idx_gt_bert = edge_idx_gt_bert[:max_len]
        edge_attr_gt = edge_attr_gt[:max_len]
    edge_idx_gt_gnn = np.array(edge_idx_gt_gnn)
    edge_idx_gt_bert = np.array(edge_idx_gt_bert)

    first_edge_node_gt_gnn = list(edge_idx_gt_gnn[:, 0])
    second_edge_node_gt_gnn = list(edge_idx_gt_gnn[:, 1])

    first_edge_node_gt_bert = list(edge_idx_gt_bert[:, 0])
    second_edge_node_gt_bert = list(edge_idx_gt_bert[:, 1])

    return question_edge_attributes, edge_attr_gt, first_edge_node_gt_gnn, second_edge_node_gt_gnn, first_edge_node_gt_bert, second_edge_node_gt_bert
    

def to_data_list(feats, batch_idx):
    feat_list = []
    device = feats.device
    left = 0
    right = 0
    batch_size = batch_idx.max().item() + 1
    for batch in range(batch_size):
        if batch == batch_size - 1:
            right = batch_idx.size(0)
        else:
            right = torch.argwhere(batch_idx == batch + 1)[0].item()
        idx = torch.arange(left, right).unsqueeze(-1).repeat(1, feats.size(1)).to(device)
        feat_list.append(torch.gather(feats, 0, idx))
        left = right

    return feat_list

