import torch
import json
import os
import time
import numpy as np
import random
from tqdm import tqdm
import copy
import pyhocon
import glog as log
from collections import OrderedDict
import argparse
import pickle
import torch.utils.data as tud
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import encode_input, encode_image_input
from dataloader.dataloader_base import DatasetBase


class VisdialDenseDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialDenseDataset, self).__init__(config)
        with open(config.tr_graph_idx_mapping, 'r') as f:
            self.tr_graph_idx_mapping = json.load(f)

        with open(config.val_graph_idx_mapping, 'r') as f:
            self.val_graph_idx_mapping = json.load(f)

        with open(config.test_graph_idx_mapping, 'r') as f:
            self.test_graph_idx_mapping = json.load(f)


        self.question_gr_paths = {
            'train': os.path.join(self.config['visdial_question_adj_matrices'], 'train'),
            'val': os.path.join(self.config['visdial_question_adj_matrices'], 'val'),
            'test': os.path.join(self.config['visdial_question_adj_matrices'], 'test')
        }

        self.history_gr_paths = {
            'train': os.path.join(self.config['visdial_history_adj_matrices'], 'train'),
            'val': os.path.join(self.config['visdial_history_adj_matrices'], 'val'),
            'test': os.path.join(self.config['visdial_history_adj_matrices'], 'test')
        }


    def __getitem__(self, index):
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        cur_dense_annotations = None
       
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            cur_dense_annotations = self.visdial_data_train_dense
            cur_question_gr_path = self.question_gr_paths['train']
            cur_history_gr_path = self.history_gr_paths['train']
            cur_gr_mapping = self.tr_graph_idx_mapping

            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_train
        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
            cur_dense_annotations = self.visdial_data_val_dense
            cur_question_gr_path = self.question_gr_paths['val']
            cur_history_gr_path = self.history_gr_paths['val']
            cur_gr_mapping = self.val_graph_idx_mapping

            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_val
        elif self._split == 'trainval':
            if index >= self.numDataPoints['train']:
                cur_data = self.visdial_data_val['data']
                cur_dense_annotations = self.visdial_data_val_dense
                cur_gr_mapping = self.val_graph_idx_mapping
                index -= self.numDataPoints['train']
                cur_question_gr_path = self.question_gr_paths['val']
                cur_history_gr_path = self.history_gr_paths['val']
                if self.config['rlv_hst_only']:
                    cur_rlv_hst = self.rlv_hst_val
            else:
                cur_data = self.visdial_data_train['data']
                cur_dense_annotations = self.visdial_data_train_dense
                cur_question_gr_path = self.question_gr_paths['train']
                cur_gr_mapping = self.tr_graph_idx_mapping
                cur_history_gr_path = self.history_gr_paths['train']
                if self.config['rlv_hst_only']:
                    cur_rlv_hst = self.rlv_hst_train
        elif self._split == 'test':
            cur_data = self.visdial_data_test['data']
            cur_question_gr_path = self.question_gr_paths['test']
            cur_history_gr_path = self.history_gr_paths['test']
            if self.config['rlv_hst_only']:
                cur_rlv_hst = self.rlv_hst_test

        # number of options to score on
        num_options = self.num_options_dense
        if self._split == 'test' or self.config['validating'] or self.config['predicting']:
            assert num_options == 100
        else:
            assert num_options >=1 and num_options <=  100

        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        if self._split != 'test':
            graph_idx = cur_gr_mapping[str(img_id)]
        else:
            graph_idx = index

        if self._split != 'test':
            assert img_id == cur_dense_annotations[index]['image_id']
        if self.config['rlv_hst_only']:
            rlv_hst = cur_rlv_hst[str(img_id)] # [10 for each round, 10 for cap + first 9 round ]

        if self._split == 'test':
            cur_rounds = len(dialog['dialog']) # 1, 2, ..., 10
        else:
            cur_rounds = cur_dense_annotations[index]['round_id'] # 1, 2, ..., 10

        # caption
        cur_rnd_utterance = []
        include_caption = True
        if self.config['rlv_hst_only']:
            if self.config['rlv_hst_dense_round']:
                if rlv_hst[0] == 0:
                    include_caption = False
            elif rlv_hst[cur_rounds - 1][0] == 0:
                include_caption = False
        if include_caption:
            sent = dialog['caption'].split(' ')
            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            cur_rnd_utterance.append(tokenized_sent)
            # tot_len += len(sent) + 1

        for rnd, utterance in enumerate(dialog['dialog'][:cur_rounds]):
            if self.config['rlv_hst_only'] and rnd < cur_rounds - 1:
                if self.config['rlv_hst_dense_round']:
                    if rlv_hst[rnd + 1] == 0:
                        continue
                elif rlv_hst[cur_rounds - 1][rnd + 1] == 0:
                    continue
            # question
            sent = cur_questions[utterance['question']].split(' ')
            tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
            cur_rnd_utterance.append(tokenized_sent)

            # answer
            if rnd != cur_rounds - 1:
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
                cur_rnd_utterance.append(tokenized_sent)

        if self.config['rlv_hst_only']:
            num_rlv_rnds = len(cur_rnd_utterance) - 1
        else:
            num_rlv_rnds = None

        if self._split != 'test':
            gt_option = dialog['dialog'][cur_rounds - 1]['gt_index']
            if self.config['training'] or self.config['debugging']:
                # first select gt option id, then choose the first num_options inds
                option_inds = []
                option_inds.append(gt_option)
                all_inds = list(range(100))
                all_inds.remove(gt_option)
                # debug
                if num_options < 100:
                    random.shuffle(all_inds)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option = 0
            else:
                option_inds = range(num_options)
            answer_options = [dialog['dialog'][cur_rounds - 1]['answer_options'][k] for k in option_inds]
            if 'relevance' in cur_dense_annotations[index]:
                key = 'relevance'
            else:
                key = 'gt_relevance'
            gt_relevance = torch.Tensor(cur_dense_annotations[index][key])
            gt_relevance = gt_relevance[option_inds]
            assert len(answer_options) == len(option_inds) == num_options
        else:
            answer_options = dialog['dialog'][-1]['answer_options']
            assert len(answer_options) == num_options

        options_all = []
        for answer_option in answer_options:
            cur_option = cur_rnd_utterance.copy()
            cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
            options_all.append(cur_option)
            if not self.config['rlv_hst_only']:
                assert len(cur_option) == 2 * cur_rounds + 1

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []
        tot_len_debug = []

        for opt_id, option in enumerate(options_all):
            option, start_segment = self.pruneRounds(option, self.config['visdial_tot_rounds'])
            tokens, segments, sep_indices, mask, start_question, end_question = encode_input(option, start_segment ,self.CLS, 
                    self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

            tokens_all.append(tokens)
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            hist_len_all.append(torch.LongTensor([len(option)-1]))

            len_tokens = sum(len(s) for s in option)
            tot_len_debug.append(len_tokens + len(option) + 1)

        tokens_all = torch.cat(tokens_all,0)
        mask_all = torch.cat(mask_all,0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all,0)
        question_limits_all = torch.tensor([start_question, end_question]).unsqueeze(0).repeat(num_options, 1)
        if self.config['rlv_hst_only']:
            assert num_rlv_rnds > 0
            hist_idx = [i * 2 for i in range(num_rlv_rnds)]
        else:
            hist_idx = [i*2 for i in range(cur_rounds)]
        history_sep_indices_all = sep_indices.squeeze(0)[hist_idx].contiguous().unsqueeze(0).repeat(num_options, 1)

        with open(os.path.join(cur_question_gr_path, f'{graph_idx}.pkl'), 'rb') as f:
            question_graphs = pickle.load(f)
        question_graph_round = question_graphs[cur_rounds - 1]
        question_edge_index = []
        question_edge_attribute = []
        for edge_index, edge_attr in question_graph_round:
            question_edge_index.append(edge_index)
            edge_attr_one_hot = np.zeros((len(self.parse_vocab) + 1,), dtype=np.float32)
            edge_attr_one_hot[self.parse_vocab.get(edge_attr, len(self.parse_vocab))] = 1.0
            question_edge_attribute.append(edge_attr_one_hot)
        question_edge_index = np.array(question_edge_index, dtype=np.float64)
        question_edge_attribute = np.stack(question_edge_attribute, axis=0)

        question_edge_indices_all = [torch.from_numpy(question_edge_index).t().long().contiguous() for _ in range(num_options)]
        question_edge_attributes_all = [torch.from_numpy(question_edge_attribute).contiguous() for _ in range(num_options)]
                    
        if self.config['rlv_hst_only']:
            with open(os.path.join(cur_history_gr_path, f'{graph_idx}.pkl'), 'rb') as f:
                _history_edge_incides_round = pickle.load(f)
        else:
            with open(os.path.join(cur_history_gr_path, f'{graph_idx}.pkl'), 'rb') as f:
                _history_edge_incides_all = pickle.load(f)
                _history_edge_incides_round = _history_edge_incides_all[cur_rounds - 1]
        
        history_edge_index_all = [torch.tensor(_history_edge_incides_round).t().long().contiguous() for _ in range(num_options)]
            
        if self.config['stack_gr_data']:
            question_edge_indices_all = torch.stack(question_edge_indices_all, dim=0)
            question_edge_attributes_all = torch.stack(question_edge_attributes_all, dim=0)
            history_edge_index_all = torch.stack(history_edge_index_all, dim=0)

        item = {}

        item['tokens'] = tokens_all.unsqueeze(0) # [1, num_options, max_len]
        item['segments'] = segments_all.unsqueeze(0)
        item['sep_indices'] = sep_indices_all.unsqueeze(0)
        item['mask'] = mask_all.unsqueeze(0)
        item['hist_len'] = hist_len_all.unsqueeze(0)
        item['question_limits'] = question_limits_all
        item['question_edge_indices'] = question_edge_indices_all
        item['question_edge_attributes'] = question_edge_attributes_all
        item['history_edge_indices'] = history_edge_index_all
        item['history_sep_indices'] = history_sep_indices_all
                
        # add dense annotation fields
        if self._split != 'test':
            item['gt_relevance'] = gt_relevance # [num_options]
            item['gt_option_inds'] = torch.LongTensor([gt_option])

            # add next sentence labels for training with the nsp loss as well
            nsp_labels = torch.ones(*tokens_all.unsqueeze(0).shape[:-1]).long()
            nsp_labels[:,gt_option] = 0
            item['next_sentence_labels'] = nsp_labels

            item['round_id'] = torch.LongTensor([cur_rounds])
        else:
            if 'round_id' in dialog:
                item['round_id'] = torch.LongTensor([dialog['round_id']])
            else:
                item['round_id'] = torch.LongTensor([cur_rounds])

        # get image features
        if not self.config['dataloader_text_only']:
            features, num_boxes, boxes, _ , image_target, image_edge_indexes, image_edge_attributes = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
        else:
            features = spatials = image_mask = image_target = image_label = torch.tensor([0])
        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_id'] = torch.LongTensor([img_id])
        item['tot_len'] = torch.LongTensor(tot_len_debug)



        item['image_edge_indices'] = [torch.from_numpy(image_edge_indexes).contiguous().long() for _ in range(num_options)]
        item['image_edge_attributes'] = [torch.from_numpy(image_edge_attributes).contiguous() for _ in range(num_options)]

        if self.config['stack_gr_data']:
            item['image_edge_indices'] = torch.stack(item['image_edge_indices'], dim=0)
            item['image_edge_attributes'] = torch.stack(item['image_edge_attributes'], dim=0)

        return item
