import torch
import os
import numpy as np
import random
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import encode_input, encode_input_with_mask, encode_image_input
from dataloader.dataloader_base import DatasetBase


class VisdialDataset(DatasetBase):

    def __init__(self, config):
        super(VisdialDataset, self).__init__(config)

    def __getitem__(self, index):
        MAX_SEQ_LEN = self.config['max_seq_len']
        cur_data = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            ques_adj_matrices_dir = os.path.join(self.config['visdial_question_adj_matrices'], 'train')
            hist_adj_matrices_dir = os.path.join(self.config['visdial_history_adj_matrices'], 'train')

        elif self._split == 'val':
            cur_data = self.visdial_data_val['data']
            ques_adj_matrices_dir = os.path.join(self.config['visdial_question_adj_matrices'], 'val')
            hist_adj_matrices_dir = os.path.join(self.config['visdial_history_adj_matrices'], 'val')

        else:
            cur_data = self.visdial_data_test['data']
            ques_adj_matrices_dir = os.path.join(self.config['visdial_question_adj_matrices'], 'test')
            hist_adj_matrices_dir = os.path.join(self.config['visdial_history_adj_matrices'], 'test')

        if self.config['visdial_version'] == 0.9:
            ques_adj_matrices_dir = os.path.join(self.config['visdial_question_adj_matrices'], 'train')
            hist_adj_matrices_dir = os.path.join(self.config['visdial_history_adj_matrices'], 'train')

        self.num_bad_samples = 0
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        num_dialog_rounds = 10
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        graph_idx = dialog.get('dialog_idx', index)

        if self._split == 'train':
            # caption
            sent = dialog['caption'].split(' ')
            sentences = ['[CLS]']
            tot_len = 1 # for the CLS token 
            sentence_map = [0] # for the CLS token 
            sentence_count = 0
            speakers = [0]

            tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

            utterances = [[tokenized_sent]]
            utterances_random = [[tokenized_sent]]

            for rnd, utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance_random = utterances[-1].copy()
                
                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

                cur_rnd_utterance.append(tokenized_sent)
                cur_rnd_utterance_random.append(tokenized_sent)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
                cur_rnd_utterance.append(tokenized_sent)

                utterances.append(cur_rnd_utterance)

                # randomly select one random utterance in that round
                num_inds = len(utterance['answer_options'])
                gt_option_ind = utterance['gt_index']

                negative_samples = []

                for _ in range(self.config["num_negative_samples"]):

                    all_inds = list(range(100))
                    all_inds.remove(gt_option_ind)
                    all_inds = all_inds[:(num_options-1)]
                    tokenized_random_utterance = None
                    option_ind = None

                    while len(all_inds):
                        option_ind = random.choice(all_inds)
                        tokenized_random_utterance = self.tokenizer.convert_tokens_to_ids(cur_answers[utterance['answer_options'][option_ind]].split(' '))
                        # the 1 here is for the sep token at the end of each utterance
                        if(MAX_SEQ_LEN >= (tot_len + len(tokenized_random_utterance) + 1)):
                            break
                        else:
                            all_inds.remove(option_ind)
                    if len(all_inds) == 0:
                        # all the options exceed the max len. Truncate the last utterance in this case.
                        tokenized_random_utterance = tokenized_random_utterance[:len(tokenized_sent)]
                    t = cur_rnd_utterance_random.copy()
                    t.append(tokenized_random_utterance)
                    negative_samples.append(t)

                utterances_random.append(negative_samples)

            # removing the caption in the beginning
            utterances = utterances[1:]
            utterances_random = utterances_random[1:]
            assert len(utterances) == len(utterances_random) == num_dialog_rounds
            assert tot_len <= MAX_SEQ_LEN, '{} {} tot_len = {} > max_seq_len'.format(
                self._split, index, tot_len
            )

            tokens_all = []
            question_limits_all = []
            question_edge_indices_all = []
            question_edge_attributes_all = []
            history_edge_indices_all = []
            history_sep_indices_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            next_labels_all = []
            hist_len_all = []

            # randomly pick several rounds to train
            pos_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)
            neg_rounds = sorted(random.sample(range(num_dialog_rounds), self.config['sequences_per_image'] // 2), reverse=True)

            tokens_all_rnd = []
            question_limits_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            next_labels_all_rnd = []
            hist_len_all_rnd = []

            for j in pos_rounds:
                context = utterances[j]
                context, start_segment = self.pruneRounds(context, self.config['visdial_tot_rounds'])
                if j == pos_rounds[0]: # dialog with positive label and max rounds
                    tokens, segments, sep_indices, mask, input_mask, start_question, end_question = encode_input_with_mask(context, start_segment, self.CLS,
                     self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                else:
                    tokens, segments, sep_indices, mask, start_question, end_question = encode_input(context, start_segment, self.CLS,
                     self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                tokens_all_rnd.append(tokens)
                question_limits_all_rnd.append(torch.tensor([start_question, end_question]))
                mask_all_rnd.append(mask)
                sep_indices_all_rnd.append(sep_indices)
                next_labels_all_rnd.append(torch.LongTensor([0]))
                segments_all_rnd.append(segments)
                hist_len_all_rnd.append(torch.LongTensor([len(context)-1]))

            tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
            mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
            question_limits_all.extend(question_limits_all_rnd)
            segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
            sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
            next_labels_all.append(torch.cat(next_labels_all_rnd, 0).unsqueeze(0))
            hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))
            
            assert len(pos_rounds) == 1
            question_graphs = pickle.load(
                open(os.path.join(ques_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb')
            )

            question_graph_pos = question_graphs[pos_rounds[0]]
            question_edge_index_pos = []
            question_edge_attribute_pos = []
            for edge_idx, edge_attr in question_graph_pos:
                question_edge_index_pos.append(edge_idx)
                edge_attr_one_hot = np.zeros((len(self.parse_vocab) + 1,), dtype=np.float32)
                edge_attr_one_hot[self.parse_vocab.get(edge_attr, len(self.parse_vocab))] = 1.0
                question_edge_attribute_pos.append(edge_attr_one_hot)
            
            question_edge_index_pos = np.array(question_edge_index_pos, dtype=np.float64)
            question_edge_attribute_pos = np.stack(question_edge_attribute_pos, axis=0)

            question_edge_indices_all.append(
                torch.from_numpy(question_edge_index_pos).t().long().contiguous()
            )

            question_edge_attributes_all.append(
                torch.from_numpy(question_edge_attribute_pos)
            )

            history_edge_indices = pickle.load(
                open(os.path.join(hist_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb')
            )

            history_edge_indices_all.append(
                torch.tensor(history_edge_indices[pos_rounds[0]]).t().long().contiguous()
            )
            # Get the [SEP] tokens that will represent the history graph node features
            hist_idx_pos = [i * 2 for i in range(pos_rounds[0] + 1)]
            sep_indices = sep_indices.squeeze(0).numpy()
            history_sep_indices_all.append(torch.from_numpy(sep_indices[hist_idx_pos]))

            if len(neg_rounds) > 0:
                tokens_all_rnd = []
                question_limits_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                next_labels_all_rnd = []
                hist_len_all_rnd = []

                for j in neg_rounds:

                    negative_samples = utterances_random[j]
                    for context_random in negative_samples:
                        context_random, start_segment = self.pruneRounds(context_random, self.config['visdial_tot_rounds'])
                        tokens_random, segments_random, sep_indices_random, mask_random, start_question, end_question = encode_input(context_random, start_segment, self.CLS, 
                        self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.config["mask_prob"])
                        tokens_all_rnd.append(tokens_random)
                        question_limits_all_rnd.append(torch.tensor([start_question, end_question]))
                        mask_all_rnd.append(mask_random)
                        sep_indices_all_rnd.append(sep_indices_random)
                        next_labels_all_rnd.append(torch.LongTensor([1]))
                        segments_all_rnd.append(segments_random)
                        hist_len_all_rnd.append(torch.LongTensor([len(context_random)-1]))

                tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
                mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
                question_limits_all.extend(question_limits_all_rnd)
                segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
                sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
                next_labels_all.append(torch.cat(next_labels_all_rnd, 0).unsqueeze(0))
                hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))

            assert len(neg_rounds) == 1

            question_graph_neg = question_graphs[neg_rounds[0]]
            question_edge_index_neg = []
            question_edge_attribute_neg = []
            for edge_idx, edge_attr in question_graph_neg:
                question_edge_index_neg.append(edge_idx)
                edge_attr_one_hot = np.zeros((len(self.parse_vocab) + 1,), dtype=np.float32)
                edge_attr_one_hot[self.parse_vocab.get(edge_attr, len(self.parse_vocab))] = 1.0
                question_edge_attribute_neg.append(edge_attr_one_hot)
            
            question_edge_index_neg = np.array(question_edge_index_neg, dtype=np.float64)
            question_edge_attribute_neg = np.stack(question_edge_attribute_neg, axis=0)

            question_edge_indices_all.append(
                torch.from_numpy(question_edge_index_neg).t().long().contiguous()
            )

            question_edge_attributes_all.append(
                torch.from_numpy(question_edge_attribute_neg)
            )

            history_edge_indices_all.append(
                torch.tensor(history_edge_indices[neg_rounds[0]]).t().long().contiguous()
            )

            # Get the [SEP] tokens that will represent the history graph node features
            hist_idx_neg = [i * 2 for i in range(neg_rounds[0] + 1)]
            sep_indices_random = sep_indices_random.squeeze(0).numpy()
            history_sep_indices_all.append(torch.from_numpy(sep_indices_random[hist_idx_neg]))

            tokens_all = torch.cat(tokens_all, 0) # [2, num_pos, max_len]
            question_limits_all = torch.stack(question_limits_all, 0) # [2, 2]
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            next_labels_all = torch.cat(next_labels_all, 0)
            hist_len_all = torch.cat(hist_len_all, 0)
            input_mask_all = torch.LongTensor(input_mask)  # [max_len]
                   
            item = {}

            item['tokens'] = tokens_all
            item['question_limits'] = question_limits_all
            item['question_edge_indices'] = question_edge_indices_all
            item['question_edge_attributes'] = question_edge_attributes_all

            item['history_edge_indices'] = history_edge_indices_all
            item['history_sep_indices'] = history_sep_indices_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['next_sentence_labels'] = next_labels_all
            item['hist_len'] = hist_len_all
            item['input_mask'] = input_mask_all

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target, image_edge_indexes, image_edge_attributes = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num)
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])

        elif self._split == 'val':
            gt_relevance = None
            gt_option_inds = []
            options_all = []

            # caption
            sent = dialog['caption'].split(' ')
            sentences = ['[CLS]']
            tot_len = 1 # for the CLS token
            sentence_map = [0] # for the CLS token
            sentence_count = 0
            speakers = [0]

            tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
            utterances = [[tokenized_sent]]

            for rnd, utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()

                # question
                sent = cur_questions[utterance['question']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)

                cur_rnd_utterance.append(tokenized_sent)

                # current round
                gt_option_ind = utterance['gt_index']
                # first select gt option id, then choose the first num_options inds
                option_inds = []
                option_inds.append(gt_option_ind)
                all_inds = list(range(100))
                all_inds.remove(gt_option_ind)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option_inds.append(0)
                cur_rnd_options = []
                answer_options = [utterance['answer_options'][k] for k in option_inds]
                assert len(answer_options) == len(option_inds) == num_options
                assert answer_options[0] == utterance['answer']

                # for evaluation of all options and dense relevance
                if self.visdial_data_val_dense:
                    if rnd == self.visdial_data_val_dense[index]['round_id'] - 1:
                        # only 1 round has gt_relevance for each example
                        if 'relevance' in self.visdial_data_val_dense[index]:
                            gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['relevance'])
                        else:
                            gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                        # shuffle based on new indices
                        gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                else:
                    gt_relevance = -1

                for answer_option in answer_options:
                    cur_rnd_cur_option = cur_rnd_utterance.copy()
                    cur_rnd_cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
                    cur_rnd_options.append(cur_rnd_cur_option)

                # answer
                sent = cur_answers[utterance['answer']].split(' ')
                tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers = \
                    self.tokenize_utterance(sent, sentences, tot_len, sentence_count, sentence_map, speakers)
                cur_rnd_utterance.append(tokenized_sent)

                utterances.append(cur_rnd_utterance)
                options_all.append(cur_rnd_options)

            # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options)            
            tokens_all = []
            question_limits_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            history_sep_indices_all = []

            for rnd, cur_rnd_options in enumerate(options_all):

                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                hist_len_all_rnd = []

                for j, cur_rnd_option in enumerate(cur_rnd_options):

                    cur_rnd_option, start_segment = self.pruneRounds(cur_rnd_option, self.config['visdial_tot_rounds'])
                    if rnd == len(options_all) - 1 and j == 0: # gt dialog
                        tokens, segments, sep_indices, mask, input_mask, start_question, end_question = encode_input_with_mask(cur_rnd_option, start_segment, self.CLS,
                         self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=0)
                    else:
                        tokens, segments, sep_indices, mask, start_question, end_question = encode_input(cur_rnd_option, start_segment,self.CLS, 
                        self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                    tokens_all_rnd.append(tokens)
                    mask_all_rnd.append(mask)
                    segments_all_rnd.append(segments)
                    sep_indices_all_rnd.append(sep_indices)
                    hist_len_all_rnd.append(torch.LongTensor([len(cur_rnd_option)-1]))

                question_limits_all.append(torch.tensor([start_question, end_question]).unsqueeze(0).repeat(100, 1)) 
                tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
                mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
                segments_all.append(torch.cat(segments_all_rnd,0).unsqueeze(0))
                sep_indices_all.append(torch.cat(sep_indices_all_rnd,0).unsqueeze(0))
                hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))
                # Get the [SEP] tokens that will represent the history graph node features
                # It will be the same for all answer candidates as the history does not change
                # for each answer
                hist_idx = [i * 2 for i in range(rnd + 1)]
                history_sep_indices_all.extend(sep_indices.squeeze(0)[hist_idx].contiguous() for _ in range(100))

            tokens_all = torch.cat(tokens_all, 0) # [10, 100, max_len]
            mask_all = torch.cat(mask_all, 0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all, 0)
            input_mask_all = torch.LongTensor(input_mask) # [max_len]

            # load graph data 
            question_limits_all = torch.stack(question_limits_all, 0) # [10, 100, 2]
            
            question_graphs = pickle.load(  
                open(os.path.join(ques_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb')
            )
            question_edge_indices_all = []  # [10, N] we do not repeat it 100 times here
            question_edge_attributes_all = []  # [10, N] we do not repeat it 100 times here

            for q_graph_round in question_graphs:
                question_edge_index = []
                question_edge_attribute = []
                for edge_index, edge_attr in q_graph_round:
                    question_edge_index.append(edge_index)
                    edge_attr_one_hot = np.zeros((len(self.parse_vocab) + 1,), dtype=np.float32)
                    edge_attr_one_hot[self.parse_vocab.get(edge_attr, len(self.parse_vocab))] = 1.0
                    question_edge_attribute.append(edge_attr_one_hot)
                question_edge_index = np.array(question_edge_index, dtype=np.float64)
                question_edge_attribute = np.stack(question_edge_attribute, axis=0)

                question_edge_indices_all.extend(
                    [torch.from_numpy(question_edge_index).t().long().contiguous() for _ in range(100)])
                question_edge_attributes_all.extend(
                    [torch.from_numpy(question_edge_attribute).contiguous() for _ in range(100)])

            _history_edge_incides_all = pickle.load(
                open(os.path.join(hist_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb')
            )
            history_edge_incides_all = []
            for hist_edge_indices_rnd in _history_edge_incides_all:
                history_edge_incides_all.extend(
                    [torch.tensor(hist_edge_indices_rnd).t().long().contiguous() for _ in range(100)]
                )
                   
            item = {}
            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['hist_len'] = hist_len_all
            item['input_mask'] = input_mask_all

            item['gt_option_inds'] = torch.LongTensor(gt_option_inds)            

            # return dense annotation data as well
            if self.visdial_data_val_dense:
                item['round_id'] = torch.LongTensor([self.visdial_data_val_dense[index]['round_id']])
                item['gt_relevance'] = gt_relevance

            item['question_limits'] = question_limits_all

            item['question_edge_indices'] = question_edge_indices_all
            item['question_edge_attributes'] = question_edge_attributes_all

            item['history_edge_indices'] = history_edge_incides_all
            item['history_sep_indices'] = history_sep_indices_all

            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target, image_edge_indexes, image_edge_attributes = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(
                    features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])

        elif self.split == 'test':
            assert num_options == 100
            cur_rnd_utterance = [self.tokenizer.convert_tokens_to_ids(dialog['caption'].split(' '))]
            options_all = []
            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance.append(self.tokenizer.convert_tokens_to_ids(cur_questions[utterance['question']].split(' ')))
                if rnd != len(dialog['dialog'])-1:
                    cur_rnd_utterance.append(self.tokenizer.convert_tokens_to_ids(cur_answers[utterance['answer']].split(' ')))
            for answer_option in dialog['dialog'][-1]['answer_options']:
                cur_option = cur_rnd_utterance.copy()
                cur_option.append(self.tokenizer.convert_tokens_to_ids(cur_answers[answer_option].split(' ')))
                options_all.append(cur_option)

            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            
            for j, option in enumerate(options_all):
                option, start_segment = self.pruneRounds(option, self.config['visdial_tot_rounds'])
                tokens, segments, sep_indices, mask = encode_input(option, start_segment ,self.CLS, 
                self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                tokens_all.append(tokens)
                mask_all.append(mask)
                segments_all.append(segments)
                sep_indices_all.append(sep_indices)
                hist_len_all.append(torch.LongTensor([len(option)-1]))
                
            tokens_all = torch.cat(tokens_all,0)
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all,0)
            hist_idx = [i*2 for i in range(len(dialog['dialog']))]
            history_sep_indices_all = [sep_indices.squeeze(0)[hist_idx].contiguous() for _ in range(num_options)]

            with open(os.path.join(ques_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb') as f:
                question_graphs = pickle.load(f)
            q_graph_last  = question_graphs[-1]
            question_edge_index = []
            question_edge_attribute = []
            for edge_index, edge_attr in q_graph_last:
                question_edge_index.append(edge_index)
                edge_attr_one_hot = np.zeros((len(self.parse_vocab) + 1,), dtype=np.float32)
                edge_attr_one_hot[self.parse_vocab.get(edge_attr, len(self.parse_vocab))] = 1.0
                question_edge_attribute.append(edge_attr_one_hot)
            question_edge_index = np.array(question_edge_index, dtype=np.float64)
            question_edge_attribute = np.stack(question_edge_attribute, axis=0)

            question_edge_indices_all = [torch.from_numpy(question_edge_index).t().long().contiguous() for _ in range(num_options)]
            question_edge_attributes_all = [torch.from_numpy(question_edge_attribute).contiguous() for _ in range(num_options)]
            
            with open(os.path.join(hist_adj_matrices_dir, f'{graph_idx}.pkl'), 'rb') as f:
                _history_edge_incides_all = pickle.load(f)
            _history_edge_incides_last = _history_edge_incides_all[-1]
            history_edge_index_all = [torch.tensor(_history_edge_incides_last).t().long().contiguous() for _ in range(num_options)]
            
            if self.config['stack_gr_data']:
                question_edge_indices_all = torch.stack(question_edge_indices_all, dim=0)
                question_edge_attributes_all = torch.stack(question_edge_attributes_all, dim=0)
                history_edge_index_all = torch.stack(history_edge_index_all, dim=0)
                history_sep_indices_all = torch.stack(history_sep_indices_all, dim=0)
                len_question_gr = torch.tensor(question_edge_indices_all.size(-1)).unsqueeze(0).repeat(num_options, 1)
                len_history_gr = torch.tensor(history_edge_index_all.size(-1)).repeat(num_options, 1)
                len_history_sep = torch.tensor(history_sep_indices_all.size(-1)).repeat(num_options, 1)

            item = {}
            item['tokens'] = tokens_all.unsqueeze(0)
            item['segments'] = segments_all.unsqueeze(0)
            item['sep_indices'] = sep_indices_all.unsqueeze(0)
            item['mask'] = mask_all.unsqueeze(0)
            item['hist_len'] = hist_len_all.unsqueeze(0)
            item['question_limits'] = question_limits_all
            item['question_edge_indices'] = question_edge_indices_all
            item['question_edge_attributes'] = question_edge_attributes_all

            item['history_edge_indices'] = history_edge_index_all
            item['history_sep_indices'] = history_sep_indices_all
            
            if self.config['stack_gr_data']:
                item['len_question_gr'] = len_question_gr
                item['len_history_gr'] = len_history_gr
                item['len_history_sep'] = len_history_sep
                
            item['round_id'] = torch.LongTensor([dialog['round_id']])
        
            # get image features
            if not self.config['dataloader_text_only']:
                features, num_boxes, boxes, _ , image_target, image_edge_indexes, image_edge_attributes = self._image_features_reader[img_id]
                features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num, mask_prob=0)
            else:
                features = spatials = image_mask = image_target = image_label = torch.tensor([0])
     
        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_target'] = image_target
        item['image_label'] = image_label
        item['image_id'] = torch.LongTensor([img_id])
        if self._split == 'train':
            # cheap hack to account for the graph data for the postitive and negatice examples
            item['image_edge_indices'] = [torch.from_numpy(image_edge_indexes).long(), torch.from_numpy(image_edge_indexes).long()]
            item['image_edge_attributes'] = [torch.from_numpy(image_edge_attributes), torch.from_numpy(image_edge_attributes)]
        elif self._split == 'val':
            # cheap hack to account for the graph data for the postitive and negatice examples
            item['image_edge_indices'] = [torch.from_numpy(image_edge_indexes).contiguous().long() for _ in range(1000)]
            item['image_edge_attributes'] = [torch.from_numpy(image_edge_attributes).contiguous() for _ in range(1000)]
        
        else:
            # cheap hack to account for the graph data for the postitive and negatice examples
            item['image_edge_indices'] = [torch.from_numpy(image_edge_indexes).contiguous().long() for _ in range(100)]
            item['image_edge_attributes'] = [torch.from_numpy(image_edge_attributes).contiguous() for _ in range(100)]
        
        if self.config['stack_gr_data']:
            item['image_edge_indices'] = torch.stack(item['image_edge_indices'], dim=0)
            item['image_edge_attributes'] = torch.stack(item['image_edge_attributes'], dim=0)
            len_image_gr = torch.tensor(item['image_edge_indices'].size(-1)).unsqueeze(0).repeat(num_options)
            item['len_image_gr'] = len_image_gr

        return item
