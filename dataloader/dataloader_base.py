import torch
from torch.utils import data
import json
import os
import glog as log
import pickle

import torch.utils.data as tud
from pytorch_transformers.tokenization_bert import BertTokenizer

from utils.image_features_reader import ImageFeaturesH5Reader


class DatasetBase(data.Dataset):

    def __init__(self, config):

        if config['display']:
            log.info('Initializing dataset')

        # Fetch the correct dataset for evaluation
        if config['validating']:
            assert config.eval_dataset in ['visdial', 'visdial_conv', 'visdial_vispro', 'visdial_v09']
            if config.eval_dataset == 'visdial_conv':
                config['visdial_val'] = config.visdialconv_val
                config['visdial_val_dense_annotations'] = config.visdialconv_val_dense_annotations
            elif config.eval_dataset == 'visdial_vispro':
                config['visdial_val'] = config.visdialvispro_val
                config['visdial_val_dense_annotations'] = config.visdialvispro_val_dense_annotations
            elif config.eval_dataset == 'visdial_v09':
                config['visdial_val_09'] = config.visdial_test_09
                config['visdial_val_dense_annotations'] = None

        self.config = config
        self.numDataPoints = {}

        if not config['dataloader_text_only']:
            self._image_features_reader = ImageFeaturesH5Reader(
                config['visdial_image_feats'],
                config['visdial_image_adj_matrices']
                )

        if self.config['training'] or self.config['validating'] or self.config['predicting']:
            split2data = {'train': 'train', 'val': 'val', 'test': 'test'}
        elif self.config['debugging']:
            split2data = {'train': 'val', 'val': 'val', 'test': 'test'}
        elif self.config['visualizing']:
            split2data = {'train': 'train', 'val': 'train', 'test': 'test'}

        filename = f'visdial_{split2data["train"]}'
        if config['train_on_dense']:
            filename += '_dense'
        if self.config['visdial_version'] == 0.9:
            filename += '_09'

        with open(config[filename]) as f:
            self.visdial_data_train = json.load(f)
            if self.config.num_samples > 0:
                self.visdial_data_train['data']['dialogs'] = self.visdial_data_train['data']['dialogs'][:self.config.num_samples]
            self.numDataPoints['train'] = len(self.visdial_data_train['data']['dialogs'])

        filename = f'visdial_{split2data["val"]}'
        if config['train_on_dense'] and config['training']:
            filename += '_dense'
        if self.config['visdial_version'] == 0.9:
            filename += '_09'

        with open(config[filename]) as f:
            self.visdial_data_val = json.load(f)
            if self.config.num_samples > 0:
                self.visdial_data_val['data']['dialogs'] = self.visdial_data_val['data']['dialogs'][:self.config.num_samples]
            self.numDataPoints['val'] = len(self.visdial_data_val['data']['dialogs'])
        
        if config['train_on_dense']:
            self.numDataPoints['trainval'] = self.numDataPoints['train'] + self.numDataPoints['val']
        with open(config[f'visdial_{split2data["test"]}']) as f:
            self.visdial_data_test = json.load(f)
            self.numDataPoints['test'] = len(self.visdial_data_test['data']['dialogs'])

        self.rlv_hst_train = None
        self.rlv_hst_val = None
        self.rlv_hst_test = None

        if config['train_on_dense'] or config['predict_dense_round']:
            with open(config[f'visdial_{split2data["train"]}_dense_annotations']) as f:
                self.visdial_data_train_dense = json.load(f)
        if config['train_on_dense']:
            self.subsets = ['train', 'val', 'trainval', 'test']
        else:
            self.subsets = ['train','val','test']
        self.num_options = config["num_options"]
        self.num_options_dense = config["num_options_dense"]
        if config['visdial_version'] != 0.9:
            with open(config[f'visdial_{split2data["val"]}_dense_annotations']) as f:
                self.visdial_data_val_dense = json.load(f)
        else:
            self.visdial_data_val_dense = None
        self._split = 'train'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config['bert_cache_dir'])
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]','[MASK]','[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP  = indexed_tokens[2]
        self._max_region_num = 37
        self.predict_each_round = self.config['predicting'] and self.config['predict_each_round']

        self.keys_to_expand = ['image_feat', 'image_loc', 'image_mask', 'image_target', 'image_label']
        self.keys_to_flatten_1d = ['hist_len', 'next_sentence_labels', 'round_id', 'image_id']
        self.keys_to_flatten_2d = ['tokens', 'segments', 'sep_indices', 'mask', 'image_mask', 'image_label', 'input_mask', 'question_limits']
        self.keys_to_flatten_3d = ['image_feat', 'image_loc', 'image_target', ]
        self.keys_other = ['gt_relevance', 'gt_option_inds']
        self.keys_lists_to_flatten = ['image_edge_indices', 'image_edge_attributes', 'question_edge_indices', 'question_edge_attributes', 'history_edge_indices', 'history_sep_indices']
        if config['stack_gr_data']:
            self.keys_to_flatten_3d.extend(self.keys_lists_to_flatten[:-1])
            self.keys_to_flatten_2d.append(self.keys_lists_to_flatten[-1])
            self.keys_to_flatten_1d.extend(['len_image_gr', 'len_question_gr', 'len_history_gr', 'len_history_sep'])
            self.keys_lists_to_flatten = []

        self.keys_to_list = ['tot_len']

        # Load the parse vocab for question graph relationship mapping
        if os.path.isfile(config['visdial_question_parse_vocab']):
            with open(config['visdial_question_parse_vocab'], 'rb') as f:
                self.parse_vocab = pickle.load(f)

    def __len__(self):
        return self.numDataPoints[self._split]
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def tokens2str(self, seq):
        dialog_sequence = ''
        for sentence in seq:
            for word in sentence:
                dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
            dialog_sequence += ' </end> '
        dialog_sequence = dialog_sequence.encode('utf8')
        return dialog_sequence

    def pruneRounds(self, context, num_rounds):
        start_segment = 1
        len_context = len(context)
        cur_rounds = (len(context) // 2) + 1
        l_index = 0
        if cur_rounds > num_rounds:
            # caption is not part of the final input
            l_index = len_context - (2 * num_rounds)
            start_segment = 0   
        return context[l_index:], start_segment

    def tokenize_utterance(self, sent, sentences, tot_len, sentence_count, sentence_map, speakers):
        sentences.extend(sent + ['[SEP]'])
        tokenized_sent = self.tokenizer.convert_tokens_to_ids(sent)
        assert len(sent) == len(tokenized_sent), 'sub-word tokens are not allowed!'

        sent_len = len(tokenized_sent)
        tot_len += sent_len + 1 # the additional 1 is for the sep token
        sentence_count += 1
        sentence_map.extend([sentence_count * 2 - 1] * sent_len)
        sentence_map.append(sentence_count * 2) # for [SEP]
        speakers.extend([2] * (sent_len + 1))

        return tokenized_sent, sentences, tot_len, sentence_count, sentence_map, speakers

    def __getitem__(self, index):
        return NotImplementedError

    def collate_fn(self, batch):
        tokens_size = batch[0]['tokens'].size()
        num_rounds, num_samples = tokens_size[0], tokens_size[1]
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}

        if self.config['stack_gr_data']:
            if (len(batch)) > 1:
                max_question_gr_len = max([length.max().item() for length in merged_batch['len_question_gr']])
                max_history_gr_len = max([length.max().item() for length in merged_batch['len_history_gr']])
                max_history_sep_len = max([length.max().item() for length in merged_batch['len_history_sep']])
                max_image_gr_len = max([length.max().item() for length in merged_batch['len_image_gr']])

                question_edge_indices_padded = []
                question_edge_attributes_padded = []

                for q_e_idx, q_e_attr in zip(merged_batch['question_edge_indices'], merged_batch['question_edge_attributes']):
                    b_size, edge_dim, orig_len = q_e_idx.size()
                    q_e_idx_padded = torch.zeros((b_size, edge_dim, max_question_gr_len))
                    q_e_idx_padded[:, :, :orig_len] = q_e_idx
                    question_edge_indices_padded.append(q_e_idx_padded)

                    edge_attr_dim = q_e_attr.size(-1)
                    q_e_attr_padded = torch.zeros((b_size, max_question_gr_len, edge_attr_dim))
                    q_e_attr_padded[:, :orig_len, :] = q_e_attr
                    question_edge_attributes_padded.append(q_e_attr_padded)

                merged_batch['question_edge_indices'] = question_edge_indices_padded
                merged_batch['question_edge_attributes'] = question_edge_attributes_padded

                history_edge_indices_padded = []
                for h_e_idx in merged_batch['history_edge_indices']:
                    b_size, _, orig_len = h_e_idx.size()
                    h_edge_idx_padded = torch.zeros((b_size, 2, max_history_gr_len))
                    h_edge_idx_padded[:, :, :orig_len] = h_e_idx
                    history_edge_indices_padded.append(h_edge_idx_padded)
                merged_batch['history_edge_indices'] = history_edge_indices_padded

                history_sep_indices_padded = []
                for hist_sep_idx in merged_batch['history_sep_indices']:
                    b_size, orig_len = hist_sep_idx.size()
                    hist_sep_idx_padded = torch.zeros((b_size, max_history_sep_len))
                    hist_sep_idx_padded[:, :orig_len] = hist_sep_idx
                    history_sep_indices_padded.append(hist_sep_idx_padded)
                merged_batch['history_sep_indices'] = history_sep_indices_padded

                image_edge_indices_padded = []
                image_edge_attributes_padded = []
                for img_e_idx, img_e_attr in zip(merged_batch['image_edge_indices'], merged_batch['image_edge_attributes']):
                    b_size, edge_dim, orig_len = img_e_idx.size()
                    img_e_idx_padded = torch.zeros((b_size, edge_dim, max_image_gr_len))
                    img_e_idx_padded[:, :, :orig_len] = img_e_idx
                    image_edge_indices_padded.append(img_e_idx_padded)

                    edge_attr_dim = img_e_attr.size(-1)
                    img_e_attr_padded = torch.zeros((b_size, max_image_gr_len, edge_attr_dim))
                    img_e_attr_padded[:, :orig_len, :] = img_e_attr
                    image_edge_attributes_padded.append(img_e_attr_padded)

                merged_batch['image_edge_indices'] = image_edge_indices_padded
                merged_batch['image_edge_attributes'] = image_edge_attributes_padded

        out = {}
        for key in merged_batch:
            if key in self.keys_lists_to_flatten:
                temp = []
                for b in merged_batch[key]:
                    for x in b:
                        temp.append(x)
                merged_batch[key] = temp
    
            elif key in self.keys_to_list:
                pass
            else:
                merged_batch[key] = torch.stack(merged_batch[key], 0)
                if key in self.keys_to_expand:
                    if len(merged_batch[key].size()) == 3:
                        size0, size1, size2 = merged_batch[key].size()
                        expand_size = (size0, num_rounds, num_samples, size1, size2)
                    elif len(merged_batch[key].size()) == 2:
                        size0, size1 = merged_batch[key].size()
                        expand_size = (size0, num_rounds, num_samples, size1)
                    merged_batch[key] = merged_batch[key].unsqueeze(1).unsqueeze(1).expand(expand_size).contiguous()
                if key in self.keys_to_flatten_1d:
                    merged_batch[key] = merged_batch[key].reshape(-1)
                elif key in self.keys_to_flatten_2d:
                    merged_batch[key] = merged_batch[key].reshape(-1, merged_batch[key].shape[-1])
                elif key in self.keys_to_flatten_3d:
                    merged_batch[key] = merged_batch[key].reshape(-1, merged_batch[key].shape[-2], merged_batch[key].shape[-1])
                else:
                    assert key in self.keys_other, f'unrecognized key in collate_fn: {key}'

            out[key] = merged_batch[key]
        return out
