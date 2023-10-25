import sys
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('../')
from utils.model_utils import listMLE, approxNDCGLoss, listNet, neuralNDCG, neuralNDCG_transposed

from utils.data_utils import sequence_mask
from utils.optim_utils import init_optim
from models.runner import Runner

from models.vilbert_dialog import BertForMultiModalPreTraining, BertConfig


class VDGR(nn.Module):

    def __init__(self, config_path, device, use_apex=False, cache_dir=None):
        super(VDGR, self).__init__()
        config = BertConfig.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased', config, device, use_apex=use_apex, cache_dir=cache_dir)
        self.bert_pretrained.train()

    def forward(self, input_ids, image_feat, image_loc, image_edge_indices, image_edge_attributes,
        question_edge_indices, question_edge_attributes, question_limits,
        history_edge_indices, history_sep_indices,
        sep_indices=None, sep_len=None, token_type_ids=None,
        attention_mask=None, masked_lm_labels=None, next_sentence_label=None,
        image_attention_mask=None, image_label=None, image_target=None):

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        seq_relationship_score = None

        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, _, _, seq_relationship_score, _  = \
                self.bert_pretrained(
                    input_ids, image_feat, image_loc, image_edge_indices, image_edge_attributes,
                    question_edge_indices, question_edge_attributes, question_limits,
                    history_edge_indices, history_sep_indices, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)
        else:
            #inference, output scores
            _, _, seq_relationship_score, _, _, _ = \
                self.bert_pretrained(
                    input_ids, image_feat, image_loc, image_edge_indices, image_edge_attributes,
                    question_edge_indices, question_edge_attributes, question_limits,
                    history_edge_indices, history_sep_indices,
                    sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)

        out = (masked_lm_loss, masked_img_loss, nsp_loss, seq_relationship_score)

        return out


class SparseRunner(Runner):
    def __init__(self, config):
        super(SparseRunner, self).__init__(config)
        self.model = VDGR(
            self.config['model_config'], self.config['device'], 
            use_apex=self.config['dp_type'] == 'apex', 
            cache_dir=self.config['bert_cache_dir'])

        self.model.to(self.config['device'])

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False):
        # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])
            elif isinstance(batch[key], list):
                if key != 'dialog_info':  # Do not send the dialog_info item to the gpu
                    batch[key] = [x.to(self.config['device']) for x in batch[key]]

        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices']
        mask = batch['mask']
        hist_len = batch['hist_len']
        image_feat = batch['image_feat']
        image_loc = batch['image_loc'] 
        image_mask = batch['image_mask']
        next_sentence_labels = batch.get('next_sentence_labels', None)
        image_target = batch.get('image_target', None)
        image_label = batch.get('image_label', None)
        # load the graph data
        image_edge_indices = batch['image_edge_indices']
        image_edge_attributes = batch['image_edge_attributes']
        question_edge_indices = batch['question_edge_indices']
        question_edge_attributes = batch['question_edge_attributes']
        question_limits = batch['question_limits']
        history_edge_indices = batch['history_edge_indices']
        history_sep_indices = batch['history_sep_indices']
       
        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        losses = OrderedDict()

        if eval_visdial:
            num_lines = tokens.size(0)
            line_batch_size = self.config['eval_line_batch_size']
            num_line_batches = num_lines // line_batch_size
            if num_lines % line_batch_size > 0:
                num_line_batches += 1
            nsp_scores = []
            for j in range(num_line_batches):
                # create chunks of the original batch
                chunk_range = range(j*line_batch_size, min((j+1)*line_batch_size, num_lines))
                tokens_chunk = tokens[chunk_range]
                segments_chunk = segments[chunk_range]
                sep_indices_chunk = sep_indices[chunk_range]
                mask_chunk = mask[chunk_range]
                sep_len_chunk = sep_len[chunk_range]
                attention_mask_lm_nsp_chunk = attention_mask_lm_nsp[chunk_range]
                image_feat_chunk = image_feat[chunk_range]
                image_loc_chunk = image_loc[chunk_range]
                image_mask_chunk = image_mask[chunk_range]
                image_edge_indices_chunk = image_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                image_edge_attributes_chunk = image_edge_attributes[chunk_range[0]: chunk_range[-1]+1]
                question_edge_indices_chunk = question_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                question_edge_attributes_chunk = question_edge_attributes[chunk_range[0]: chunk_range[-1]+1]
                question_limits_chunk = question_limits[chunk_range[0]: chunk_range[-1]+1]
                history_edge_indices_chunk = history_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                history_sep_indices_chunk = history_sep_indices[chunk_range[0]: chunk_range[-1]+1]

                _ , _ , _, nsp_scores_chunk = \
                    self.model(
                        tokens_chunk,
                        image_feat_chunk,
                        image_loc_chunk,
                        image_edge_indices_chunk,
                        image_edge_attributes_chunk,
                        question_edge_indices_chunk,
                        question_edge_attributes_chunk,
                        question_limits_chunk,
                        history_edge_indices_chunk,
                        history_sep_indices_chunk,
                        sep_indices=sep_indices_chunk,
                        sep_len=sep_len_chunk,
                        token_type_ids=segments_chunk,
                        masked_lm_labels=mask_chunk,
                        attention_mask=attention_mask_lm_nsp_chunk,
                        image_attention_mask=image_mask_chunk
                    )
                nsp_scores.append(nsp_scores_chunk)
            nsp_scores = torch.cat(nsp_scores, 0)

        else:
            losses['lm_loss'], losses['img_loss'], losses['nsp_loss'], nsp_scores = \
                self.model(
                    tokens,
                    image_feat,
                    image_loc,
                    image_edge_indices,
                    image_edge_attributes,
                    question_edge_indices,
                    question_edge_attributes,
                    question_limits,
                    history_edge_indices,
                    history_sep_indices,
                    next_sentence_label=next_sentence_labels,
                    image_target=image_target,
                    image_label=image_label,
                    sep_indices=sep_indices,
                    sep_len=sep_len,
                    token_type_ids=segments,
                    masked_lm_labels=mask,
                    attention_mask=attention_mask_lm_nsp,
                    image_attention_mask=image_mask
                )

        losses['tot_loss'] = 0
        for key in ['lm_loss', 'img_loss', 'nsp_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        output = {
            'losses': losses,
            'nsp_scores': nsp_scores
            }
        return output


class DenseRunner(Runner):
    def __init__(self, config):
        super(DenseRunner, self).__init__(config)
        self.model = VDGR(
            self.config['model_config'], self.config['device'], 
            use_apex=self.config['dp_type'] == 'apex', 
            cache_dir=self.config['bert_cache_dir'])

        if not(self.config['parallel'] and self.config['dp_type'] == 'dp'):
            self.model.to(self.config['device'])

        if self.config['dense_loss'] == 'ce':
            self.dense_loss = nn.KLDivLoss(reduction='batchmean')
        elif self.config['dense_loss'] == 'listmle':
            self.dense_loss = listMLE
        elif self.config['dense_loss'] == 'listnet':
            self.dense_loss = listNet
        elif self.config['dense_loss'] == 'approxndcg':
            self.dense_loss = approxNDCGLoss
        elif self.config['dense_loss'] == 'neural_ndcg':
            self.dense_loss = neuralNDCG
        elif self.config['dense_loss'] == 'neural_ndcg_transposed':
            self.dense_loss = neuralNDCG_transposed
        else:
            raise ValueError('dense_loss must be one of ce, listmle, listnet, approxndcg, neural_ndcg, neural_ndcg_transposed')

        if not self.config['validating'] or self.config['dp_type'] == 'apex':
            self.optimizer, self.scheduler = init_optim(self.model, self.config)

    def forward(self, batch, eval_visdial=False):
       # load data
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])
            elif isinstance(batch[key], list):
                if key != 'dialog_info':  # Do not send the dialog_info item to the gpu
                    batch[key] = [x.to(self.config['device']) for x in batch[key]]

        # get embedding and forward visdial
        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices']
        mask = batch['mask']
        hist_len = batch['hist_len']
        image_feat = batch['image_feat']
        image_loc = batch['image_loc']
        image_mask = batch['image_mask']
        next_sentence_labels = batch.get('next_sentence_labels', None)
        image_target = batch.get('image_target', None)
        image_label = batch.get('image_label', None)

        # load the graph data
        image_edge_indices = batch['image_edge_indices']
        image_edge_attributes = batch['image_edge_attributes']
        question_edge_indices = batch['question_edge_indices']
        question_edge_attributes = batch['question_edge_attributes']
        question_limits = batch['question_limits']
        history_edge_indices = batch['history_edge_indices']
        assert history_edge_indices[0].size(0) == 2
        history_sep_indices = batch['history_sep_indices']

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        losses = OrderedDict()

        if eval_visdial:
            num_lines = tokens.size(0)
            line_batch_size = self.config['eval_line_batch_size']
            num_line_batches = num_lines // line_batch_size
            if num_lines % line_batch_size > 0:
                num_line_batches += 1
            nsp_scores = []
            for j in range(num_line_batches):
                # create chunks of the original batch
                chunk_range = range(j*line_batch_size, min((j+1)*line_batch_size, num_lines))
                tokens_chunk = tokens[chunk_range]
                segments_chunk = segments[chunk_range]
                sep_indices_chunk = sep_indices[chunk_range]
                mask_chunk = mask[chunk_range]
                sep_len_chunk = sep_len[chunk_range]
                attention_mask_lm_nsp_chunk = attention_mask_lm_nsp[chunk_range]
                image_feat_chunk = image_feat[chunk_range]
                image_loc_chunk = image_loc[chunk_range]
                image_mask_chunk = image_mask[chunk_range]
                image_edge_indices_chunk = image_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                image_edge_attributes_chunk = image_edge_attributes[chunk_range[0]: chunk_range[-1]+1]
                question_edge_indices_chunk = question_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                question_edge_attributes_chunk = question_edge_attributes[chunk_range[0]: chunk_range[-1]+1]
                question_limits_chunk = question_limits[chunk_range[0]: chunk_range[-1]+1]
                history_edge_indices_chunk = history_edge_indices[chunk_range[0]: chunk_range[-1]+1]
                history_sep_indices_chunk = history_sep_indices[chunk_range[0]: chunk_range[-1]+1]

                _, _, _, nsp_scores_chunk = \
                    self.model(
                        tokens_chunk,
                        image_feat_chunk,
                        image_loc_chunk,
                        image_edge_indices_chunk,
                        image_edge_attributes_chunk,
                        question_edge_indices_chunk,
                        question_edge_attributes_chunk,
                        question_limits_chunk,
                        history_edge_indices_chunk,
                        history_sep_indices_chunk,
                        sep_indices=sep_indices_chunk,
                        sep_len=sep_len_chunk,
                        token_type_ids=segments_chunk,
                        masked_lm_labels=mask_chunk,
                        attention_mask=attention_mask_lm_nsp_chunk,
                        image_attention_mask=image_mask_chunk
                    )
                nsp_scores.append(nsp_scores_chunk)
            nsp_scores = torch.cat(nsp_scores, 0)

        else:
            _, _, _, nsp_scores = \
                self.model(
                    tokens,
                    image_feat,
                    image_loc,
                    image_edge_indices,
                    image_edge_attributes,
                    question_edge_indices,
                    question_edge_attributes,
                    question_limits,
                    history_edge_indices,
                    history_sep_indices,
                    next_sentence_label=next_sentence_labels,
                    image_target=image_target,
                    image_label=image_label,
                    sep_indices=sep_indices,
                    sep_len=sep_len,
                    token_type_ids=segments,
                    masked_lm_labels=mask,
                    attention_mask=attention_mask_lm_nsp,
                    image_attention_mask=image_mask
                )


        if nsp_scores is not None:
            nsp_scores_output = nsp_scores.detach().clone()
            if not eval_visdial:
                nsp_scores = nsp_scores.view(-1, self.config['num_options_dense'], 2)
            if 'next_sentence_labels' in batch and self.config['nsp_loss_coeff'] > 0:
                next_sentence_labels = batch['next_sentence_labels'].to(self.config['device'])
                losses['nsp_loss'] = F.cross_entropy(nsp_scores.view(-1,2), next_sentence_labels.view(-1)) 
            else:
                losses['nsp_loss'] = None

            if not eval_visdial:
                gt_relevance = batch['gt_relevance'].to(self.config['device'])
                nsp_scores = nsp_scores[:, :, 0]
                if self.config['dense_loss'] == 'ce':
                    losses['dense_loss'] = self.dense_loss(F.log_softmax(nsp_scores, dim=1), F.softmax(gt_relevance, dim=1))
                else:
                    losses['dense_loss'] = self.dense_loss(nsp_scores, gt_relevance)
            else:
                losses['dense_loss'] = None
        else:
            nsp_scores_output = None
            losses['nsp_loss'] = None
            losses['dense_loss'] = None

        losses['tot_loss'] = 0
        for key in ['nsp_loss', 'dense_loss']:
            if key in losses and losses[key] is not None:
                losses[key] = losses[key].mean()
                losses['tot_loss'] += self.config[f'{key}_coeff'] * losses[key]

        output = {
            'losses': losses,
            'nsp_scores': nsp_scores_output
            }

        return output
