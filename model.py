"""
@Author: zhkun
@Time:  16:27
@File: model
@Description: model structure file
@Something to attention
"""
import torch
from torch import nn
import json
import photinia as ph
import os

from torch.nn.parameter import Parameter
from transformers import AutoModel
from util_classes import ProjectionHead, ResProjectionHead


class LED(nn.Module):
    def __init__(self, config):
        super(LED, self).__init__()
        self.config = config

        use_cuda = config.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.bert = AutoModel.from_pretrained(config.pre_trained_model,
                                              cache_dir=config.cache_dir,
                                              output_hidden_states=True,
                                              output_attentions=False)

        if config.use_desp and config.desp_seperate:
            # self.desp_bert = AutoModel.from_pretrained(config.pre_trained_model, output_hidden_states=True,
            #                                       output_attentions=False)
            self.desp_bert = AutoModel.from_pretrained(config.pre_trained_model,
                                                       cache_dir=config.cache_dir,
                                                       output_hidden_states=True,
                                                       output_attentions=False)

            for param in self.desp_bert.parameters():
                param.requires_grad = False

        if self.config.num_bert_layers == 0:
            self.num_hidden_layers = self.bert.config.num_hidden_layers
        else:
            self.num_hidden_layers = self.bert.config.num_hidden_layers if self.config.num_bert_layers > self.bert.config.num_hidden_layers else self.config.num_bert_layers

        # weight average
        if config.use_sentence_weight:
            self.sentence_weight = Parameter(torch.Tensor(1, self.num_hidden_layers))
            self.sentence_weight = nn.init.normal(self.sentence_weight)

        # Label free embedding
        self.label_embedding = nn.Embedding(config.num_classes, config.in_features)
        self.label_emb_size = config.in_features
        # Label description embedding
        if config.use_desp:
            if config.desp_type == 'atten':
                self.label_atten = ph.nn.MLPAttention(
                    key_size=config.in_features,
                    attention_size=config.attention_size,
                    query_vec_size=config.in_features,
                    use_norm=True
                )

            if config.label_weight:
                self.label_emb_weight = nn.Parameter(torch.tensor(0.5))
            if config.label_fuse_type == 'concat':
                self.label_emb_size += config.in_features
            else:
                self.label_emb_size = config.in_features

        if config.mi_type in ['mi', 'self']:
            self.label_semantic_atten = ph.nn.MLPAttention(
                key_size=self.label_emb_size,
                attention_size=config.attention_size,
                query_vec_size=config.in_features,
                use_norm=False
            )

        self.sentence_semantic_atten = ph.nn.MLPAttention(
            key_size=config.in_features,
            attention_size=config.attention_size,
            query_vec_size=self.label_emb_size,
            use_norm=False
        )
        if config.label_fuse_type == 'concat':
            self.sentence_semantic_projection = nn.Linear(config.in_features, self.label_emb_size, True)

        # projection
        if config.head == 'default':
            self.proj_head = ProjectionHead(self.label_emb_size, config.proj_size)
        else:
            self.proj_head = ResProjectionHead(self.label_emb_size, config.proj_size)

        # prediction
        if config.sentence_rep == 'concat':
            predict_in_features = self.label_emb_size * 2
        else:
            predict_in_features = self.label_emb_size

        if config.num_pred_layer == 1:
            self.pred_layer = nn.Linear(in_features=predict_in_features, out_features=config.num_classes)
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(in_features=predict_in_features, out_features=config.mlp_size),
                nn.Tanh(),
                # nn.ReLU(),
                nn.Linear(in_features=config.mlp_size, out_features=config.num_classes)
            )

        if config.r2_loss in ['double', 'single']:
            if config.num_pred_layer == 1:
                self.pred_r2 = nn.Linear(in_features=predict_in_features*4, out_features=2)
            else:
                self.pred_r2 = nn.Sequential(
                    nn.Linear(in_features=predict_in_features*4, out_features=config.mlp_size*2),
                    nn.Tanh(),
                    # nn.ReLU(),
                    nn.Linear(in_features=config.mlp_size*2, out_features=2)
                )
        self.dropout = nn.Dropout(p=config.dropout)

        # not only for display, but also collect different parameters that needs updating
        for param in self.bert.parameters():
            param.requires_grad = False

        self.req_grad_params = self.get_req_grad_params(debug=True)

        if config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = True

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, sentence_pairs, all_labels, label_desp, is_train=True):

        if len(sentence_pairs) == 3:
            token_ids = sentence_pairs[0]
            segment_ids = sentence_pairs[1]
            attention_masks = sentence_pairs[2]
            bert_output = self.bert(token_ids, token_type_ids=segment_ids,
                                    attention_mask=attention_masks)
        else:
            token_ids = sentence_pairs[0]
            attention_masks = sentence_pairs[1]
            bert_output = self.bert(token_ids, attention_mask=attention_masks)

        bert_output_dropout = {}
        if is_train:
            for item in bert_output:
                if type(bert_output[item]) is not tuple:
                    bert_output_dropout[item] = self.dropout(bert_output[item])
                else:
                    bert_output_dropout[item] = [self.dropout(z) for z in bert_output[item]]
        else:
            bert_output_dropout = bert_output

        if self.config.use_sentence_weight:
           rep_outputs = torch.stack(bert_output_dropout['hidden_states'][-self.num_hidden_layers:], dim=2)
           rep_outputs = torch.matmul(self.sentence_weight, rep_outputs)
           rep_outputs = rep_outputs.squeeze()
           cls_emb, word_embs = rep_outputs[:, 0], rep_outputs[:, 1:]
        else:
            cls_emb, word_embs = bert_output_dropout['last_hidden_state'][:, 0], bert_output_dropout['last_hidden_state'][:, 1:]

        # label embeddings
        current_label_emb = self.label_embedding(all_labels)
        if self.config.use_desp:
            if self.config.desp_seperate:
                transformers = self.desp_bert
            else:
                transformers = self.bert
            desp_outputs = []
            for desp_idx in range(self.config.num_classes):
                if len(label_desp) == 3:
                    item_tokens = label_desp[0][:, desp_idx]
                    item_seg = label_desp[1][:, desp_idx]
                    item_mask = label_desp[2][:, desp_idx]
                    item_output = transformers(item_tokens, token_type_ids=item_seg, attention_mask=item_mask)['last_hidden_state']
                else:
                    item_tokens = label_desp[0][:, desp_idx]
                    item_mask = label_desp[1][:, desp_idx]
                    item_output = transformers(item_tokens, attention_mask=item_mask)['last_hidden_state']
                if self.config.desp_type == 'atten':
                    item_resp, _ = self.label_atten(
                        key=item_output,
                        value=item_output,
                        key_mask=item_mask
                    )
                    desp_outputs.append(item_resp)
                elif self.config.desp_type == 'adaptive':
                    item_resp, _ = self.label_atten(
                        key=item_output,
                        value=item_output,
                        query_vec=cls_emb,
                        key_mask=item_mask
                    )
                    desp_outputs.append(item_resp)
                elif self.config.desp_type == 'default':
                    desp_outputs.append(item_output[:, 0])
                else:
                    desp_outputs.append(torch.mean(item_output, dim=1))
            desp_outputs = torch.stack(desp_outputs, dim=1)
            if self.config.label_weight:
                desp_outputs = self.label_emb_weight * desp_outputs
            if self.config.label_fuse_type == 'concat':
                label_emb = torch.cat([current_label_emb, desp_outputs], dim=-1)
            else:
                label_emb = torch.add(current_label_emb, desp_outputs)
        else:
            label_emb = current_label_emb

        if self.config.mi_type == 'mi':
            label_semantic, _ = self.label_semantic_atten(
                key=label_emb,
                value=label_emb,
                query_vec=cls_emb,
            )
        elif self.config.mi_type == 'self':
            label_semantic, _ = self.label_semantic_atten(
                key=label_emb,
                value=label_emb,
                # query_vec=cls_emb,
            )
        elif self.config.mi_type == 'max':
            label_semantic = torch.max(label_emb, dim=1)[0]
        elif self.config.mi_type == 'mean':
            label_semantic = torch.mean(label_emb, dim=1)
        else:
            raise ValueError('wrong key words for mi_type, please try again')

        # label enhanced sentence representation-> [batch_size, representation_size]
        sentence_semantic = []

        for idx in range(self.config.num_classes):
            semantics, _ = self.sentence_semantic_atten(
                key=word_embs,
                value=word_embs,
                query_vec=label_emb[:, idx],
                key_mask=attention_masks[:, 1:]
            )
            sentence_semantic.append(semantics)

        sentence_semantic = torch.stack(sentence_semantic, dim=1)

        if self.config.sentence_pooling == 'sum':
            sentence_semantic = torch.sum(sentence_semantic, dim=1)
        elif self.config.sentence_pooling == 'max':
            sentence_semantic = torch.max(sentence_semantic, dim=1)[0]
        else:
            sentence_semantic = torch.mean(sentence_semantic, dim=1)

        if self.config.label_fuse_type == 'concat':
            sentence_semantic = self.sentence_semantic_projection(sentence_semantic)
            sentence_semantic = torch.relu(sentence_semantic)

        z1 = self.proj_head(label_semantic)
        z2 = self.proj_head(sentence_semantic)

        if self.config.sentence_rep == 'sentence':
            predict = self.pred_layer(sentence_semantic)
        elif self.config.sentence_rep == 'label':
            predict = self.pred_layer(label_semantic)
        else:
            predict = self.pred_layer(torch.cat([sentence_semantic, label_semantic], dim=-1))

        label_semantic_dropout = self.dropout(label_semantic)
        sentence_semantic_dropout = self.dropout(sentence_semantic)
        if self.config.r2_loss == 'double':
            r2_predict = self.pred_r2(self._fusing(label_semantic_dropout))
            r2_predict2 = self.pred_r2(self._fusing(sentence_semantic_dropout))
            return [predict, r2_predict, r2_predict2, z1, z2, label_semantic, sentence_semantic]
        elif self.config.r2_loss == 'single':
            r2_predict = self.pred_r2(self._fusing(label_semantic_dropout))
            return [predict, r2_predict, z1, z2, label_semantic, sentence_semantic]
        else:
            return [predict, z1, z2, label_semantic, sentence_semantic]

    def _fusing(self, label_semantic):
        slice_point = int(self.config.batch_size/2)
        part1 = label_semantic[:slice_point, :]
        part2 = label_semantic[slice_point:, :]

        concat = torch.cat([part1, part2], dim=-1)
        cdot = torch.mul(part1, part2)
        mini = torch.abs(part1 - part2)
        results = torch.cat([concat, cdot, mini], dim=-1)

        return results

    def get_req_grad_params(self, debug=False):
        print(f'#{self.config.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())  # the product of all dimensions, i.e., # of parameters
                total_size += n_params
                if debug:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')
        print('{:,}'.format(total_size))
        return params


class Base_model(nn.Module):
    def __init__(self, config):
        super(Base_model, self).__init__()
        self.config = config

        use_cuda = config.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.bert = AutoModel.from_pretrained(config.pre_trained_model, cache_dir=config.cache_dir,
                                              output_hidden_states=True, output_attentions=False)

        if self.config.num_bert_layers == 0:
            self.num_hidden_layers = self.bert.config.num_hidden_layers
        else:
            self.num_hidden_layers = self.bert.config.num_hidden_layers if self.config.num_bert_layers > self.bert.config.num_hidden_layers else self.config.num_bert_layers

        # weight average
        if config.use_sentence_weight:
            self.sentence_weight = Parameter(torch.Tensor(1, self.num_hidden_layers))
            self.sentence_weight = nn.init.normal(self.sentence_weight)

        if not self.config.use_output_pooling:
            self.sentence_semantic_atten = ph.nn.MLPAttention(
                key_size=config.in_features,
                attention_size=config.attention_size,
                use_norm=False
            )

        self.pred_layer = nn.Linear(config.in_features, config.num_classes)

        self.dropout = nn.Dropout(p=config.dropout)

        # for display
        for param in self.bert.parameters():
            param.requires_grad = False

        self.req_grad_params = self.get_req_grad_params(debug=True)

        if config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = True

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, sentence_pairs, is_train=True):

        if len(sentence_pairs) == 3:
            bert_output = self.bert(sentence_pairs[0], token_type_ids=sentence_pairs[1], attention_mask=sentence_pairs[2])
        else:
            bert_output = self.bert(sentence_pairs[0], attention_mask=sentence_pairs[1])

        if is_train:
            bert_output_dropout = {}
            for item in bert_output:
                if type(bert_output[item]) is not tuple:
                    bert_output_dropout[item] = self.dropout(bert_output[item])
                else:
                    bert_output_dropout[item] = [self.dropout(z) for z in bert_output[item]]
        else:
            bert_output_dropout = bert_output

        # sentence embedding and word embeddings
        if self.config.use_sentence_weight:
           rep_outputs = torch.stack(bert_output_dropout['hidden_states'][-self.num_hidden_layers:], dim=2)
           rep_outputs = torch.matmul(self.sentence_weight, rep_outputs)
           rep_outputs = rep_outputs.squeeze()
           cls_emb, word_embs = rep_outputs[:, 0], rep_outputs[:, 1:]
        elif self.config.use_output_pooling:
            word_embs = bert_output_dropout['pooler_output']
            cls_emb = bert_output_dropout['last_hidden_state'][:, 0]
        else:
            cls_emb, word_embs = bert_output_dropout['last_hidden_state'][:, 0], bert_output_dropout['last_hidden_state'][:, 1:]

        if not self.config.use_output_pooling:
            sentence_semantic, _ = self.sentence_semantic_atten(
                key=word_embs,
                value=word_embs,
                key_mask=sentence_pairs[2][:, 1:]
            )
        else:
            sentence_semantic = cls_emb

        predict = self.pred_layer(sentence_semantic)

        return predict, sentence_semantic

    def get_req_grad_params(self, debug=False):
        print(f'# {self.config.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())  # the product of all dimensions, i.e., # of parameters
                total_size += n_params
                if debug:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')
        print('{:,}'.format(total_size))
        return params


class LEDv2(nn.Module):
    def __init__(self, config):
        super(LEDv2, self).__init__()
        self.config = config

        use_cuda = config.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.bert = AutoModel.from_pretrained(config.pre_trained_model,
                                              cache_dir=config.cache_dir,
                                              output_hidden_states=True,
                                              output_attentions=False)

        if self.config.num_bert_layers == 0:
            self.num_hidden_layers = self.bert.config.num_hidden_layers
        else:
            self.num_hidden_layers = self.bert.config.num_hidden_layers if self.config.num_bert_layers > self.bert.config.num_hidden_layers else self.config.num_bert_layers

        # weight average
        if config.use_sentence_weight:
            self.sentence_weight = Parameter(torch.Tensor(1, self.num_hidden_layers))
            self.sentence_weight = nn.init.normal(self.sentence_weight)

        # Label free embedding
        self.label_embedding = nn.Embedding(config.num_classes, config.in_features)
        self.label_emb_size = config.in_features
        # Label description embedding
        if config.use_desp:
            if config.desp_type == 'atten':
                self.label_atten = ph.nn.MLPAttention(
                    key_size=config.in_features,
                    attention_size=config.attention_size,
                    query_vec_size=config.in_features,
                    use_norm=True
                )

            if config.label_weight:
                self.label_emb_weight = nn.Parameter(torch.tensor(0.5))
            if config.label_fuse_type == 'concat':
                self.label_emb_size += config.in_features
            else:
                self.label_emb_size = config.in_features

        # cls enhanced attention for label representation
        self.sentence_semantic_atten = ph.nn.MLPAttention(
            key_size=config.in_features,
            attention_size=config.attention_size,
            query_vec_size=self.label_emb_size,
            use_norm=False
        )
        if config.label_fuse_type == 'concat':
            self.sentence_semantic_projection = nn.Linear(config.in_features, self.label_emb_size, True)

        # projection
        if config.head == 'default':
            self.proj_head = ProjectionHead(self.label_emb_size, config.proj_size)
        else:
            self.proj_head = ResProjectionHead(self.label_emb_size, config.proj_size)

        # prediction
        if config.sentence_rep == 'concat':
            predict_in_features = self.label_emb_size * 2
        else:
            predict_in_features = self.label_emb_size

        if config.num_pred_layer == 1:
            self.pred_layer = nn.Linear(in_features=predict_in_features, out_features=config.num_classes)
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(in_features=predict_in_features, out_features=config.mlp_size),
                nn.Tanh(),
                # nn.ReLU(),
                nn.Linear(in_features=config.mlp_size, out_features=config.num_classes)
            )
        self.dropout = nn.Dropout(p=config.dropout)

        # not only for display, but also collect different parameters that needs updating
        for param in self.bert.parameters():
            param.requires_grad = False

        self.req_grad_params = self.get_req_grad_params(debug=True)

        if config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = True

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def single_process(self, sentence_pairs, all_labels, is_train=True):
        if len(sentence_pairs) == 3:
            token_ids = sentence_pairs[0]
            segment_ids = sentence_pairs[1]
            attention_masks = sentence_pairs[2]
            bert_output = self.bert(token_ids, token_type_ids=segment_ids,
                                    attention_mask=attention_masks)
        else:
            token_ids = sentence_pairs[0]
            attention_masks = sentence_pairs[1]
            bert_output = self.bert(token_ids, attention_mask=attention_masks)

        bert_output_dropout = {}
        if is_train:
            for item in bert_output:
                if type(bert_output[item]) is not tuple:
                    bert_output_dropout[item] = self.dropout(bert_output[item])
                else:
                    bert_output_dropout[item] = [self.dropout(z) for z in bert_output[item]]
        else:
            bert_output_dropout = bert_output

        # sentence embedding and word embeddings
        if self.config.use_sentence_weight:
           rep_outputs = torch.stack(bert_output_dropout['hidden_states'][-self.num_hidden_layers:], dim=2)
           rep_outputs = torch.matmul(self.sentence_weight, rep_outputs)
           rep_outputs = rep_outputs.squeeze()
           cls_emb, word_embs = rep_outputs[:, 0], rep_outputs[:, 1:]
        else:
            cls_emb, word_embs = bert_output_dropout['last_hidden_state'][:, 0], bert_output_dropout['last_hidden_state'][:, 1:]

        current_label_emb = self.label_embedding(all_labels)

        label_emb = current_label_emb

        sentence_semantic = []

        for idx in range(self.config.num_classes):
            semantics, _ = self.sentence_semantic_atten(
                key=word_embs,
                value=word_embs,
                query_vec=label_emb[:, idx],
                key_mask=attention_masks[:, 1:]
            )
            sentence_semantic.append(semantics)

        sentence_semantic = torch.stack(sentence_semantic, dim=1)

        if self.config.sentence_pooling == 'sum':
            sentence_semantic = torch.sum(sentence_semantic, dim=1)
        elif self.config.sentence_pooling == 'max':
            sentence_semantic = torch.max(sentence_semantic, dim=1)[0]
        else:
            sentence_semantic = torch.mean(sentence_semantic, dim=1)

        if self.config.label_fuse_type == 'concat':
            sentence_semantic = self.sentence_semantic_projection(sentence_semantic)
            sentence_semantic = torch.relu(sentence_semantic)

        z = self.proj_head(sentence_semantic)

        predict = self.pred_layer(sentence_semantic)

        return predict, z, sentence_semantic

    def forward(self, sentence_pairs, all_labels, label_desp, is_train=True):

        predict1, z1, sentence_semantic1 = self.single_process(sentence_pairs, all_labels)
        predict2, z2, sentence_semantic2 = self.single_process(sentence_pairs, all_labels)

        predict = torch.mean(torch.stack([predict1, predict2], dim=0), dim=0)

        return predict, z1, z2, sentence_semantic1, sentence_semantic2

    def get_req_grad_params(self, debug=False):
        print(f'#{self.config.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())  # the product of all dimensions, i.e., # of parameters
                total_size += n_params
                if debug:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')
        print('{:,}'.format(total_size))
        return params




