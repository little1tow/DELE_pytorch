"""
@Author: zhkun
@Time:  16:05
@File: other_dataset
@Description: other datasets process and load file
@Something to attention
"""
import os

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import pickle
from transformers import AutoTokenizer

import csv
import pandas as pd
from tqdm import tqdm


class MSRPDataBert(Dataset):
    def __init__(self, args):
        self.label_dict = {'no': 0, 'yes': 1}
        self.args = args
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'dataset/sentence_pair'
        self.data_path = os.path.join(self.args.base_path, self.args.data_name)
        self.pkl_path = os.path.join(self.args.base_path, self.args.data_name, 'pkl_data')
        self.label_desp_path = os.path.join(self.args.base_path, self.args.data_name, f'{self.args.data_name}_label_desp.txt')

        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model, do_lower_case=True, cache_dir=self.args.cache_dir)
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.train_all_data = None
        self.label_content = None
        self.init_data()

    def load_desp(self):
        print('Loading description data....{}'.format(self.label_desp_path))

        content = []
        with open(self.label_desp_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label, desp = line.strip().split('\t')
                content.insert(self.label_dict[label], desp)

        self.label_content = content
        assert len(self.label_content) == self.args.num_classes

    def init_data(self):

        if self.label_content is None:
            self.load_desp()

        needed_file = ['train', 'dev', 'test', 'train-all']
        file_name = os.listdir(self.data_path)

        for name in file_name:
            if name not in needed_file:
                continue
            print(f'Initializing {name} data')
            pkl_name = f'{name}_data_{self.args.pre_trained_model}_pair.pkl'
            if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
                print(f'Found {pkl_name} data')
                with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                    if name == 'train':
                        self.train_data = pickle.load(f)
                    elif name == 'dev':
                        self.dev_data = pickle.load(f)
                    elif name == 'test':
                        self.test_data = pickle.load(f)
                    else:
                        self.train_all_data = pickle.load(f)
            else:
                current_data_path = os.path.join(self.data_path, name)
                if name == 'train':
                    self.train_data = self.load_data(current_data_path)
                    with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                        pickle.dump(self.train_data, f)
                elif name == 'dev':
                    self.dev_data = self.load_data(current_data_path)
                    with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                        pickle.dump(self.dev_data, f)
                elif name == 'test':
                    self.test_data = self.load_data(current_data_path)
                    with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                        pickle.dump(self.test_data, f)
                else:
                    self.train_all_data = self.load_data(current_data_path)
                    with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                        pickle.dump(self.train_all_data, f)

    def load_data(self, path):
        print('Loading data....{}'.format(path))
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        desp_tokens_ids = []
        desp_mask_ids = []
        desp_seg_ids = []
        all_labels = []

        sentence_a = []
        with open(os.path.join(path, 'a.toks'), 'r', encoding='utf-8') as f:
            for line in f:
                sentence_a.append(line.strip('\n'))

        sentence_b = []
        with open(os.path.join(path, 'b.toks'), 'r', encoding='utf-8') as f:
            for line in f:
                sentence_b.append(line.strip('\n'))

        labels = []
        with open(os.path.join(path, 'sim.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                labels.append(int(line.strip('\n')))

        assert len(sentence_a) == len(sentence_b) == len(labels)

        for a, b, l in zip(sentence_a, sentence_b, labels):
            premise, hypothesis = a, b

            premise_ids = self.tokenizer.encode(premise, add_special_tokens=False)
            hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
            pair_token_ids = [101] + premise_ids + [102] + hypothesis_ids + [
                102]  # 101-->[CLS], 102-->[SEP]. This is the format of sentence-pair embedding for BERT
            premise_len = len(premise_ids)  # the length does not consider the added SEP in the end
            hypothesis_len = len(hypothesis_ids)
            segment_ids = torch.tensor(
                [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

            # label desp
            desp_token = []
            desp_seg = []
            desp_mask = []
            for desp in self.label_content:
                desp_idx = self.tokenizer.encode(desp, add_special_tokens=True)
                desp_len = len(desp_idx)
                desp_seg_idx = torch.tensor([0] * desp_len)
                desp_mask_idx = torch.tensor([1] * desp_len)
                desp_token.append(torch.tensor(desp_idx))
                desp_seg.append(torch.tensor(desp_seg_idx))
                desp_mask.append(torch.tensor(desp_mask_idx))
            desp_token = pad_sequence(desp_token, batch_first=True)
            desp_seg = pad_sequence(desp_seg, batch_first=True)
            desp_mask = pad_sequence(desp_mask, batch_first=True)

            # all label index
            all_label_index = []
            for current_idx in range(self.args.num_classes):
                all_label_index.append(current_idx)
            all_label_index = torch.tensor(all_label_index)

            desp_tokens_ids.append(torch.tensor(desp_token))
            desp_seg_ids.append(torch.tensor(desp_seg))
            desp_mask_ids.append(torch.tensor(desp_mask))
            all_labels.append(torch.tensor(all_label_index))

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)

            y.append(l)

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        desp_tokens_ids = pad_sequence(desp_tokens_ids, batch_first=True)
        desp_seg_ids = pad_sequence(desp_seg_ids, batch_first=True)
        desp_mask_ids = pad_sequence(desp_mask_ids, batch_first=True)
        all_labels = pad_sequence(all_labels, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids, all_labels, y)
        print(len(dataset))
        return dataset

    def get_loader(self, type='train', batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            data = self.train_data
        else:
            data = self.test_data

        loader = DataLoader(
            data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return loader

    def get_train_test_loader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        dev_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return train_loader, dev_loader

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return train_loader, test_loader


class SentimentDataBert(Dataset):
    def __init__(self, args):
        self.label_dict = {}
        self.args = args
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'dataset/sentence_pair'
        self.data_path = os.path.join(self.args.base_path, self.args.data_name)
        self.pkl_path = os.path.join(self.args.base_path, self.args.data_name, 'pkl_data')
        if self.args.debug:
            self.train_data_path = os.path.join(self.args.base_path, self.args.data_name, 'test_clean.csv')
            self.args.pre_trained_model = 'bert-base-cased'
        else:
            self.train_data_path = os.path.join(self.args.base_path, self.args.data_name, 'train_clean.csv')
        self.test_data_path = os.path.join(self.args.base_path, self.args.data_name, 'test_clean.csv')
        self.label_desp_path = os.path.join(self.args.base_path, self.args.data_name, f'{self.args.data_name}_label_desp.txt')

        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model, do_lower_case=True, cache_dir=self.args.cache_dir)
        self.train_data = None
        self.test_data = None
        self.label_content = None
        self.init_data()

    def load_desp(self):
        print('Loading description data....{}'.format(self.label_desp_path))

        content = []
        with open(self.label_desp_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label, desp = line.strip().split('\t')
                self.label_dict[label] = idx
                content.insert(idx, desp)

        self.label_content = content
        assert len(self.label_content) == self.args.num_classes

    def init_data(self):

        if self.label_content is None:
            self.load_desp()

        load_data = self.load_csv_data

        print('Initializing test data')
        pkl_name = f'test_data_{self.args.pre_trained_model}_pair.pkl'
        if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
            print('Found test data')
            with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = load_data(self.test_data_path)
            with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                pickle.dump(self.test_data, f)

        print('Initializing train data')
        pkl_name = f'train_data_{self.args.pre_trained_model}_pair.pkl'
        if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
            print('Found train data')
            with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = load_data(self.train_data_path)
            with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
                pickle.dump(self.train_data, f)

    def load_csv_data(self, path):
        print('Loading data....{}'.format(path))
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        desp_tokens_ids = []
        desp_mask_ids = []
        desp_seg_ids = []
        all_labels = []

        with open(path, 'r', newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for idx, line in tqdm(enumerate(csv_reader)):

                if line[0] == '-':
                    continue

                label = int(line[0]) - 1
                title = line[1:]
                if self.args.data_name == 'ag_news':
                    sentence1 = title[0]
                    sentence1 = sentence1[:400] if len(sentence1) > 400 else sentence1
                    title_ids = self.tokenizer.encode(sentence1, add_special_tokens=True)
                    title_length = len(title_ids)
                    segment_ids = torch.tensor([0] * title_length)
                    attention_mask_ids = torch.tensor([1] * title_length)
                elif self.args.data_name == 'depedia':
                    if self.args.use_content:
                        sentence1 = title[0]
                        sentence1 = sentence1[:400] if len(sentence1) > 400 else sentence1
                        sentence2 = title[1]
                        sentence2 = sentence2[:400] if len(sentence2) > 400 else sentence2
                        sentence1_ids = self.tokenizer.encode(sentence1, add_special_tokens=False)
                        sentence2_ids = self.tokenizer.encode(sentence2, add_special_tokens=False)
                        title_ids = [101] + sentence1_ids + [102] + sentence2_ids + [102]
                        sentence1_len = len(sentence1_ids)
                        sentence2_len = len(sentence2_ids)
                        segment_ids = torch.tensor([0] * (sentence1_len + 2) + [1] * (sentence2_len + 1))
                        attention_mask_ids = torch.tensor([1] * (sentence1_len + sentence2_len + 3))
                    else:
                        sentence1 = title[0]
                        sentence1 = sentence1[:400] if len(sentence1) > 400 else sentence1
                        title_ids = self.tokenizer.encode(sentence1, add_special_tokens=True)
                        title_length = len(title_ids)
                        segment_ids = torch.tensor([0] * title_length)
                        attention_mask_ids = torch.tensor([1] * title_length)
                elif self.args.data_name == 'yahoo':
                    # sentence1 = title[0] + ' ' + title[1]
                    sentence1 = title[0]
                    sentence2 = title[2]
                    sentence1_ids = self.tokenizer.encode(sentence1, add_special_tokens=False)
                    sentence2_ids = self.tokenizer.encode(sentence2, add_special_tokens=False)

                    sentence1_len = len(sentence1_ids)
                    if sentence1_len >= 500:
                        continue
                    sentence2_len = len(sentence2_ids) if len(sentence2_ids) < 500-len(sentence1_ids) else 500-len(sentence1_ids)

                    title_ids = [101] + sentence1_ids + [102] + sentence2_ids[:sentence2_len] + [102]

                    segment_ids = torch.tensor([0] * (sentence1_len + 2) + [1] * (sentence2_len + 1))
                    attention_mask_ids = torch.tensor([1] * (sentence1_len + sentence2_len + 3))
                else:
                    raise ValueError('wrong dataset name, please check again')

                desp_token = []
                desp_seg = []
                desp_mask = []
                for desp in self.label_content:
                    desp_idx = self.tokenizer.encode(desp, add_special_tokens=True)
                    desp_len = len(desp_idx)
                    desp_seg_idx = torch.tensor([0] * desp_len)
                    desp_mask_idx = torch.tensor([1] * desp_len)
                    desp_token.append(torch.tensor(desp_idx))
                    desp_seg.append(torch.tensor(desp_seg_idx))
                    desp_mask.append(torch.tensor(desp_mask_idx))
                desp_token = pad_sequence(desp_token, batch_first=True)
                desp_seg = pad_sequence(desp_seg, batch_first=True)
                desp_mask = pad_sequence(desp_mask, batch_first=True)

                # all label index
                all_label_index = []
                for current_idx in range(self.args.num_classes):
                    all_label_index.append(current_idx)
                all_label_index = torch.tensor(all_label_index)

                token_ids.append(torch.tensor(title_ids))
                seg_ids.append(segment_ids)
                mask_ids.append(attention_mask_ids)
                y.append(label)

                desp_tokens_ids.append(torch.tensor(desp_token))
                desp_seg_ids.append(torch.tensor(desp_seg))
                desp_mask_ids.append(torch.tensor(desp_mask))
                all_labels.append(torch.tensor(all_label_index))

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)

        desp_tokens_ids = pad_sequence(desp_tokens_ids, batch_first=True)
        desp_seg_ids = pad_sequence(desp_seg_ids, batch_first=True)
        desp_mask_ids = pad_sequence(desp_mask_ids, batch_first=True)
        all_labels = pad_sequence(all_labels, batch_first=True)

        dataset = TensorDataset(token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids, all_labels,
                                y)
        print(len(dataset))
        return dataset

    def get_loader(self, type='train', batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            data = self.train_data
        else:
            data = self.test_data

        loader = DataLoader(
            data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return loader

    def get_train_test_loader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        dev_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return train_loader, dev_loader

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return train_loader, test_loader


class YahooDataSet(Dataset):
    def __init__(self, args, split='train'):
        self.label_dict = {}
        self.args = args
        self.split = split
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'dataset/sentence_pair'

        self.data_path = os.path.join(self.args.base_path, self.args.data_name)
        self.pkl_path = os.path.join(self.args.base_path, self.args.data_name, 'pkl_data')

        if self.split == 'train':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name, 'train_qqclean.csv')
        elif self.split == 'test':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name, 'test_qqclean.csv')
        else:
            raise ValueError('wrong file name, please try again')

        self.label_desp_path = os.path.join(self.args.base_path, self.args.data_name,
                                            f'{self.args.data_name}_label_desp.txt')

        lines = open(self.data_path, 'r', encoding='utf8')
        self.total_length = sum(1 for i in lines) - 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pre_trained_model, do_lower_case=True, cache_dir=self.args.cache_dir)

        self.origin_pddata = pd.read_csv(self.data_path, ';\t;')

        self.load_desp()

    def load_desp(self):
        print('Loading description data....{}'.format(self.label_desp_path))

        content = []
        with open(self.label_desp_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label, desp = line.strip().split('\t')
                self.label_dict[label] = idx
                content.insert(idx, desp)

        self.label_content = content
        assert len(self.label_content) == self.args.num_classes

    def __getitem__(self, index):

        current_sample = self.origin_pddata.loc[index]
        sentence1 = current_sample['text'].split(' ')
        if len(sentence1) > 300:
            sentence1 = ' '.join(sentence1[:300])
        else:
            sentence1 = ' '.join(sentence1)

        label_index = int(current_sample['label']) - 1

        try:
            values = self.process_sent(sentence1)

            desp_token = []
            desp_seg = []
            desp_mask = []
            for single_desp in self.label_content:
                desp_values = self.process_sent(single_desp)
                if len(desp_values) == 3:
                    desp_token.append(desp_values[0])
                    desp_seg.append(desp_values[1])
                    desp_mask.append(desp_values[2])
                else:
                    desp_token.append(desp_values[0])
                    desp_mask.append(desp_values[1])

            desp_token = pad_sequence(desp_token, batch_first=True)
            desp_mask = pad_sequence(desp_mask, batch_first=True)
            if len(desp_seg) != 0:
                desp_seg = pad_sequence(desp_seg, batch_first=True)

            # all label index
            all_label_index = []
            for current_idx in range(self.args.num_classes):
                all_label_index.append(current_idx)
            all_label_index = torch.tensor(all_label_index)

            if len(values) == 2:
                return values[0], values[1], desp_token, desp_mask, all_label_index, label_index
            else:
                return values[0], values[1], values[2], desp_token, desp_seg, desp_mask, all_label_index, label_index
        except TypeError:
            print(sentence1)

    def __len__(self):
        return self.total_length

    def tokenizers(self, sentence):
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        return tokens

    def seg_mask(self, sent1_id, sent2_id=None):
        if sent2_id is not None:
            sent1_len = len(sent1_id)
            sent2_len = len(sent2_id)

            segment_ids = torch.tensor(
                [0] * (sent1_len + 2) + [1] * (sent2_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (sent1_len + sent2_len + 3))
        else:
            sent1_len = len(sent1_id)

            segment_ids = torch.tensor([0] * (sent1_len + 2))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (sent1_len + 2))

        return segment_ids, attention_mask_ids

    def process_sent(self, sent1, sent2=None):
        if sent2 is not None:
            sentence1_ids = self.tokenizer.encode(sent1, add_special_tokens=False)
            sentence2_ids = self.tokenizer.encode(sent2, add_special_tokens=False)

            sentence1_len = len(sentence1_ids)
            if sentence1_len >= 500:
                sentence1_ids = sentence1_ids[:300]
                sentence1_len = len(sentence1_ids)
            sentence2_len = len(sentence2_ids) if len(sentence2_ids) < 500 - len(sentence1_ids) else 500 - len(sentence1_ids)

            title_ids = [101] + sentence1_ids + [102] + sentence2_ids[:sentence2_len] + [102]

            segment_ids = torch.tensor([0] * (sentence1_len + 2) + [1] * (sentence2_len + 1))
            attention_mask_ids = torch.tensor([1] * (sentence1_len + sentence2_len + 3))
        else:
            sentence1_ids = self.tokenizer.encode(sent1, add_special_tokens=False)

            sentence1_len = len(sentence1_ids)
            if sentence1_len >= 500:
                sentence1_ids = sentence1_ids[:500]

            segment_ids, attention_mask_ids = self.seg_mask(sentence1_ids)
            title_ids = [101] + sentence1_ids + [102]
        title_ids = torch.tensor(title_ids)
        segment_ids = torch.tensor(segment_ids)
        attention_mask_ids = torch.tensor(attention_mask_ids)

        values = [title_ids, segment_ids, attention_mask_ids]

        return values


class SentimentDataBertV2(Dataset):
    def __init__(self, args):
        self._args = args
        if self._args.debug:
            self._train_set = YahooDataSet(self._args, split='dev')
        else:
            self._train_set = YahooDataSet(self._args, split='train')

        self._test_set = YahooDataSet(self._args, split='test')

    def get_dataloaders(self, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = self._get_loader(
            batch_size=batch_size,
            type='train',
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = self._get_loader(
            batch_size=batch_size,
            type='test',
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, test_loader

    def _get_loader(self, batch_size, type='train', shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            current_dataset = self._train_set
        elif type == 'test':
            current_dataset = self._test_set
        else:
            raise ValueError

        loader = DataLoader(
            current_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._custom_fn,
            pin_memory=pin_memory,
            drop_last=True
        )

        return loader

    def _custom_fn(self, batch):
        # print(len(batch))
        # print(len(batch[0]))

        input_text = []
        for idx in range(len(batch[0]) - 1):
            if idx == 1:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            else:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True))

        labels = torch.tensor([item[-1] for item in batch])

        # return input_text, labels
        if len(input_text) == 5:
            return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], labels
        else:
            return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], input_text[5], input_text[6], labels
