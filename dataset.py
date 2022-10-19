"""
@Author: zhkun
@Time:  16:27
@File: dataset
@Description: 数据集文件
@Something to attention
"""
import os

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pickle
import pandas as pd
from transformers import AutoTokenizer

import json


class SickDataBert(Dataset):
    def __init__(self, args):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.args = args
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'dataset/sentence_pair'
        self.pkl_path = os.path.join(self.args.base_path, self.args.data_name, 'pkl_data')
        self.data_path = os.path.join(self.args.base_path, self.args.data_name)
        if self.args.data_name == 'sick':
            self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            self.train_data_path = os.path.join(self.data_path, 'SICK_train.txt')
            self.dev_data_path = os.path.join(self.data_path, 'SICK_trial.txt')
            self.test_data_path = os.path.join(self.data_path, 'SICK_test_annotated.txt')
            self.label_desp_path = os.path.join(self.data_path, 'sick_label_desp.txt')
        elif self.args.data_name == 'scitail':
            self.label_dict = {'entailment': 0, 'neutral': 1}
            self.train_data_path = os.path.join(self.data_path, 'scitail_1.0_train.txt')
            self.dev_data_path = os.path.join(self.data_path, 'scitail_1.0_dev.txt')
            self.test_data_path = os.path.join(self.data_path, 'scitail_1.0_test.txt')
            self.label_desp_path = os.path.join(self.data_path, 'scitail_label_desp.txt')
        else:
            # Quora dataset
            self.label_dict = {'no': 0, 'yes': 1}
            self.train_data_path = os.path.join(self.data_path, 'train.tsv')
            self.dev_data_path = os.path.join(self.data_path, 'dev.tsv')
            self.test_data_path = os.path.join(self.data_path, 'test.tsv')
            self.label_desp_path = os.path.join(self.data_path, f'{self.args.data_name}_label_desp.txt')

        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model, do_lower_case=True, cache_dir=self.args.cache_dir)
        self.train_data = None
        self.dev_data = None
        self.test_data = None
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

    def write_file(self, name, content):
        if not os.path.exists(self.pkl_path):
            os.makedirs(self.pkl_path)

        with open(os.path.join(self.pkl_path, name), 'wb') as f:
            pickle.dump(content, f)

    def init_data(self):

        if self.label_content is None:
            self.load_desp()

        if self.args.data_name == 'sick':
            load_func = self.load_data
        elif self.args.data_name == 'scitail':
            load_func = self.load_jsonl_data
        else:
            load_func = self.load_txt_data

        print('Initializing dev data')
        pkl_name = f'dev_data_{self.args.pre_trained_model}_pair.pkl'
        if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
            print('Found dev data')
            with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                self.dev_data = pickle.load(f)
        else:
            self.dev_data = load_func(self.dev_data_path)
            self.write_file(pkl_name, self.dev_data)
            # with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
            #     pickle.dump(self.dev_data, f)

        print('Initializing test data')
        pkl_name = f'test_data_{self.args.pre_trained_model}_pair.pkl'
        if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
            print('Found test data')
            with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = load_func(self.test_data_path)
            self.write_file(pkl_name, self.test_data)
            # with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
            #     pickle.dump(self.test_data, f)

        print('Initializing train data')
        pkl_name = f'train_data_{self.args.pre_trained_model}_pair.pkl'
        if os.path.exists(os.path.join(self.pkl_path, pkl_name)):
            print('Found train data')
            with open(os.path.join(self.pkl_path, pkl_name), 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = load_func(self.train_data_path)
            self.write_file(pkl_name, self.train_data)
            # with open(os.path.join(self.pkl_path, pkl_name), 'wb') as f:
            #     pickle.dump(self.train_data, f)

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

        with open(path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # skip the first line
                if idx == 0:
                    continue
                if idx % 5000 == 0:
                    print('{}'.format(idx, ',d'))

                cols = line.strip().split('\t')

                #   '–' indicates a lack of consensus from the human annotators, ignore it
                if cols[4] == '-':
                    continue

                premise, hypothesis = cols[1], cols[2]
                premise_ids = self.tokenizer.encode(premise, add_special_tokens=False)
                hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
                pair_token_ids = [101] + premise_ids + [102] + hypothesis_ids + [
                    102]  # 101-->[CLS], 102-->[SEP]. This is the format of sentence-pair embedding for BERT
                premise_len = len(premise_ids)  # the length does not consider the added SEP in the end
                hypothesis_len = len(hypothesis_ids)
                segment_ids = torch.tensor(
                    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
                attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

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

                token_ids.append(torch.tensor(pair_token_ids))
                seg_ids.append(segment_ids)
                mask_ids.append(attention_mask_ids)
                y.append(self.label_dict[cols[4].lower()])

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

        dataset = TensorDataset(token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids, all_labels, y)
        print(len(dataset))
        return dataset

    def load_jsonl_data(self, path):
        print('Loading data....{}'.format(path))
        token_ids = []
        mask_ids = []
        seg_ids = []

        desp_tokens_ids = []
        desp_mask_ids = []
        desp_seg_ids = []
        all_labels = []

        y = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                example = json.loads(line)

                premise = example['sentence1']
                hypothesis = example['sentence2']
                label = example['gold_label']

                if label == '-':
                    continue

                premise_ids = self.tokenizer.encode(premise, add_special_tokens=False)
                hypothesis_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
                pair_token_ids = [101] + premise_ids + [102] + hypothesis_ids + [
                    102]  # 101-->[CLS], 102-->[SEP]. This is the format of sentence-pair embedding for BERT
                premise_len = len(premise_ids)  # the length does not consider the added SEP in the end
                hypothesis_len = len(hypothesis_ids)
                segment_ids = torch.tensor(
                    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
                attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

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
                for idx in range(self.args.num_classes):
                    all_label_index.append(idx)
                all_label_index = torch.tensor(all_label_index)

                desp_tokens_ids.append(torch.tensor(desp_token))
                desp_seg_ids.append(torch.tensor(desp_seg))
                desp_mask_ids.append(torch.tensor(desp_mask))
                all_labels.append(torch.tensor(all_label_index))

                token_ids.append(torch.tensor(pair_token_ids))
                seg_ids.append(segment_ids)
                mask_ids.append(attention_mask_ids)
                y.append(self.label_dict[label])

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

    def load_txt_data(self, path):
        print('Loading data....{}'.format(path))
        token_ids = []
        mask_ids = []
        seg_ids = []
        desp_tokens_ids = []
        desp_mask_ids = []
        desp_seg_ids = []
        all_labels = []
        y = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx % 5000 == 0:
                    print('{}'.format(idx))

                cols = line.strip().split('\t')

                premise, hypothesis = cols[1], cols[2]
                premise_ids = self.tokenizer.encode(premise)
                hypothesis_ids = self.tokenizer.encode(hypothesis)
                pair_token_ids = [101] + premise_ids + [102] + hypothesis_ids + [
                    102]  # 101-->[CLS], 102-->[SEP]. This is the format of sentence-pair embedding for BERT
                premise_len = len(premise_ids)  # the length does not consider the added SEP in the end
                hypothesis_len = len(hypothesis_ids)
                segment_ids = torch.tensor(
                    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
                attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

                token_ids.append(torch.tensor(pair_token_ids))
                seg_ids.append(segment_ids)
                mask_ids.append(attention_mask_ids)
                y.append(int(cols[0]))

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
                for idx in range(self.args.num_classes):
                    all_label_index.append(idx)
                all_label_index = torch.tensor(all_label_index)

                desp_tokens_ids.append(torch.tensor(desp_token))
                desp_seg_ids.append(torch.tensor(desp_seg))
                desp_mask_ids.append(torch.tensor(desp_mask))
                all_labels.append(torch.tensor(all_label_index))

        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        desp_tokens_ids = pad_sequence(desp_tokens_ids, batch_first=True)
        desp_seg_ids = pad_sequence(desp_seg_ids, batch_first=True)
        desp_mask_ids = pad_sequence(desp_mask_ids, batch_first=True)
        all_labels = pad_sequence(all_labels, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids,
                                    all_labels, y)
        print(len(dataset))
        return dataset

    def get_train_dev_loader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = DataLoader(
            self.train_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        dev_loader = DataLoader(
            self.dev_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        return train_loader, dev_loader

    def get_loader(self, type='train', batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            data = self.train_data
        elif type == 'dev':
            data = self.dev_data
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


class SNLIDataSet(Dataset):
    def __init__(self, args, split='train'):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.args = args
        self.split = split
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'dataset/sentence_pair'

        self.data_base_path = os.path.join(self.args.base_path, self.args.data_name, 'origin_file')

        if self.split == 'train':
            self.data_path = os.path.join(self.data_base_path, 'snli_1.0_train.txt')
        elif self.split == 'test':
            self.data_path = os.path.join(self.data_base_path, 'snli_1.0_test.txt')
        elif self.split == 'dev':
            self.data_path = os.path.join(self.data_base_path, 'snli_1.0_dev.txt')
        elif self.split == 'hard':
            self.data_path = os.path.join(self.data_base_path, 'snli_1.0_test_hard.txt')

        self.label_desp_path = os.path.join(self.data_base_path, 'snli_label_desp.txt')

        lines = open(self.data_path, 'r', encoding='utf8')
        self.total_length = sum(1 for i in lines) - 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pre_trained_model, cache_dir=self.args.cache_dir)

        self.origin_pddata = pd.read_csv(self.data_path, '\t')
        self.origin_pddata = self.origin_pddata[['gold_label', 'sentence1', 'sentence2']]

        self.load_desp()

    def load_desp(self):
        print('Loading description data....{}'.format(self.label_desp_path))

        content = []
        with open(self.label_desp_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label, desp = line.strip().split('\t')
                content.insert(self.label_dict[label], desp)

        self.label_content = content
        assert len(self.label_content) == self.args.num_classes

    def __getitem__(self, index):

        current_sample = self.origin_pddata.loc[index]
        sentence1 = current_sample['sentence1']
        sentence2 = current_sample['sentence2']

        label = self.label_dict[current_sample['gold_label']]

        try:
            values = self.process_sent(sentence1, sentence2)

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
                return values[0], values[1], desp_token, desp_mask, all_label_index, label
            else:
                return values[0], values[1], values[2], desp_token, desp_seg, desp_mask, all_label_index, label
        except TypeError:
            print(sentence1)
            print(sentence2)

    def __len__(self):
        return self.total_length

    def remove_lines(self, sample):
        label = sample['label']
        if label in self.label_dict.keys():
            return sample
        else:
            return []

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
            results = self.tokenizer.encode_plus([sent1, sent2])

        else:
            results = self.tokenizer.encode_plus(sent1)


        sentence_ids = torch.tensor(results['input_ids'])
        attention_mask_ids = torch.tensor(results['attention_mask'])
        if 'token_type_ids' in results.keys():
            segment_ids = torch.tensor(results['token_type_ids'])
            values = [sentence_ids, segment_ids, attention_mask_ids]
        else:
            values = [sentence_ids, attention_mask_ids]

        return values


class SNLIDataBertV2(Dataset):
    def __init__(self, args):
        self._args = args
        if self._args.debug:
            self._train_set = SNLIDataSet(self._args, split='dev')
        else:
            self._train_set = SNLIDataSet(self._args, split='train')
        self._dev_set = SNLIDataSet(self._args, split='dev')
        self._test_set = SNLIDataSet(self._args, split='test')
        self._hard_set = SNLIDataSet(self._args, split='hard')

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = self._get_loader(
            batch_size=batch_size,
            type='train',
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        dev_loader = self._get_loader(
            batch_size=batch_size,
            type='dev',
            shuffle=False,
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

        test_hard_loader = self._get_loader(
            batch_size=batch_size,
            type='hard',
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader,  dev_loader, test_loader, test_hard_loader

    def _get_loader(self, batch_size, type='train', shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            current_dataset = self._train_set
        elif type == 'dev':
            current_dataset = self._dev_set
        elif type == 'test':
            current_dataset = self._test_set
        elif type == 'hard':
            current_dataset = self._hard_set
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




