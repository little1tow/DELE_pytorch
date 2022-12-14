"""
@Author: zhkun
@Time:  16:22
@File: baselines
@Description: baseline file
@Something to attention
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle as pkl

# from model_log.modellog import ModelLog

from transformers import AdamW
import time
import datetime

import photinia as ph

from utils import get_current_time
from model import Base_model
from dataset import SickDataBert, SNLIDataBertV2
from other_dataset import MSRPDataBert, SentimentDataBert
from utils import BiClassCalculator, write_file


class Baseline:
    def __init__(self, args):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = max([4 * torch.cuda.device_count(), 4])

        train_loader = None
        dev_loader = None
        test_loader = None
        test_hard_loader = None

        torch.manual_seed(args.seed)
        # prepar_data()
        if not args.data_name == 'snli':
            if args.data_name in ['sick', 'scitail', 'quora']:
                dataset = SickDataBert(args)
            elif args.data_name.lower() == 'msrp':
                dataset = MSRPDataBert(args)
            else:
                dataset = SentimentDataBert(args)

            if args.data_name.lower() in ['sick', 'scitail', 'quora', 'msrp']:
                train_loader = dataset.get_loader(
                    type='train',
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device == 'cuda'
                )
                dev_loader = dataset.get_loader(
                    type='dev',
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device == 'cuda'
                )
                test_loader = dataset.get_loader(
                    type='test',
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device == 'cuda'
                )
                print('#examples:',
                      '#train', len(train_loader.dataset),
                      '#dev', len(dev_loader.dataset),
                      '#test', len(test_loader.dataset),
                      )
            else:
                train_loader = dataset.get_loader(
                    type='train',
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device == 'cuda'
                )
                test_loader = dataset.get_loader(
                    type='test',
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device == 'cuda'
                )
                print('#examples:',
                      '#train', len(train_loader.dataset),
                      '#test', len(test_loader.dataset))
        else:
            dataset = SNLIDataBertV2(args)
            train_loader, dev_loader, test_loader, test_hard_loader = dataset.get_dataloaders(
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                # num_workers=args.num_workers,
                pin_memory=device == 'cuda'
            )
            # prepare data
            print('#examples:',
                  '\n#train ', len(train_loader.dataset),
                  '\n#dev', len(dev_loader.dataset),
                  '\n#test', len(test_loader.dataset),
                  '\n#test_hard', len(test_hard_loader.dataset)
                  )

        model = Base_model(args)

        device_count = 0
        if device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        model.to(device)

        # Other optimizer
        params = model.module.req_grad_params if device_count > 1 else model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), amsgrad=True, weight_decay=args.weight_decay)

        # Bert optimizer
        param_optimizer = list(model.module.bert.named_parameters() if device_count > 1 else model.bert.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer_bert = AdamW(optimizer_grouped_parameters, lr=1e-5)

        classify_loss = nn.CrossEntropyLoss()

        # save name and path
        args.name += '_bert' if args.train_bert else ''
        ckpt_path = os.path.join('checkpoint', '{}'.format(args.name))
        if not os.path.exists(ckpt_path+'.pth'):
            print('Not found ckpt', ckpt_path)

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.optimizer_bert = optimizer_bert
        self.classify_loss = classify_loss
        self.device = device
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.train_loader = train_loader

        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.test_hard_loader = test_hard_loader

        # self.batch_idx = 0
        self.training_log = []
        self.testing_log = []

    def train(self):
        print('Starting Traing....')
        best_loss = float('inf')
        best_acc = 0.

        self.big_epochs = len(self.train_loader.dataset) // self.args.batch_size

        self.learning_scheduler = ph.optim.lr_scheduler.CosineWarmUpDecay(
            optimizer=self.optimizer,
            num_loops=self.args.epochs * self.big_epochs,
            min_factor=1e-8,
        )

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            print('-' * 20 + 'Epoch: {}, {}'.format(epoch, get_current_time()) + '-' * 20)
            if epoch != 1:
                write_file(os.path.join(self.args.log_path, f'testing_log_{self.args.name}.log'), self.testing_log)
                self.testing_log.clear()

            train_loss, train_acc = self.train_epoch(epoch)
            dev_loss, dev_acc, metrics, dev_correct_count = self.evaluate_epoch('Dev')
            if self.args.use_f1:
                if metrics[-1] > best_acc:
                    best_loss = dev_loss
                    best_acc = metrics[-1]
                    self.save_model('dev')

            else:
                if dev_acc > best_acc:
                    best_loss = dev_loss
                    best_acc = dev_acc
                    self.save_model('dev')

            test_log = f'------------------{datetime.datetime.now()}----------------------------\t' \
                       f'Epoch:{epoch}\t' \
                       f'{self.args.name} \t' \
                       f'Train loss:{train_loss:.5f}, Train acc:{train_acc:.5f}\t' \
                       f'Dev Loss:{dev_loss:.5f}, Dev Acc:{dev_acc:.5f}\t' \
                       f'Dev count:{dev_correct_count}, Total dev count:{len(self.dev_loader.dataset)}\t' \
                       f'Best Dev Loss:{best_loss:.5f}, Best Dev Acc:{best_acc:.5f}, \t'

            self.testing_log.append(test_log)

            print(test_log.replace('\t', '\n'))

            write_file(os.path.join(self.args.log_path, f'training_log_{self.args.name}.log'), self.training_log)
            self.training_log.clear()

        print('Training Finished!')

        self.save_model(name='last')

        self.test()

    def test(self):
        # Load the best checkpoint
        test_logs = []
        for name in ['dev', 'last']:
            self.load_model(name)
            if self.args.data_name == 'snli':
                print('*'*25 + f'Final dev result at {name}' + '*'*25)
                # print(f'Final dev result at {name}..............')
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('Dev')
                log = 'Dev Loss: {:.3f}, \t' \
                    'Dev Acc: {:.3f}, \t'\
                    'Dev correct_count: {}, \t'\
                    'Dev Acc_cal: {:.5f}, \t'\
                    'Dev precision: {:.5f}, \t'\
                    'Dev recall: {:.5f},\t'\
                    'Dev f1: {:.5f}'.format(test_loss, test_acc, test_correct_count, metrics[0], metrics[1], metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)

                # Test
                print('*'*25 + f'Final test result at {name}' + '*'*25)
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('Test')
                # print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                log = 'Test Loss: {:.3f}, \t' \
                    'Test Acc: {:.3f}, \t'\
                    'Test correct_count: {},\t'\
                    'Test Acc_cal: {:.5f}, \t'\
                    'Test precision: {:.5f}, \t'\
                    'Test recall: {:.5f}, \t'\
                    'Test f1: {:.5f}'.format(test_loss, test_acc, test_correct_count, metrics[0], metrics[1], metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)

                print('*'*25 + f'Final test hard result at {name}' + '*'*25)
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('hard')
                # print('Test hard Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                log = 'Test hard Loss: {:.3f}, \t'\
                    'Test hard Acc: {:.3f}, \t'\
                    'Test hard correct_count: {}, \t'\
                    'Test hard Acc_cal: {:.5f}, \t'\
                    'Test hard precision: {:.5f}, \t'\
                    'Test hard recall: {:.5f}, \t'\
                    'Test hard f1: {:.5f}'.format(
                        test_loss, test_acc, test_correct_count, metrics[0], metrics[1], metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)

            elif self.args.data_name in ['trecQA', 'wikiQA', 'sick', 'scitail', 'quora', 'msrp']:
                print('*'*25 + f'Final dev result at {name}' + '*'*25)
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('Dev')
                # print('Dev Loss: {:.3f}, Dev Acc: {:.3f}'.format(test_loss, test_acc))
                log = 'Dev Loss: {:.3f}, \t' \
                      'Dev Acc: {:.3f}, \t' \
                      'Dev correct_count: {}, \t' \
                      'Dev Acc_cal: {:.5f}, \t' \
                      'Dev precision: {:.5f}, \t' \
                      'Dev recall: {:.5f},\t' \
                      'Dev f1: {:.5f}'.format(test_loss, test_acc, test_correct_count, metrics[0], metrics[1],
                                              metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)
                # Test
                print('*'*25 + f'Final test result at {name}' + '*'*25)
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('Test')
                # print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                log = 'Test Loss: {:.3f}, \t' \
                      'Test Acc: {:.3f}, \t' \
                      'Test correct_count: {},\t' \
                      'Test Acc_cal: {:.5f}, \t' \
                      'Test precision: {:.5f}, \t' \
                      'Test recall: {:.5f}, \t' \
                      'Test f1: {:.5f}'.format(test_loss, test_acc, test_correct_count, metrics[0], metrics[1],
                                               metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)

            else:
                # Test
                print('*'*25 + f'Final test result at {name}' + '*'*25)
                test_loss, test_acc, metrics, test_correct_count = self.evaluate_epoch('Test')
                # print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                log = 'Test Loss: {:.3f}, \t' \
                      'Test Acc: {:.3f}, \t' \
                      'Test correct_count: {},\t' \
                      'Test Acc_cal: {:.5f}, \t' \
                      'Test precision: {:.5f}, \t' \
                      'Test recall: {:.5f}, \t' \
                      'Test f1: {:.5f}'.format(test_loss, test_acc, test_correct_count, metrics[0], metrics[1],
                                               metrics[2], metrics[3])
                print(log.replace('\t', '\n'))
                test_logs.append(log)

        write_file(os.path.join(self.args.log_path, f'testing_log_{self.args.name}.log'), test_logs)

    def train_epoch(self, epoch_idx):
        self.model.train()
        train_loss = 0.
        example_count = 0
        correct = 0

        for batch_idx, (token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids,
                                    all_labels, y) in enumerate(self.train_loader):
            pairs_info = [token_ids, seg_ids, mask_ids]

            pairs_info = [pair.to(self.device) for pair in pairs_info]

            target = y.to(self.device)

            predict, _ = self.model(
                sentence_pairs=pairs_info,
                is_train=True
                )

            self.optimizer.zero_grad()
            self.optimizer_bert.zero_grad()

            loss = self.classify_loss(predict, target)

            loss.backward()

            if self.args.grad_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.req_grad_params, self.args.grad_max_norm)
            self.optimizer.step()
            self.optimizer_bert.step()
            self.learning_scheduler.step()

            pred_c = torch.max(predict, 1)[1]
            correct_pred = pred_c.eq(target.view_as(pred_c)).sum().item()
            correct += correct_pred

            example_count += len(predict)
            train_loss += loss.item()

            current_learning_rate = self.learning_scheduler.get_last_lr()[0]

            screen_log = f'{datetime.datetime.now()}\tBatch:{epoch_idx}--{batch_idx + 1}\t' \
                         f'{self.args.name}\t' \
                         f'loss:{loss:.4f}\t' \
                         f'lr:{current_learning_rate:.6f}\t' \
                         f'Batch acc: {correct_pred} / {len(predict)} = {correct_pred / (len(predict) * 1.0):.4f}'
            self.training_log.append(screen_log.replace('\t', ', '))

            if batch_idx == 0 or (batch_idx + 1) % self.args.display_step == 0:
                print(screen_log.replace('\t', ', '))

            # exit(0)

        train_loss /= (example_count * 1.0)
        acc = correct / (example_count * 1.0)
        return train_loss, acc

    def evaluate_epoch(self, mode):
        print(f'Evaluating {mode}....')
        self.model.eval()
        if self.args.data_name == 'snli':
            if mode == 'Dev':
                loader = self.dev_loader
            elif mode == 'hard':
                loader = self.test_hard_loader
            else:
                loader = self.test_loader
        elif self.args.data_name.lower() in ['trecQA', 'wikiQA', 'sick', 'scitail', 'quora', 'msrp']:
            if mode == 'Dev':
                loader = self.dev_loader
            else:
                loader = self.test_loader
        else:
            loader = self.test_loader
        eval_loss = 0.
        correct = 0
        representation = []
        matrix_cal = BiClassCalculator()
        with torch.no_grad():
            for batch_idx, (token_ids, seg_ids, mask_ids, desp_tokens_ids, desp_seg_ids, desp_mask_ids,
                            all_labels, y) in enumerate(loader):
                pairs_info = [token_ids, seg_ids, mask_ids]

                pairs_info = [pair.to(self.device) for pair in pairs_info]

                target = y.to(self.device)
                predict, rep = self.model(
                    sentence_pairs=pairs_info,
                    is_train=False
                )

                loss = self.classify_loss(predict, target)
                eval_loss += loss.item()
                pred = torch.max(predict, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                matrix_cal.update(pred.cpu().numpy(), target.cpu().numpy())

                for idx, item in enumerate(zip(rep, pred, y)):
                    representation.append([item[0].cpu().numpy(), item[1].cpu().item(), item[2].cpu().item()])

            if self.args.test:
                with open(os.path.join(self.args.log_path, f'{self.args.name}_rep.pkl'), 'wb') as f:
                    pkl.dump(representation, f)

        eval_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)

        return eval_loss, acc, [matrix_cal.accuracy, matrix_cal.precision, matrix_cal.recall, matrix_cal.f1], correct

    def save_model(self, name):
        model_dict = dict()
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self.args


        model_dict['optimizer'] = self.optimizer.state_dict()
        if name is None:
            ckpt_path = self.ckpt_path + '.pth'
        else:
            ckpt_path = self.ckpt_path + name + '.pth'
        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(model_dict, ckpt_path)
        print('Saved', ckpt_path)

    def load_model(self, name='dev'):
        ckpt_path = self.ckpt_path + name + '.pth'
        print('Load checkpoint', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # if saving a paralleled model but loading an unparalleled model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'])

        print(f'> best model at {ckpt_path} is loaded!')