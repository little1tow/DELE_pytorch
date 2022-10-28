"""
@Author: zhkun
@Time:  16:26
@File: my_parser
@Description: model parameters
@Something to attention
"""
import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="DELE")

    parser.add_argument('--net', type=str, default='led')
    parser.add_argument('--data_name', type=str, default='snli')
    parser.add_argument('--base_path', type=str, default='/dataset/sentence_pair')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--cache_dir', type=str, default='/data/pretrained_models')

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--cl_loss', type=str, default='default', help='default')

    parser.add_argument('--num_bert_layers', type=int, default=12)
    parser.add_argument('--in_features', type=int, default=768)
    parser.add_argument('--attention_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--mlp_size', type=int, default=200)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--cl_weight', type=float, default=0.5)
    parser.add_argument('--r2_weight', type=float, default=0.5)

    parser.add_argument('--display_step', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='add l2-norm to the added layers except Bert')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--test', action='store_true', default=False, help='Whether to just test the model')
    parser.add_argument('--pre_trained_model', type=str, default='bert-base-uncased',
                        help='bert-base-uncased, bert-large-uncased')

    parser.add_argument('--gpu', type=str, default='0', help='which gpus to use')
    parser.add_argument('--train_bert', action='store_true', default=False, help='Whether to fine-tune bert')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--grad_max_norm', type=float, default=0.)  #
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_pred_layer', type=int, default=2)

    parser.add_argument('--desp_type', type=str, default='default',
                        help='how to process the bert output of description, [default(cls), atten, mean]')
    parser.add_argument('--label_fuse_type', type=str, default='default',
                        help='how to process the two kinds of label embedding, [default(add), concat]')
    parser.add_argument('--head', type=str, default='default',
                        help='how to project the representation, [default, res]')
    parser.add_argument('--sentence_pooling', type=str, default='default',
                        help='how to process the label guided sentence embedding, [default(mean), sum, max]')
    parser.add_argument('--sentence_rep', type=str, default='sentence',
                        help='how to predict the final results. [sentence, label, concat]')
    parser.add_argument('--mi_type', type=str, default='mi',
                        help='how to process the bert output of description, [mi, self, max, mean]')
    parser.add_argument('--r2_loss', type=str, default='single',
                        help='how to process calculate the r2 task loss, [double, single]')

    parser.add_argument('--label_weight', action='store_true', default=False,
                        help='whether use label weight for free embedding and description. [true, false]')
    parser.add_argument('--use_sentence_weight', action='store_true', default=False,
                        help='whether select all layers results. [true, false]')
    parser.add_argument('--use_desp', action='store_true', default=False,
                        help='whether use label descriptions. [true, false]')
    parser.add_argument('--use_output_pooling', action='store_true', default=False,
                        help='whether use the pooling results from bert, [true, false]')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether debug the whole model, [true, false]')
    parser.add_argument('--use_content', action='store_true', default=False,
                        help='whether use content for other dataset. [true, false]')

    parser.add_argument('--only_sentence', action='store_true', default=False,
                        help='whether only use the sentence input as the prediction input. [true, false]')

    parser.add_argument('--desp_seperate', action='store_true', default=False,
                        help='whether use an untrained bert to process the description. [true, false]')

    parser.add_argument('--use_f1', action='store_true', default=False, help='Whether to utilize f1 to test model')

    args = parser.parse_args()
    # if args.debug:
    pprint.PrettyPrinter().pprint(args.__dict__)
    return args
