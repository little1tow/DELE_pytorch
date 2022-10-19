"""
@Author: zhkun
@Time:  18:04
@File: main
@Description: main entrance
@Something to attention
"""
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

import os
from my_parser import parser

from solverV2 import Solver as Solverv2, SolverDouble


def main():
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if args.net in ['led', 'ledv2']:
        solver = Solverv2(args)
    elif args.net == 'ledr2':
        solver = SolverDouble(args)
    else:
        raise ValueError('the key word is not exist')
    if not args.test:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    main()
