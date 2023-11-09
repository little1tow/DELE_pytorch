# DELE_pytorch
This is the official implementation of the TNNLS2022 work "Description-Enhanced Label Embedding Contrastive Learning for Text Classification"

## Requirement
    1. pytorch >= 1.7.1
    2. photinia
    3. Transformer

## Data Preparation
    1. SNLI [https://nlp.stanford.edu/projects/snli/]
    2. SICK [https://huggingface.co/datasets/sick]
    3. SciTail [https://allenai.org/data/scitail]
    4. Quora Question Pair [https://huggingface.co/datasets/quora]
    5. MSRP [https://www.microsoft.com/en-us/download/details.aspx?id=52398]
    6. Yahoo Answer [https://huggingface.co/datasets/yahoo_answers_topics]
    7. SST [https://nlp.stanford.edu/sentiment/treebank.html]

Note that most of the datasets can be obtained from the [huggingface dataset](https://huggingface.co/datasets)

# Bibtex
If you use this dataset or code in your work, please cite [our AAAI2021 paper](https://www.aaai.org/AAAI21Papers/AAAI-6199.ZhangK.pdf) and [our_TNNLS2022_paper](https://ieeexplore.ieee.org/document/10154259):
```
@inproceedings{zhang2021making,
  title={Making the relation matters: Relation of relation learning network for sentence semantic matching},
  author={Zhang, Kun and Wu, Le and Lv, Guangyi and Wang, Meng and Chen, Enhong and Ruan, Shulan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={16},
  pages={14411--14419},
  year={2021}
}

@article{zhang2022descrip,
  author={Zhang, Kun and Wu, Le and Lv, Guangyi and Chen, Enhong and Ruan, Shulan and Liu, Jing and Zhang, Zhiqiang and Zhou, Jun and Wang, Meng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Description-Enhanced Label Embedding Contrastive Learning for Text Classification}, 
  year={2023},
  pages={1-14},
  doi={10.1109/TNNLS.2023.3282020}
}
