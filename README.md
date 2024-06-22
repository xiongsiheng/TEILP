## [AAAI 24 (oral)] TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning
This repository contains the code for the paper [TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning](https://arxiv.org/pdf/2312.15816.pdf).

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/task.png' width=600>
</p>

## Introduction
This is a follow-up work of [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX). We convert TKGs into a temporal event knowledge graph (TEKG) which equips us to develop a differentiable random walk approach. We also introduce conditional probability density functions, associated with the logical rules involving the query interval, using which we arrive at the time prediction. 

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/TEKG_example.png' width=450>
</p>


## Quick Start

The structure of the file folder should be like

```sh
TEILP/
│
├── src/
│
├── data/
│
├── exps/
│
└── output/

```

Get into the dir

```sh
cd src
```

Random walk:
```sh
# For wiki, YAGO:
python random_walk_for_interval_datasets.py --dataset {$dataset name}

# For icews14, icews05-15, gdelt100:
python random_walk_for_timestamp_datasets.py --dataset {$dataset name}

# Examples:
# python random_walk_for_interval_datasets.py --dataset wiki
# python random_walk_for_timestamp_datasets.py --dataset icews14
```

Rule learning:
```sh
python main.py --dataset {$dataset name} --train

# Examples:
# python main.py --dataset YAGO --train
# python main.py --dataset icews14 --train
```

Rule application:
```sh
python main.py --dataset {$dataset name} --test --from_model_ckpt {$path to saved model}

# Examples:
# python main.py --dataset YAGO --test --from_model_ckpt ../exps/YAGO_24-02-17-20-57/ckpt/model-30
# python main.py --dataset icews14 --test --from_model_ckpt ../exps/icews14_24-02-18-11-03/ckpt/model-30
```


## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
@inproceedings{xiong2024teilp,
  title={Teilp: Time prediction over knowledge graphs via logical reasoning},
  author={Xiong, Siheng and Yang, Yuan and Payani, Ali and Kerce, James C and Fekri, Faramarz},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={16112--16119},
  year={2024}
}
```
