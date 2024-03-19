## TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning
This repository contains the code for the paper [TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning](https://arxiv.org/pdf/2312.15816.pdf).

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/task.png' width=600>
</p>

## Introduction
This is a follow-up work of [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX). We convert TKGs into a temporal event knowledge graph (TEKG) which equips us to develop a differentiable random walk approach. We also introduce conditional probability density functions, associated with the logical rules involving the query interval, using which we arrive at the time prediction. 

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/TEKG_example.png' width=450>
</p>


## How to run

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

First step

```sh
cd src
```

#### For (interval-based) datasets: wiki, YAGO

Random walk:
```sh
python main_random_walk_for_interval_datasets.py --dataset {$dataset name}
```

Rule learning:
```sh
python main_rule_learning_interval_dataset.py --dataset {$dataset name} --train
```

Rule application:
```sh
python main_rule_learning_interval_dataset.py --dataset {$dataset name} --test --from_model_ckpt {$your_model_location}
```

#### For (timestamp-based) datasets: icews14, icews05-15, gdelt100

Random walk:
```sh
python main_random_walk_for_timestamp_datasets.py --dataset {$dataset name}
```

Rule learning:
```sh
python main_rule_learning_timestamp_dataset.py --dataset {$dataset name} --train
```

Rule application:
```sh
python main_rule_application_timestamp_dataset.py --dataset {$dataset name}
python main_rule_learning_timestamp_dataset.py --dataset {$dataset name} --test --from_model_ckpt {$your_model_location}
```

## Citation
```
@misc{xiong2023teilp,
      title={TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning}, 
      author={Siheng Xiong and Yuan Yang and Ali Payani and James C Kerce and Faramarz Fekri},
      year={2023},
      eprint={2312.15816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
