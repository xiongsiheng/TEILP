## TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning
This repository contains the code for the paper [TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning](https://arxiv.org/pdf/2312.15816.pdf).

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/task.png' width=600>
</p>

## Introduction
This is a follow-up work of [TILP: Differentiable Learning of Temporal Logical Rules on Knowledge Graphs](https://openreview.net/pdf?id=_X12NmQKvX). We convert TKGs into a temporal event knowledge graph (TEKG) which equips us to develop a differentiable random walk approach. We also introduce conditional probability density functions, associated with the logical rules involving the query interval, using which we arrive at the time prediction. 

<p align="center">
  <img src='https://github.com/xiongsiheng/TEILP/blob/main/misc/TEKG_example.png' width=400>
</p>


## Commands

#### (Interval-based) Datasets: wiki, YAGO

Random walk:
```sh
python src/main_random_walk_for_interval_datasets.py --dataset YAGO

python src/main_random_walk_for_interval_datasets.py --dataset wiki
```
Rule learning:
```sh
python src/main_rule_learning_interval_dataset.py --dataset YAGO --train

python src/main_rule_learning_interval_dataset.py --dataset wiki --train
```
Rule application:
```sh
python src/main_rule_learning_interval_dataset.py --dataset YAGO --test --from_model_ckpt {$your_model_location}

python src/main_rule_learning_interval_dataset.py --dataset wiki --test --from_model_ckpt {$your_model_location}
```

#### (Timestamp-based) Datasets: icews14, icews05-15, gdelt100

Random walk:
```sh
python src/main_random_walk_for_timestamp_datasets.py --dataset icews14

python src/main_random_walk_for_timestamp_datasets.py --dataset icews05-15

python src/main_random_walk_for_timestamp_datasets.py --dataset gdelt100
```
Rule learning:
```sh
python src/main_rule_learning_timestamp_dataset.py --dataset icews14 --train

python src/main_rule_learning_timestamp_dataset.py --dataset icews05-15 --train

python src/main_rule_learning_timestamp_dataset.py --dataset gdelt100 --train
```
Rule application:
```sh
python src/main_rule_application_timestamp_dataset.py --dataset icews14

python src/main_rule_application_timestamp_dataset.py --dataset icews05-15

python src/main_rule_application_timestamp_dataset.py --dataset gdelt100

python src/main_rule_learning_timestamp_dataset.py --dataset icews14 --test --from_model_ckpt {$your_model_location}

python src/main_rule_learning_timestamp_dataset.py --dataset icews05-15 --test --from_model_ckpt {$your_model_location}

python src/main_rule_learning_timestamp_dataset.py --dataset gdelt100 --test --from_model_ckpt {$your_model_location}
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
