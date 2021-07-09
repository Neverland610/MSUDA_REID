# Unsupervised Multi-Source Domain Adaptation for Person Re-Identification

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.1](https://img.shields.io/badge/pytorch-1.3-yellow.svg)

Unofficial implementation of CVPR 2021 Oral paper [Unsupervised Multi-Source Domain Adaptation for Person Re-Identification](https://arxiv.org/abs/2104.12961)

This repo provides the code to implement 3-source (duke+cuhk03+msmt -> market) domain adaptive person re-ID task. It runs on Python 3.6 with Pytorch 1.3.1. For other dependencies, see `setup.py`.

### Installation

```shell
# clone this repo
cd MSUDA_REID
python setup.py install
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Prepare DukeMTMC-reID, Market-1501, CUHK03 and MSMT17 datasets as in [MMT](https://github.com/yxgeee/MMT/tree/master/mmt).

### Train

Two 32GB V100 GPUs are used to train the 3-source adaptation. You can also reduce the batch size to fit your GPU memory.

#### Stage I: Pre-training on the source domains

```shell
bash scripts/multi_src_pretrain.sh 1
bash scripts/multi_src_pretrain.sh 2
```

#### Stage II: End-to-end training with MSUDA (RDSBN-MDIF) 
```shell
bash scripts/multi_src_train_mmt_msuda.sh
```

### Test

Test the trained model with best performance by

```shell
bash scripts/test_msuda.sh
```

### Result

| Method   | mAP  | R-1  | R-5  | R-10 |
| -------- | ---- | ---- | ---- | ---- |
| MMT+RDSBN-MDIF | 85.9 | 94.3 | 97.6 | 98.8 |

### Pre-trained models
The best performance model can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1WZ-pKS59kOQgq2FZJyc3QA)  password: n0cm.

Place the downloaded model in `logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50_rcdsbn_mdif-MMT-DBSCAN/`

### Ack

This repo borrows a lot of code from [MMT](https://github.com/yxgeee/MMT/tree/master/mmt), thanks!
