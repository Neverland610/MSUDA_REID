#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 \
python examples/multi_src_train_mmt_msuda.py -ds-1 dukemtmc -ds-2 cuhk03 -ds-3 msmt17 \
    -dt market1501 -a resnet50_rdsbn_mdif \
	--num-instances 4 --lr 0.00035 --iters 400 -b 32 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
	--init-1 logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50dsbn-pretrain-1/model_best.pth.tar \
	--init-2 logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50dsbn-pretrain-2/model_best.pth.tar \
	--logs-dir logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50_rdsbn_mdif-MMT-DBSCAN

