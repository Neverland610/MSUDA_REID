#!/bin/sh
SEED=$1

if [ $# -ne 1 ]
  then
    echo "Arguments error: <SEED>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 \
python examples/multi_src_pretrain.py -ds-1 dukemtmc -ds-2 cuhk03 -ds-3 msmt17 -dt market1501 \
    -a resnet50dsbn --seed ${SEED} --margin 0.0 \
	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 40 \
	--logs-dir logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50dsbn-pretrain-${SEED}

