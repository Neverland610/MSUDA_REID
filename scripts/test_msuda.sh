#!/bin/sh

CUDA_VISIBLE_DEVICES=0 \
python examples/test_msuda.py -b 512 -j 8 \
    -ds-1 dukemtmc -ds-2 cuhk03 -ds-3 msmt17 -dt market1501 \
    -a resnet50_rdsbn_mdif \
    --resume logs/dukemtmc_cuhk03_msmt17TOmarket1501/resnet50_rdsbn_mdif-MMT-DBSCAN/model_best.pth.tar

