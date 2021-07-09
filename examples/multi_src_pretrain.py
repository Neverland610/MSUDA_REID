from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from mmt import datasets
from mmt import models
from mmt.trainers import PreTrainerDSBN
from mmt.evaluators import Evaluator
from mmt.utils.data import IterLoader
from mmt.utils.data import transforms as T
from mmt.utils.data.sampler import RandomMultipleGallerySampler
from mmt.utils.data.preprocessor import Preprocessor
from mmt.utils.logging import Logger
from mmt.utils.serialization import save_checkpoint
from mmt.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0


def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer
         ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                           batch_size=batch_size, num_workers=workers, sampler=sampler,
                           shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source1, num_classes1, train_loader_source1, test_loader_source1 = \
        get_data(args.dataset_source1, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)

    dataset_source2, num_classes2, train_loader_source2, test_loader_source2 = \
        get_data(args.dataset_source2, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)

    dataset_source3, num_classes3, train_loader_source3, test_loader_source3 = \
        get_data(args.dataset_source3, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)

    dataset_target, _, train_loader_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, iters)

    num_classes = num_classes1 + num_classes2 + num_classes3

    # Create model
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout,
                          num_classes=num_classes,
                          num_domains=4, pretrained=True, within_single_batch=False)
    model.cuda()
    model = nn.DataParallel(model)

    evaluator = Evaluator(model)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        else:
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    trainer = PreTrainerDSBN(model, num_classes1, num_classes2, num_classes3, margin=args.margin)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step()
        train_loader_source1.new_epoch()
        train_loader_source2.new_epoch()
        train_loader_source3.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch,
                      train_loader_source1, train_loader_source2, train_loader_source3,
                      train_loader_target, optimizer,
                      train_iters=len(train_loader_source1), print_freq=args.print_freq)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

            _, mAP = evaluator.evaluate(test_loader_source3, dataset_source3.query, dataset_source3.gallery,
                                        cmc_flag=True, domain_label=3)
            print('Epoch {:3d}, {} mAP: {:5.1%}'.format(epoch, args.dataset_source3, mAP))

            _, mAP = evaluator.evaluate(test_loader_source2, dataset_source2.query, dataset_source2.gallery,
                                        cmc_flag=True, domain_label=2)
            print('Epoch {:3d}, {} mAP: {:5.1%}'.format(epoch, args.dataset_source2, mAP))

            _, mAP = evaluator.evaluate(test_loader_source1, dataset_source1.query, dataset_source1.gallery,
                                        cmc_flag=True, domain_label=1)
            print('Epoch {:3d}, {} mAP: {:5.1%}'.format(epoch, args.dataset_source1, mAP))

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print("Test on target domain with 0-bn:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                       cmc_flag=True, rerank=args.rerank, domain_label=0)
    print("Test on target domain with 1-bn duke:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                       cmc_flag=True, rerank=args.rerank, domain_label=1)
    print("Test on target domain with 2-bn cuhk:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                       cmc_flag=True, rerank=args.rerank, domain_label=2)
    print("Test on target domain with 3-bn msmt:")
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                       cmc_flag=True, rerank=args.rerank, domain_label=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds-1', '--dataset-source1', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-ds-2', '--dataset-source2', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-ds-3', '--dataset-source3', type=str, default='msmt',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
