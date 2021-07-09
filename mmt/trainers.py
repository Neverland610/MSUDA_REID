from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainerDSBN(object):
    def __init__(self, model,
                 num_classes1, num_classes2, num_classes3,
                 margin=0.0):
        super(PreTrainerDSBN, self).__init__()
        self.model = model

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3

        self.criterion_ce1 = CrossEntropyLabelSmooth(num_classes1).cuda()
        self.criterion_ce2 = CrossEntropyLabelSmooth(num_classes2).cuda()
        self.criterion_ce3 = CrossEntropyLabelSmooth(num_classes3).cuda()

        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch,
              data_loader_source1, data_loader_source2, data_loader_source3,
              data_loader_target,
              optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            ########################################################################################################
            #                                        process first source                                          #
            ########################################################################################################
            source_inputs = data_loader_source1.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs, domain_label=1)
            t_inputs, _ = self._parse_data(target_inputs, domain_label=0)

            s_features, s_cls_out = self.model(s_inputs[0], s_inputs[1])
            s_cls_out = s_cls_out[:, :self.num_classes1]
            t_features, _ = self.model(t_inputs[0], t_inputs[1])

            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets, self.criterion_ce1)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########################################################################################################
            #                                        process second source                                          #
            ########################################################################################################

            source_inputs = data_loader_source2.next()
            s_inputs, targets = self._parse_data(source_inputs, domain_label=2)

            s_features, s_cls_out = self.model(s_inputs[0], s_inputs[1])
            s_cls_out = s_cls_out[:, self.num_classes1:self.num_classes1+self.num_classes2]

            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets, self.criterion_ce2)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########################################################################################################
            #                                        process third source                                          #
            ########################################################################################################

            source_inputs = data_loader_source3.next()
            s_inputs, targets = self._parse_data(source_inputs, domain_label=3)

            s_features, s_cls_out = self.model(s_inputs[0], s_inputs[1])
            s_cls_out = s_cls_out[:, self.num_classes1 + self.num_classes2:
                                     self.num_classes1 + self.num_classes2 + self.num_classes3]

            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets, self.criterion_ce3)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs, domain_label):
        imgs, _, pids, _ = inputs
        inputs = [imgs.cuda(), domain_label * torch.ones(imgs.shape[0], dtype=torch.long).cuda()]
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets, criterion_ce):
        loss_ce = criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class MMTTrainerRDSBN_MDIF(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema,
                 num_cluster=500, num_classes1=1000, num_classes2=1000, num_classes3=1000,
                 alpha=0.999):
        super(MMTTrainerRDSBN_MDIF, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster + num_classes1 + num_classes2 + num_classes3).cuda()

        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
              data_loader_source1, data_loader_source2, data_loader_source3,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter()]

        end = time.time()
        for i in range(train_iters):

            # load data
            source_inputs0 = data_loader_target.next()      # 0 is target domain
            source_inputs1 = data_loader_source1.next()
            source_inputs2 = data_loader_source2.next()
            source_inputs3 = data_loader_source3.next()
            data_time.update(time.time() - end)
            stride = 16

            # process inputs for target pseudo label and domain label
            inputs_0_1, inputs_0_2, targets_0 = self._parse_data(source_inputs0, domain_label=0)
            inputs_1_1, inputs_1_2, targets_1 = self._parse_data(source_inputs1, domain_label=1)
            inputs_2_1, inputs_2_2, targets_2 = self._parse_data(source_inputs2, domain_label=2)
            inputs_3_1, inputs_3_2, targets_3 = self._parse_data(source_inputs3, domain_label=3)

            # re-organize the batch to be in domain order
            real_input_1 = [inputs_0_1[0][:stride], inputs_1_1[0][:stride], inputs_2_1[0][:stride], inputs_3_1[0][:stride],
                            inputs_0_1[0][stride:], inputs_1_1[0][stride:], inputs_2_1[0][stride:], inputs_3_1[0][stride:]]
            real_domain_1 = [inputs_0_1[1][:stride], inputs_1_1[1][:stride], inputs_2_1[1][:stride], inputs_3_1[1][:stride],
                             inputs_0_1[1][stride:], inputs_1_1[1][stride:], inputs_2_1[1][stride:], inputs_3_1[1][stride:]]

            real_input_2 = [inputs_0_2[0][:stride], inputs_1_2[0][:stride], inputs_2_2[0][:stride], inputs_3_2[0][:stride],
                            inputs_0_2[0][stride:], inputs_1_2[0][stride:], inputs_2_2[0][stride:], inputs_3_2[0][stride:]]
            real_domain_2 = [inputs_0_2[1][:stride], inputs_1_2[1][:stride], inputs_2_2[1][:stride], inputs_3_2[1][:stride],
                             inputs_0_2[1][stride:], inputs_1_2[1][stride:], inputs_2_2[1][stride:], inputs_3_2[1][stride:]]

            real_target = [targets_0[:stride], targets_1[:stride]+self.num_cluster,
                           targets_2[:stride]+self.num_cluster+self.num_classes1,
                           targets_3[:stride] + self.num_cluster + self.num_classes1 + self.num_classes2,
                           targets_0[stride:], targets_1[stride:]+self.num_cluster,
                           targets_2[stride:]+self.num_cluster+self.num_classes1,
                           targets_3[stride:] + self.num_cluster + self.num_classes1 + self.num_classes2]
            real_input_1 = torch.cat(real_input_1, dim=0)
            real_domain_1 = torch.cat(real_domain_1, dim=0)
            real_input_2 = torch.cat(real_input_2, dim=0)
            real_domain_2 = torch.cat(real_domain_2, dim=0)
            real_target = torch.cat(real_target, dim=0)

            # forward
            f_out_t1, p_out_t1, p_out_gcn1 = self.model_1(real_input_1, real_domain_1)
            f_out_t2, p_out_t2, p_out_gcn2 = self.model_2(real_input_2, real_domain_2)
            p_out_t1 = p_out_t1[:, :self.num_cluster+self.num_classes1+self.num_classes2+self.num_classes3]
            p_out_t2 = p_out_t2[:, :self.num_cluster+self.num_classes1+self.num_classes2+self.num_classes3]
            p_out_gcn1 = p_out_gcn1[:, :self.num_cluster + self.num_classes1 + self.num_classes2+self.num_classes3]
            p_out_gcn2 = p_out_gcn2[:, :self.num_cluster + self.num_classes1 + self.num_classes2+self.num_classes3]

            f_out_t1_ema, p_out_t1_ema, p_out_gcn1_ema = self.model_1_ema(real_input_1, real_domain_1)
            f_out_t2_ema, p_out_t2_ema, p_out_gcn2_ema = self.model_2_ema(real_input_2, real_domain_2)
            p_out_t1_ema = p_out_t1_ema[:, :self.num_cluster+self.num_classes1+self.num_classes2+self.num_classes3]
            p_out_t2_ema = p_out_t2_ema[:, :self.num_cluster+self.num_classes1+self.num_classes2+self.num_classes3]
            p_out_gcn1_ema = p_out_gcn1_ema[:, :self.num_cluster + self.num_classes1 + self.num_classes2+self.num_classes3]
            p_out_gcn2_ema = p_out_gcn2_ema[:, :self.num_cluster + self.num_classes1 + self.num_classes2+self.num_classes3]

            loss_ce_1 = self.criterion_ce(p_out_t1, real_target)
            loss_ce_2 = self.criterion_ce(p_out_t2, real_target)
            loss_ce_gcn_1 = self.criterion_ce(p_out_gcn1, real_target)
            loss_ce_gcn_2 = self.criterion_ce(p_out_gcn2, real_target)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, real_target)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, real_target)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            loss_ce_soft_gcn = self.criterion_ce_soft(p_out_gcn1, p_out_gcn2_ema) + \
                               self.criterion_ce_soft(p_out_gcn2, p_out_gcn1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, real_target) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, real_target)

            loss = (loss_ce_1 + loss_ce_2 + loss_ce_gcn_1 + loss_ce_gcn_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                   (loss_ce_soft + loss_ce_soft_gcn)*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, real_target.data)
            prec_2, = accuracy(p_out_t2.data, real_target.data)
            prec_3, = accuracy(p_out_gcn1.data, real_target.data)
            prec_4, = accuracy(p_out_gcn2.data, real_target.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])
            precisions[2].update(prec_3[0])
            precisions[3].update(prec_4[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      'Prec-GCN {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg,
                              precisions[2].avg, precisions[3].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs, domain_label=0):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = [imgs_1.cuda(), domain_label * torch.ones(imgs_1.shape[0], dtype=torch.long).cuda()]
        inputs_2 = [imgs_2.cuda(), domain_label * torch.ones(imgs_2.shape[0], dtype=torch.long).cuda()]
        targets = pids.cuda()
        return inputs_1, inputs_2, targets

