import torch
from torch import nn


class DomainSpecificBatchNorm2d(nn.Module):

    def __init__(self, num_channel, num_domains, eps=1e-9, momentum=0.1, affine=True,
                 track_running_stats=True, within_single_batch=True):
        super(DomainSpecificBatchNorm2d, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_channel, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])
        self.single_batch = within_single_batch

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def bias_requires_grad(self, flag):
        for bn in self.bns:
            bn.bias.requires_grad_(flag)

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        N, C, _, _ = x.size()

        if self.training and self.single_batch:
            stride = 16
            assert stride * 4 == N

            assert domain_label[0] == 0
            bn0 = self.bns[0]

            assert domain_label[stride] == 1
            bn1 = self.bns[1]

            assert domain_label[2 * stride] == 2
            bn2 = self.bns[2]

            assert domain_label[3 * stride] == 3
            bn3 = self.bns[3]

            out = [bn0(x[:stride]), bn1(x[stride:2 * stride]), bn2(x[2 * stride:3 * stride]), bn3(x[3 * stride:])]
            out = torch.cat(out)
        else:
            bn = self.bns[domain_label[0]]
            out = bn(x)

        return out, domain_label


class DomainSpecificBatchNorm1d(nn.Module):

    def __init__(self, num_channel, num_domains, eps=1e-9, momentum=0.1, affine=True,
                 track_running_stats=True, within_single_batch=True):
        super(DomainSpecificBatchNorm1d, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_channel, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])
        self.single_batch = within_single_batch

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def bias_requires_grad(self, flag):
        for bn in self.bns:
            bn.bias.requires_grad_(flag)

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        N, C = x.size()
        if self.training and self.single_batch:
            stride = 16
            assert stride * 4 == N

            assert domain_label[0] == 0
            bn0 = self.bns[0]

            assert domain_label[stride] == 1
            bn1 = self.bns[1]

            assert domain_label[2 * stride] == 2
            bn2 = self.bns[2]

            assert domain_label[3 * stride] == 3
            bn3 = self.bns[3]

            out = [bn0(x[:stride]), bn1(x[stride:2 * stride]), bn2(x[2 * stride:3 * stride]), bn3(x[3 * stride:])]
            out = torch.cat(out)
        else:
            bn = self.bns[domain_label[0]]
            out = bn(x)

        return out, domain_label

