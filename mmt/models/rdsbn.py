from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
import torch


class RcBatchNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(RcBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.cfc = Parameter(torch.Tensor(num_features, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def recalibration(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)

        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)

        return g

    def forward(self, input, epochs=-1):
        self._check_input_dim(input)

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

        if not self.training and epochs == 0:
            # print('epoch 0 for stable clustering.')
            out = out_bn
        else:
            g = self.recalibration(input)
            out = out_bn * g

        return out


class DomainSpecificRcBatchNorm2d(nn.Module):

    def __init__(self, num_channel, num_domains, eps=1e-9, momentum=0.1, affine=True,
                 track_running_stats=True, within_single_batch=False):
        super(DomainSpecificRcBatchNorm2d, self).__init__()
        self.bns = nn.ModuleList(
            [RcBatchNorm2d(num_channel, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])
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

        epochs = -1
        if x.size(0) != domain_label.size(0):
            epochs = domain_label[-1]

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

            out = [bn0(x[:stride], epochs), bn1(x[stride:2*stride], epochs),
                   bn2(x[2 * stride: 3 * stride], epochs),
                   bn3(x[3 * stride:], epochs)]
            out = torch.cat(out)
        else:
            bn = self.bns[domain_label[0]]
            out = bn(x, epochs)

        return out, domain_label

