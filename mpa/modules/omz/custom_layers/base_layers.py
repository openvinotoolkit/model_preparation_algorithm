# import numpy as np
import torch
# import torchvision.transforms as transforms
# from torch.nn import *
from torch import nn
from torch.autograd import Variable


class Input(nn.Module):
    def __init__(self, dims):
        super(Input, self).__init__()
        self.dims = dims

    def __repr__(self):
        return 'Input %s' % self.dims

    def forward(self, inputs):
        return inputs


class Label(nn.Module):
    def __init__(self):
        super(Label, self).__init__()

    def __repr__(self):
        return 'Labels'

    def forward(self, labels):
        return labels


class Const(nn.Module):
    def __init__(self, data):
        super(Const, self).__init__()
        # self.data = torch.nn.Parameter(data=torch.from_numpy(data), requires_grad=False)
        self.data = torch.from_numpy(data)
        # self.register_buffer('_data', self.data)
        # self.register_buffer('data', torch.from_numpy(data))

    def __repr__(self):
        return 'Const'

    def forward(self):
        return self.data


class ScaleShift(nn.Module):
    def __init__(self, dims, is_trainable=False):
        super(ScaleShift, self).__init__()
        self.dims = dims
        self.weight = torch.nn.Parameter(data=torch.zeros(dims[1], dtype=torch.float32), requires_grad=is_trainable)
        self.bias = torch.nn.Parameter(data=torch.zeros(dims[1], dtype=torch.float32), requires_grad=is_trainable)
        # self.register_buffer('weight', torch.zeros(dims[1], dtype=torch.float32))
        # self.register_buffer('bias', torch.zeros(dims[1], dtype=torch.float32))

    def __repr__(self):
        return 'ScaleShift %s' % self.dims

    def forward(self, inputs):
        return self.weight[None, :, None, None]*inputs + self.bias[None, :, None, None]


class Padding(nn.Module):
    def __init__(self, padding):
        super(Padding, self).__init__()
        self.padding = padding  # (x1,x2,y1,y2)

    def __repr__(self):
        return 'Padding %s' % self.padding

    def forward(self, inputs):
        return nn.functional.pad(inputs, self.padding)


class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.initialized = False
        self.num_init_iter = 0
        self.max_init_iter = 10

    def forward(self, input):
        output = super().forward(input)

        if self.training and not self.initialized:
            # Use linear output at the first few iteration
            output = input*self.weight[None, :, None, None] + self.bias[None, :, None, None]
            # output = input*self.weight.detach()[None,:,None, None] + self.bias.detach()[None,:,None, None]
            self.num_init_iter += 1
            if self.num_init_iter >= self.max_init_iter:
                # Adapt weight & bias using the first batch statistics to undo normalization approximately
                self.weight.data = self.weight.data * self.running_var
                self.bias.data = self.bias.data + (self.running_mean/(self.running_var + self.eps))
                self.initialized = True

        return output


class Conv2dPadBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, isbias, isBN, training):
        super(Conv2dPadBN, self).__init__()
        self.isbias = isbias
        self.isBN = isBN
        self.training = training
        self.pad = Padding(padding)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              dilation=dilation, groups=groups, bias=isbias)
        self.weight = self.conv.weight

        if isbias:
            self.bias = self.conv.bias
            if isBN:
                if self.training:
                    self.bn = AdaptiveBatchNorm2d(out_channels, eps=1.0e-10).train()
                else:
                    self.bn = AdaptiveBatchNorm2d(out_channels, eps=1.0e-10).eval()
                nn.init.ones_(self.bn.weight.data)
                nn.init.zeros_(self.bn.bias.data)
                nn.init.ones_(self.bn.running_var)
                nn.init.zeros_(self.bn.running_mean)

    def __repr__(self):
        return 'Padding + Conv2D + BN'

    def forward(self, inputs):
        x = self.pad(inputs)
        x = self.conv(x)
        if self.isbias and self.isBN:
            x = self.bn(x)
        return x


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, isbias, isBN, training):
        super(Conv2dBN, self).__init__()
        self.isbias = isbias
        self.isBN = isBN
        self.training = training
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              dilation=dilation, groups=groups, bias=isbias)
        self.weight = self.conv.weight

        if isbias:
            self.bias = self.conv.bias
            if isBN:
                if self.training:
                    self.bn = AdaptiveBatchNorm2d(out_channels, eps=1.0e-10).train()
                else:
                    self.bn = AdaptiveBatchNorm2d(out_channels, eps=1.0e-10).eval()
                nn.init.ones_(self.bn.weight.data)
                nn.init.zeros_(self.bn.bias.data)
                nn.init.ones_(self.bn.running_var)
                nn.init.zeros_(self.bn.running_mean)

    def __repr__(self):
        return repr(self.conv) + '\n     ' + repr(self.bn)

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.isbias and self.isBN:
            x = self.bn(x)
        return x


class MaxPool2DPad(nn.Module):
    def __init__(self, kernel_size, stride, padding, ceil_mode):
        super(MaxPool2DPad, self).__init__()
        self.pad = Padding(padding)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)

    def __repr__(self):
        return 'Padding + MaxPooling2D'

    def forward(self, inputs):
        x = self.pad(inputs)
        return self.maxpool(x)


class AvgPool2DPad(nn.Module):
    def __init__(self, kernel_size, stride, padding, ceil_mode):
        super(AvgPool2DPad, self).__init__()
        self.pad = Padding(padding)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)

    def __repr__(self):
        return 'Padding + AvgPooling2D'

    def forward(self, inputs):
        x = self.pad(inputs)
        return self.avgpool(x)


class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, inputs):
        if self.operation == '+' or self.operation == 'sum':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x + inputs[i]
        elif self.operation == 'prod' or self.operation == 'mul':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = torch.mul(x, inputs[i])
        elif self.operation == '/' or self.operation == 'div':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'max':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x


class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, inputs):
        return torch.cat(inputs, self.axis)


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Flatten(axis=%d)' % self.axis

    def forward(self, x):
        left_size = 1
        for i in range(self.axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1).contiguous()


class Reshape(nn.Module):
    def __init__(self, dims, from_const=True, post_nms_topn=0):
        super(Reshape, self).__init__()
        self.dims = dims
        self.from_const = from_const
        self.post_nms_topn = int(post_nms_topn)
        if self.from_const and self.post_nms_topn != 0 and (self.dims[0] >= self.post_nms_topn):
            self.ratio = int(self.dims[0]/self.post_nms_topn)
        else:
            self.ratio = 1

    def __repr__(self):
        return 'Reshape(dims=%s)' % (self.dims)

    def forward(self, x):
        if self.from_const:
            x = x[0]
        orig_dims = x.size()
        new_dims = [orig_dims[i] if self.dims[i] == 0 else self.dims[i] for i in range(len(self.dims))]
        if self.from_const and (self.dims[0] >= self.post_nms_topn):
            new_dims[0] = orig_dims[0]*self.ratio
        # print('.......origin: ', orig_dims)
        # print('.......new: ', new_dims)

        return x.reshape(new_dims)


class Crop(nn.Module):
    def __init__(self, axis, offset, dim):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset
        self.ref = dim

    def __repr__(self):
        return 'Crop()'

    def forward(self, x):
        idx = 0
        ref = x.shape
        x = torch.from_numpy(x)
        for axis in self.axis:
            ref_size = ref[idx]
            offset = int(self.offset[idx])
            indices = torch.arange(offset, ref_size).long()
            # indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = torch.index_select(x, int(axis), Variable(indices))
            idx += 1
        return x


class Power(nn.Module):
    def __init__(self, power, scale, shift):
        super(Power, self).__init__()
        self.power = power
        self.scale = scale
        self.shift = shift

    def __repr__(self):
        return 'Power(power=%d, scale=%d, shift=%d)' % (self.power, self.scale, self.shift)

    def forward(self, x):
        x = torch.pow(x, self.power)
        x = x * self.scale
        x = x + self.shift
        return x


class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def __repr__(self):
        return 'Permute'

    def forward(self, x):
        return x.permute(self.order)


class Interpolate(nn.Module):
    def __init__(self, align_corners, pad_beg, pad_end):
        super(Interpolate, self).__init__()
        if align_corners == 1:
            self.align_corners = True
        else:
            self.align_corners = False
        self.pad_beg = pad_beg
        self.pad_end = pad_end

    def __repr__(self):
        return 'Interpolate'

    def forward(self, x):
        size = tuple((int(x.shape[2]*2), int(x.shape[3]*2)))
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=self.align_corners)


class GAPooling(nn.Module):
    def __init__(self, stride, padding, ceiling):
        super(GAPooling, self).__init__()
        self.stride = stride
        self.padding = padding
        self.ceiling = ceiling

    def __repr__(self):
        return 'GAPooling'

    def forward(self, x):
        kernel_size = (int(x.shape[2]), int(x.shape[3]))
        return nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=self.stride,
                                        padding=self.padding, ceil_mode=self.ceiling)
        # return x.mean([2,3])
