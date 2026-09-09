"""CUT generator used for virtual staining inference (resnet_9blocks)."""
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise ValueError('filt_size %s is not supported' % filt_size)

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1]
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad2d
    if pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad2d
    if pad_type == 'zero':
        return nn.ZeroPad2d
    raise NotImplementedError('Pad type [%s] not recognized' % pad_type)


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == 'none':
        def norm_layer(_x):
            return Identity()
        return norm_layer
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def init_weights(net, init_type='xavier', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='xavier', init_gain=0.02, gpu_ids=None, initialize_weights=True):
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='instance',
             use_dropout=False, init_type='xavier', init_gain=0.02, no_antialias=False,
             no_antialias_up=False, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        n_blocks = 9
    elif netG == 'resnet_6blocks':
        n_blocks = 6
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    net = ResnetGenerator(
        input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
        no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=n_blocks,
    )
    return init_net(net, init_type, init_gain, gpu_ids)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type='reflect', no_antialias=False, no_antialias_up=False,
                 n_downsampling=2):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2),
                ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                       padding=1, output_padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    Upsample(ngf * mult),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
