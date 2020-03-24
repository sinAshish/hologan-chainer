import math
import chainer
import chainer.links as L
from chainer import functions as F
from common.instance_normalization import InstanceNormalization
from common.adain import AdaptiveInstanceNormalization


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, dims=2):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        self.dims = dims
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            if self.dims == 2:
                self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            else:
                self.c1 = L.ConvolutionND(self.dims, in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
                self.c2 = L.ConvolutionND(self.dims, hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.b1 = AdaptiveInstanceNormalization(in_channels) #L.BatchNormalization(in_channels)
            self.b2 = AdaptiveInstanceNormalization(hidden_channels) #L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                if self.dims == 2:
                    self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
                else:
                    self.c_sc = L.ConvolutionND(self.dims, in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
    def residual(self, x, z, **kwargs):
        h = x
        h = self.b1(h, z, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, z, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)
