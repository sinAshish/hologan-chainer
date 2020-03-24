import numpy as np
import sys
import chainer
from chainer import links as L
from chainer import functions as F
from chainer import Parameter
sys.path.append('../')
from common.transform_unit import TransformUnit
from common.adain import AdaptiveInstanceNormalization
from common.gen_resblocks import Block


class UpsampleBlock(chainer.Chain):

    def __init__(self, dims, in_ch, out_ch, upsample=True):
        self.dims = dims
        self.upsample = upsample
        super().__init__()
        with self.init_scope():
            initial_w = chainer.initializers.Normal(scale=0.2)
            initial_b = chainer.initializers.Zero()
            if dims == 2:
                self.conv = L.Convolution2D(in_ch, out_ch, ksize=3, pad=1, stride=1, initialW=initial_w, initial_bias=initial_b)
            else:
                self.conv = L.ConvolutionND(dims, in_ch, out_ch, ksize=3, pad=1, stride=1, initialW=initial_w, initial_bias=initial_b)
            self.norm = AdaptiveInstanceNormalization(out_ch)

    def __call__(self, x, z):
        h = x
        if self.upsample:
            if self.dims == 2:
                h = F.unpooling_2d(h, 2, outsize=tuple([2*xi for xi in x.shape[-2:]]))
            else:
                h = F.unpooling_nd(h, 2, outsize=tuple([2*xi for xi in x.shape[-3:]]))
        h = self.conv(h)
        h = self.norm(h, z)
        return F.leaky_relu(h)
        
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    N, C = size[:2]
    #import pdb; pdb.set_trace()
    feat_var = feat.reshape(N, C, -1).var(axis=2) + eps
    if len(size) == 4:
        feat_std = feat_var.sqrt().reshape(N, C, 1, 1)
        feat_mean = feat.reshape(N, C, -1).mean(axis=2).reshape(N, C, 1, 1)
    elif len(size) == 5:
        feat_std = feat_var.sqrt().reshape(N, C, 1, 1, 1)
        feat_mean = feat.reshape(N, C, -1).mean(axis=2).reshape(N, C, 1, 1, 1)
    else:
        assert 1 == 0
    return feat_mean, feat_std

def AdaIN(features, style_feat):
    #import pdb; pdb.set_trace()
    partition = style_feat.shape
    partition = partition[1] // 2
    scale, bias = style_feat[:, :partition], style_feat[:, partition:]
    mean, variance = calc_mean_std(features)  # Only consider spatial dimension
    sigma = F.sqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = scale.reshape(mean.shape)
    bias_broadcast = bias.reshape(mean.shape)
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized
    
    

class HoloGANGenerator2(chainer.Chain):

    def __init__(self, ch=512, const_size=4, theta_range=(-np.pi, np.pi)):
        self.ch = ch
        self.f = 2
        self.const_size = const_size
        #self.theta_range = theta_range
        super().__init__()
        with self.init_scope():
            initial_w = chainer.initializers.Normal(scale=0.2)
            initial_b = chainer.initializers.Zero()

            self.const = Parameter(chainer.initializers.Normal(), shape=(1, ch, const_size, const_size, const_size))
            #self.const_bias = L.Bias(axis=1, shape=(ch,))
            self.conv3d_0 = L.Deconvolution3D(ch, ch // self.f, ksize=4, pad=1, stride=2)
            self.adain_0 = L.Linear(None, ch // self.f)
            self.conv3d_1 = L.Deconvolution3D(ch // self.f, ch // (self.f*2), ksize=4, pad=1, stride=2)
            self.adain_1 = L.Linear(None, ch // (self.f*2))
            self.transform = TransformUnit(theta_range=theta_range)
            self.conv3d_2 = L.Convolution3D(None, ch // (self.f*4), ksize=3, pad=1, initialW=initial_w, initial_bias=initial_b)
            self.conv3d_3 = L.Convolution3D(None, ch // (self.f*4), ksize=3, pad=1, initialW=initial_w, initial_bias=initial_b)
            self.proj = L.Convolution2D(None, ch, ksize=1, initialW=initial_w, initial_bias=initial_b)
            self.conv2d_0 = L.Deconvolution2D(None, ch // (self.f*4), ksize=4, pad=1, stride=2)
            self.adain_2 = L.Linear(None, ch // (self.f*4))

            self.conv2d_1 = L.Deconvolution2D(None, ch // (self.f*8), ksize=4, pad=1, stride=2)
            self.adain_3 = L.Linear(None, ch // (self.f*8))

            self.conv2d_2 = L.Deconvolution2D(None, ch // (self.f*16), ksize=3, pad=1, stride=2)
            self.adain_4 = L.Linear(None, ch // (self.f*16))

            self.out = L.Convolution2D(None, 3, ksize=3, pad=1, stride=2, initialW=initial_w, initial_bias=initial_b)

    def __call__(self, z1, z2, theta=None):
        h = F.broadcast_to(self.const, (z1.shape[0],) + self.const.shape[1:])
        #h = F.leaky_relu(self.const_bias(h))

        # 3d network
        h = self.conv3d_0(h)
        z1 = F.leaky_relu(self.adain_0(z1))
        h = F.leaky_relu(AdaIN(h, z1))

        h = self.conv3d_1(h)
        z1 = F.leaky_relu(self.adain_1(z1))
        h = F.leaky_relu(AdaIN(h, z1))

        # transform unit
        #if theta is None:
        #    theta = self.theta_range[0] + self.xp.random.rand()*(self.theta_range[1]-self.theta_range[0])
        h = self.transform(h, theta)

        # projection unit
        h = F.leaky_relu(self.conv3d_2(h))
        h = F.leaky_relu(self.conv3d_3(h))

        # NCHWD 
        #h = F.transpose(h, (0, 2, 3, 4, 1))
        #h = F.reshape(h, h.shape[:3] + (-1,))
        #h = F.transpose(h, (0, 3, 1, 2))

        # NCDHW - most likely this
        # https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#five-D-tensor-descriptor
        h = F.reshape(h, (h.shape[0], -1) + h.shape[3:])
        h = F.leaky_relu(self.proj(h))

        # 2d network
        h = self.conv2d_0(h)
        z2 = F.leaky_relu(self.adain_2(z2))
        h = F.leaky_relu(AdaIN(h, z2))

        h = self.conv2d_1(h)
        z2 = F.leaky_relu(self.adain_3(z2))
        h = F.leaky_relu(AdaIN(h, z2))

        h = self.conv2d_2(h)
        z2 = F.leaky_relu(self.adain_4(z2))
        h = F.leaky_relu(AdaIN(h, z2))

        h = F.tanh(self.out(h))
        return h
    

class HoloGANGenerator(chainer.Chain):

    def __init__(self, ch=512, const_size=4):
        self.ch = ch
        self.f = 2
        self.const_size = const_size
        super().__init__()
        with self.init_scope():
            initial_w = chainer.initializers.Normal(scale=0.2)
            initial_b = chainer.initializers.Zero()
            self.const = Parameter(chainer.initializers.One(), shape=(1, ch, const_size, const_size, const_size))

            #self.conv3d_0 = L.DeconvolutionND(3, ch, ch // self.f, ksize=4, pad=1, stride=2)
            #self.adain_0 = AdaptiveInstanceNormalization(ch // self.f)
            self.block3d_0 = UpsampleBlock(3, ch, ch // self.f)

            #self.conv3d_1 = L.DeconvolutionND(3, ch // self.f, ch // (self.f*2), ksize=4, pad=1, stride=2)
            #self.adain_1 = AdaptiveInstanceNormalization(ch // (self.f*2))
            self.block3d_1 = UpsampleBlock(3, ch // self.f, ch // (self.f*2))

            self.transform = TransformUnit()
            self.conv3d_2 = L.ConvolutionND(3, None, ch // (self.f*4), ksize=3, pad=1, initialW=initial_w, initial_bias=initial_b)
            self.conv3d_3 = L.ConvolutionND(3, None, ch // (self.f*4), ksize=3, pad=1, initialW=initial_w, initial_bias=initial_b)
            self.proj = L.Convolution2D(None, ch, ksize=1, initialW=initial_w, initial_bias=initial_b)
            #self.proj_out = L.Convolution2D(None, ch // (self.f*2), ksize=1)

            #self.conv2d_0 = L.Deconvolution2D(None, ch // (self.f*4), ksize=4, pad=1, stride=2)
            #self.adain_2 = AdaptiveInstanceNormalization(ch // (self.f*4))
            self.block2d_0 = UpsampleBlock(2, None, ch // (self.f))

            #self.conv2d_1 = L.Deconvolution2D(None, ch // (self.f*8), ksize=4, pad=1, stride=2)
            #self.adain_3 = AdaptiveInstanceNormalization(ch // (self.f*8))
            self.block2d_1 = UpsampleBlock(2, None, ch // (self.f*4))

            #self.conv2d_2 = L.Deconvolution2D(None, ch // (self.f*16), ksize=4, pad=1, stride=1)
            #self.adain_4 = AdaptiveInstanceNormalization(ch // (self.f*16))
            #self.block2d_2 = UpsampleBlock(2, None, ch // (self.f*8), upsample=True)

            self.out = L.Convolution2D(None, 3, ksize=1, initialW=initial_w, initial_bias=initial_b)

    def _style_params(self, x):
        # check if this is a 2d or 3d embedding
        idx = (-3, -2, -1) if len(x.shape) > 4 else (-2, -1)
        mu = F.mean(x, axis=idx, keepdims=True)
        std_dev = F.sqrt(F.mean((x - mu)**2, axis=idx))
        return F.concat([F.squeeze(mu), std_dev], axis=1)

    def __call__(self, z1, z2, theta=None):
        #features = []

        # normalize noise
        #z1 /= z1.sum(axis=1, keepdims=True)
        #z2 /= z2.sum(axis=1, keepdims=True)

        #h = self.xp.ones((z1.shape[0], 512, 4, 4, 4), dtype=np.float32)
        h = F.broadcast_to(self.const, (z1.shape[0],) + self.const.shape[1:])

        #h = F.leaky_relu(h)
        #print(h.shape)
        # 3d block 0
        h = self.block3d_0(h, z1)
        #h = self.conv3d_0(h)
        #h = F.leaky_relu(self.adain_0(h, z1))
        #features.append(self._style_params(h))
        #print("1",h.shape)
        # 3d block 1
        h = self.block3d_1(h, z1)
        #h = self.conv3d_1(h)
        #h = F.leaky_relu(self.adain_1(h, z1))
        #features.append(self._style_params(h))

        if theta is None:
            theta = self.xp.random.rand()*2*np.pi
        #print("projection")
        #print("2",h.shape)
        #exit()
        #h = self.transform(h, theta)
        #print(h.shape)
        h = F.leaky_relu(self.conv3d_2(h))

        #print("3",h.shape)
        h = F.leaky_relu(self.conv3d_3(h))
        #print("4",h.shape)
        #exit()

        # projection unit
        #h = F.transpose(h, (0, 2, 3, 4, 1))
        #h = F.reshape(h, h.shape[:3] + (-1,))
        #h = F.transpose(h, (0, 3, 1, 2))
        h = F.reshape(h, (h.shape[0], -1) + h.shape[3:])
        h = F.leaky_relu(self.proj(h))
        #h = F.leaky_relu(self.proj_out(h))
        #print("5",h.shape)
        # 2d block 0
        h = self.block2d_0(h, z2)
        #h = self.conv2d_0(h)
        #h = F.leaky_relu(self.adain_2(h, z2))
        #features.append(self._style_params(h))
        #print("6",h.shape)
        # 2d block 1
        h = self.block2d_1(h, z2)
        #h = self.conv2d_1(h)
        #h = F.leaky_relu(self.adain_3(h, z2))
        #features.append(self._style_params(h))
        #print("7",h.shape)
        # 2d block 2
        #h = self.block2d_2(h, z2)
        #h = self.conv2d_2(h)
        #h = F.leaky_relu(self.adain_4(h, z2))
        #features.append(self._style_params(h))
        #print("8",h.shape)
        #exit()
        h = F.tanh(self.out(h))
        #print(h.shape)
        #exit()
        if chainer.config.train is True:
            return h
        else:
            return h


if __name__ == "__main__":
    z = np.random.rand(1, 128).astype(np.float32)
    
    const = np.random.rand(1, 512, 2, 2, 2).astype(np.float32)
    gen = HoloGANGenerator2()
    y = gen(z, z)
    print(y.shape)
