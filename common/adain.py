# https://raw.githubusercontent.com/pfnet-research/chainer-stylegan/master/src/common/networks/component/normalization/adain.py
import warnings

import chainer
from chainer import links as L
from chainer import backend
from chainer import functions as F
from chainer.backends import cuda
from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.normalization import batch_normalization


def do_normalization(x, groups, gamma, beta, eps=1e-5):
    """Group normalization like function, modified for AdaIN.
    This function implements a "group normalization"
    which divides the channels into groups and computes within each group
    the mean and variance, then normalize by these statistics,
    scales and shifts them.
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Batch tensors.
            First dimension of this value must be the size of minibatch and
            second dimension must be the number of channels.
            Moreover, this value must have one or more following dimensions,
            such as height and width.
        groups (int):
            The number of channel groups.
            This value must be a divisor of the number of channels.
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        eps (float): Epsilon value for numerical stability of normalization.
    Returns:
        ~chainer.Variable: The output variable which has the same shape
        as :math:`x`.
    See: `Group Normalization <https://arxiv.org/abs/1803.08494>`_
    """
    if x.ndim <= 2:
        raise ValueError('Input dimension must be grater than 2, '
                         'including batch size dimension '
                         '(first dimension).')

    if not isinstance(groups, int):
        raise TypeError('Argument: \'groups\' type must be (int).')

    xp = backend.get_array_module(x)

    batch_size, channels = x.shape[:2]
    original_shape = x.shape

    if channels % groups != 0:
        raise ValueError('Argument: \'groups\' must be a divisor '
                         'of the number of channel.')

    # By doing this reshaping, calling batch_normalization function becomes
    # equivalent to Group Normalization.
    # And redundant dimension is added in order to utilize ideep64/cuDNN.
    x = reshape.reshape(x, (1, batch_size * groups, -1, 1))

    with cuda.get_device_from_array(x.array):
        dummy_gamma = xp.ones(batch_size * groups).astype(xp.float32)
        dummy_beta = xp.zeros(batch_size * groups).astype(xp.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = batch_normalization.batch_normalization(
            x, dummy_gamma, dummy_beta, eps=eps)

    x = reshape.reshape(x, original_shape)

    target_shape = [batch_size, channels] + [1] * (x.ndim - 2)
    gamma_broadcast = broadcast.broadcast_to(
        reshape.reshape(gamma, target_shape), x.shape)
    beta_broadcast = broadcast.broadcast_to(
        reshape.reshape(beta, target_shape), x.shape)

    return x * gamma_broadcast + beta_broadcast


def AdaIN(x, s_scale, s_bias):
    return do_normalization(x, x.shape[1], s_scale, s_bias)


class AdaptiveInstanceNormalization(chainer.Chain):

    def __init__(self, ch, n_hidden=512):
        self.ch = ch
        self.n_hidden = n_hidden
        super().__init__()
        with self.init_scope():
            self.hidden = L.Linear(None, self.n_hidden)
            self.scale = L.Linear(None, self.ch, initial_bias=chainer.initializers.One())
            self.bias = L.Linear(None, self.ch, initial_bias=chainer.initializers.Zero())

    def __call__(self, x, z):
        h = self.hidden(z)
        h = F.leaky_relu(h)
        s = self.scale(h)
        b = self.bias(h)
        return AdaIN(x, s, b)
