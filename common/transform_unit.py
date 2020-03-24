import numpy as np
import cupy as cp

import chainer
from chainer import functions as F


class TransformUnit(chainer.Chain):

    def __init__(self, theta_range, bandwidth=0.001):
        self.theta_range = theta_range
        self.b = bandwidth
        super().__init__()

    def get_weights(self, x_shape, theta=0., tx=0., ty=0., tz=0.):
        # construct transformation matrix

        mat = self.xp.array([[np.cos(theta), 0, np.sin(theta), tx],
                             [0, 1, 0, ty],
                             [-np.sin(theta), 0, np.cos(theta), tz],
                             [0, 0, 0, 1]], \
                            dtype=np.float32)
        """
        mat = self.xp.array([[ np.cos(theta), np.sin(theta), 0, tx],
                             [-np.sin(theta), np.cos(theta), 0, ty],
                             [ 0,             0,             1, tz],
                             [ 0,             0,             0, 1 ]], dtype=np.float32)
        """
        # make basis
        i = np.linspace(-1, 1, num=x_shape[2])
        j = np.linspace(-1, 1, num=x_shape[3])
        k = np.linspace(-1, 1, num=x_shape[4])
        ii, jj, kk = np.meshgrid(i, j, k)
        pos = self.xp.array([ii.flatten(), jj.flatten(), kk.flatten()]).T.astype(np.float32)

        # pad the input tensor
        pos = F.concat([pos, self.xp.ones((pos.shape[0], 1)).astype(np.float32)], axis=1)
        pos_2 = F.matmul(pos, mat, transb=True)
        #print(pos.shape, pos_2.shape)
        dist = pos[:, None, :3] - pos_2[:, :3]
        # calculate euclidean distance
        dist = F.sqrt(F.sum(dist**2, axis=2))

        # apply kernel
        norm = 1. / (np.sqrt(2*np.pi*self.b*self.b))
        prob = norm * F.exp(-dist**2 / (2*self.b*self.b))
        return prob

    def __call2__(self, x, theta):
        x_shape = x.shape
        # first transform the input features
        w = self.get_weights(x.shape, theta, 10.0)
        #from matplotlib import pyplot as plt
        #w = F.reshape(w, [16] * 6)
        #plt.imsave("weights.png", w[:,0,0,:,0,0].data)
        #print(w[:,0,0,:,0,0])
        #exit()

        x = F.reshape(F.transpose(x, (0, 2, 3, 4, 1)), (x.shape[0], -1, x.shape[1]))
        #print(x.shape, w.shape)
        #exit()
        y = F.einsum("ijl,jk->ijl", x, w)

        return F.reshape(y, x_shape)

    def density(self, samples, x, bandwidth=None):
        if bandwidth is None:
            bandwidth = samples.std(keepdims=True)

        b2 = bandwidth * bandwidth
        norm = 1. / (self.xp.sqrt(2 * np.pi * b2))

        x = x[:, None, :]
        distance = x - samples
        prob = norm * self.xp.exp(-distance**2 / (2 * b2))
        #print(prob.shape, prob.min(), prob.max())
        prob = self.xp.prod(prob, axis=2)
        #prob = np.sqrt(np.prod(prob, axis=2))
        #print(prob.min(), prob.max())
        #return prob.sum(axis=1)/ prob.shape[1]**2
        return (prob / prob.shape[1]).astype(np.float32)

    def nearest_neighbours(self, samples, x):
        x = x[:, None, :]
        dist = self.xp.sqrt(self.xp.sum((x - samples)**2, axis=2))

        nn = self.xp.argmin(dist, axis=1, keepdims=True)
        #print(nn[0], dist)

        ret = self.xp.zeros_like(dist, dtype=np.float32)
        ret[self.xp.arange(dist.shape[0]), dist.argmin(axis=1)] = 1.0
        return ret

    def trilinear(self, data, fx, fy, fz):
        print(data.shape, fx.shape, fy.shape, fz.shape)
        z = F.pad(data, pad_width=[(0,0), (1,1), (1,1), (1,1)], mode="constant", constant_values=0.)
        c, d, h, w = z.shape

        # clip
        fx = self.xp.clip(fx+1, 0., data.shape[1] + 0.99)
        fy = self.xp.clip(fy+1, 0., data.shape[2] + 0.99)
        fz = self.xp.clip(fz+1, 0., data.shape[3] + 0.99)

        xw, ix_low = self.xp.modf(fx)
        yw, iy_low = self.xp.modf(fy)
        zw, iz_low = self.xp.modf(fz)
        xw = F.expand_dims(xw.astype(np.float32), axis=0)
        yw = F.expand_dims(yw.astype(np.float32), axis=0)
        zw = F.expand_dims(zw.astype(np.float32), axis=0)
        #xw = F.expand_dims(xw, axis=0)
        #yw = F.expand_dims(yw, axis=0)
        #zw = F.expand_dims(zw, axis=0)
        ix_low = ix_low.astype(np.int32)
        iy_low = iy_low.astype(np.int32)
        iz_low = iz_low.astype(np.int32)
        ix_high = ix_low + 1
        iy_high = iy_low + 1
        iz_high = iz_low + 1

        def get_element(i, j, k):
            return z[:, i, j, k]

        x_0_y_0_z_0 = get_element(ix_low, iy_low, iz_low)
        x_1_y_0_z_0 = get_element(ix_high, iy_low, iz_low)
        x_0_y_1_z_0 = get_element(ix_low, iy_high, iz_low)
        x_1_y_1_z_0 = get_element(ix_high, iy_high, iz_low)

        x_0_y_0_z_1 = get_element(ix_low, iy_low, iz_high)
        x_1_y_0_z_1 = get_element(ix_high, iy_low, iz_high)
        x_0_y_1_z_1 = get_element(ix_low, iy_high, iz_high)
        x_1_y_1_z_1 = get_element(ix_high, iy_high, iz_high)

        #print(xw.shape, x_0_y_0_z_0.shape, x_1_y_0_z_0.shape)
        #exit()
        vx_0_z0 = F.linear_interpolate(1-xw, x_0_y_0_z_0, x_1_y_0_z_0)
        vx_1_z0 = F.linear_interpolate(1-xw, x_0_y_1_z_0, x_1_y_1_z_0)
        vxy_z0 = F.linear_interpolate(1-yw, vx_0_z0, vx_1_z0)
        #return F.transpose(vxy, (0, 2, 1))
        vx_0_z1 = F.linear_interpolate(1-xw, x_0_y_0_z_1, x_1_y_0_z_1)
        vx_1_z1 = F.linear_interpolate(1-xw, x_0_y_1_z_1, x_1_y_1_z_1)
        vxy_z1 = F.linear_interpolate(1-yw, vx_0_z1, vx_1_z1)
        #return vxy_z1
        vxyz = F.linear_interpolate(1-zw, vxy_z0, vxy_z1)
        return F.transpose(vxyz, (0, 2, 1, 3))

    def batched_trilinear(self, data, x, y, z):

        def flatten_3d_index(x, y, z, h, w):
            return h*w*z + y*w + x

        #print(data.shape, x.shape)
        data = F.pad(data, pad_width=[(0,0),(0,0),(1,1),(1,1),(1,1)], mode="constant", constant_values=0.)
        b, c, d, h, w = data.shape
        #print(data.shape)

        flat_data = F.reshape(F.transpose(data, (0, 2, 3, 4, 1)), (-1, c))

        out_size = x.shape[1]

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        # clip
        x = self.xp.clip(x+1, 0., h-2 + 0.999)
        y = self.xp.clip(y+1, 0., w-2 + 0.999)
        z = self.xp.clip(z+1, 0., d-2 + 0.999)

        xw, x_low = self.xp.modf(x)
        x_low = x_low.astype(np.int32)
        x_high = x_low + 1
        yw, y_low = self.xp.modf(y)
        y_low = y_low.astype(np.int32)
        y_high = y_low + 1
        zw, z_low = self.xp.modf(z)
        z_low = z_low.astype(np.int32)
        z_high = z_low + 1

        base = self.xp.repeat(self.xp.arange(b) * w * h * d, out_size).astype(np.int32)

        x0y0z0 = flat_data[base + flatten_3d_index(x_low, y_low, z_low, h, w),:]
        x1y0z0 = flat_data[base + flatten_3d_index(x_high, y_low, z_low, h, w),:]
        x0y1z0 = flat_data[base + flatten_3d_index(x_low, y_high, z_low, h, w),:]
        x1y1z0 = flat_data[base + flatten_3d_index(x_high, y_high, z_low, h, w),:]

        x0y0z1 = flat_data[base + flatten_3d_index(x_low, y_low, z_high, h, w),:]
        x1y0z1 = flat_data[base + flatten_3d_index(x_high, y_low, z_high, h, w),:]
        x0y1z1 = flat_data[base + flatten_3d_index(x_low, y_high, z_high, h, w),:]
        x1y1z1 = flat_data[base + flatten_3d_index(x_high, y_high, z_high, h, w),:]

        xw = self.xp.broadcast_to(xw[:, None], x0y0z0.shape)
        yw = self.xp.broadcast_to(yw[:, None], x0y0z0.shape)
        zw = self.xp.broadcast_to(zw[:, None], x0y0z0.shape)

        v0a = F.linear_interpolate(1-xw, x0y0z0, x1y0z0)
        v1a = F.linear_interpolate(1-xw, x0y1z0, x1y1z0)
        va = F.linear_interpolate(1-yw, v0a, v1a)

        v0b = F.linear_interpolate(1-xw, x0y0z1, x1y0z1)
        v1b = F.linear_interpolate(1-xw, x0y1z1, x1y1z1)
        vb = F.linear_interpolate(1-yw, v0b, v1b)

        v = F.linear_interpolate(1-zw, va, vb)

        return F.transpose(F.reshape(v, (b, -1, c)), (0, 2, 1))

    def __call__(self, x, theta):
        if theta is None:
            theta = self.xp.random.uniform(self.theta_range[0],
                                           self.theta_range[1],
                                           size=(x.shape[0]))
        theta = np.radians(theta)

        x_shape = x.shape

        m_stack = []
        for th in theta:
            #th = 0.
            tx, ty, tz = 0., 0., 0.

            mat = self.xp.array([[[np.cos(th), 0, np.sin(th), tx],
                                 [0, 1, 0, ty],
                                 [-np.sin(th), 0, np.cos(th), tz],
                                 [0, 0, 0, 1]]], \
                                dtype=np.float32)
            """mat = self.xp.array([[[np.cos(th), np.sin(th), 0, tx],
                                  [-np.sin(th), np.cos(th), 0, ty],
                                  [0, 0, 1, tz],
                                  [0, 0, 0, 1]]], \
                                dtype=np.float32)"""
            """mat = self.xp.array([[[1., 0, 0, 0],
                                  [0,  np.cos(th), np.sin(th), 0],
                                  [0, -np.sin(th), np.cos(th), 0],
                                  [0, 0, 0, 1]]], dtype=np.float32)
            """
            
            m_stack.append(mat)

        mat = self.xp.concatenate(m_stack)

        size = x.shape[-1]

        T = self.xp.array([[1., 0, 0, -size * 0.5],
                           [0,  1, 0, -size * 0.5],
                           [0,  0, 1, -size * 0.5],
                           [0,  0, 0, 1]], dtype=np.float32)
        T = self.xp.tile(self.xp.reshape(T, (1, 4, 4)), [mat.shape[0], 1, 1])
        T_inv = self.xp.array([[1., 0, 0, size * 0.5],
                               [0,  1, 0, size * 0.5],
                               [0,  0, 1, size * 0.5],
                               [0,  0, 0, 1]], dtype=np.float32)
        T_inv = self.xp.tile(self.xp.reshape(T_inv, (1, 4, 4)), [mat.shape[0], 1, 1])
        scale_factor = 0.9
        R = self.xp.array([[scale_factor, 0, 0, 0],
                           [0, scale_factor, 0, 0],
                           [0, 0, scale_factor, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        R = self.xp.tile(self.xp.reshape(R, (1, 4, 4)), [mat.shape[0], 1, 1])
        
        total_M = F.matmul(F.matmul(F.matmul(T_inv, R), mat), T).data
        #total_M = F.matmul(F.matmul(R, mat), T).data
        total_M = self.xp.array(np.linalg.inv(total_M.get()), dtype=np.float32)
        
        # make basis vectors
        ##i = np.linspace(-1., 1, num=x_shape[2])
        ##j = np.linspace(-1., 1, num=x_shape[3])
        #k = np.linspace(-1., 1, num=x_shape[4])
        i = np.arange(x_shape[2])
        j = np.arange(x_shape[3])
        k = np.arange(x_shape[4])
        ii, jj, kk = np.meshgrid(i, j, k)
        pos = self.xp.array([ii.flatten(), jj.flatten(), kk.flatten()]).T.astype(np.float32)

        # pad the input tensor
        pos = self.xp.concatenate([pos, self.xp.ones((pos.shape[0], 1)).astype(np.float32)], axis=1)

        pos = self.xp.broadcast_to(pos[None, :, :], (mat.shape[0],) + pos.shape)
        #pos_rot = F.matmul(pos, mat, transb=True).data
        pos_rot = F.matmul(pos, total_M, transb=True).data
        
        fx = (pos_rot[:,:,0])# * x.shape[2]# * 1.25
        fy = (pos_rot[:,:,1])# * x.shape[3]# * 1.25
        fz = (pos_rot[:,:,2])# * x.shape[4]# * 1.25
        
        y = self.batched_trilinear(x, fx, fy, fz)
        y = F.reshape(y, x.shape)
        y = F.transpose(y, (0, 1, 4, 2, 3))
        return y

        #for b in range(x.shape[0]):
        #    y = self.trilinear(x[b], pos_rot[b, :, 0], pos_rot[b, :, 1], pos_rot[b, :, 2])
        #    print(y.shape)

        #    exit()
        #w = self.nearest_neighbours(pos, pos_rot)

        w = self.density(pos, pos_rot, bandwidth=0.001)
        #print(w.shape)
        #print(pos.shape, pos_rot.shape)
        #exit()
        x = F.reshape(F.transpose(x, (0, 2, 1, 4, 3)), (x.shape[0], -1, x.shape[1]))
        #print(x.shape, w.shape)
        #exit()
        y = F.einsum("ijl,jk->ijl", x, w)

        # reweight
        w = self.xp.nan_to_num(1. / w.sum(axis=1))

        y *= F.broadcast_to(w[None, :, None], y.shape)
        #print(y.data.min(), y.data.max())
        return F.reshape(y, x_shape)


if __name__ == "__main__":
    transform = TransformUnit(0.1)
    #transform.to_gpu(0)

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.random.randn(1, 64, 16, 16, 16).astype(np.float32)
    y = transform(x, [360.])

    #print(y.shape)
    print(y.data.min(), y.data.max(), F.mean(x-y))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
