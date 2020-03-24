import sys
import os
#import tensorflow as tf
import numpy as np
import math
#import scipy.misc
#import time
import chainer.functions as F


def np_interpolate(voxel, x,y,z, out_size):
    """
    Trilinear interpolation for resampling after rotation. Work for batches of voxels
    :param voxel: The whole voxel grid
    :param x,y,z: indices of voxel
    :param output_size: output size of voxel
    :return: numpy array that are with cell value trilinearly interpolated
    """

    batch_size = voxel.shape[0]
    height = voxel.shape[1]
    width  = voxel.shape[2]
    depth  = voxel.shape[3]
    n_channels = voxel.shape[4]


    x = np.float32(x)
    y = np.float32(y)
    z = np.float32(z)

    out_height = out_size[1]
    out_width  = out_size[2]
    out_depth  = out_size[3]

    zero  = np.zeros([],dtype='int32')
    max_y = int(height - 1)
    max_x = int(width - 1)
    max_z = int(depth - 1)

    # do sampling
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)
    z0 = np.clip(z0, zero, max_z)
    z1 = np.clip(z1, zero, max_z)

    #A 1D tensor of base indices describe
    # First index for each shape/map in the whole batch
    # tf.range(batch_size) * width * height * depth : Element to repeat.
    # Each element in the list is incremented by width*height*depth amount
    # out_height * out_width * out_depth: n of repeat. Create chunks of out_height*out_width*out_depth length with the same value created by tf.range(batch_size) *width*height*depth
    base = np.repeat(np.arange(batch_size) * width * height * depth, out_height * out_width * out_depth)
    #Find the Z element of each index
    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height

    #Find the Y element based on Z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # Find the X element based on Y, Z for Z=0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1

    # Find the X element based on Y,Z for Z =1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    voxel_flat = F.reshape(voxel, [-1, n_channels])
    #voxel_flat = np.float32(voxel_flat)

    Ia = F.reshape(F.get_item(voxel_flat, idx_a), [-1,1])
    Ib = F.reshape(F.get_item(voxel_flat, idx_b), [-1,1])
    Ic = F.reshape(F.get_item(voxel_flat, idx_c), [-1,1])
    Id = F.reshape(F.get_item(voxel_flat, idx_d), [-1,1])
    Ie = F.reshape(F.get_item(voxel_flat, idx_e), [-1,1])
    If = F.reshape(F.get_item(voxel_flat, idx_f), [-1,1])
    Ig = F.reshape(F.get_item(voxel_flat, idx_g), [-1,1])
    Ih = F.reshape(F.get_item(voxel_flat, idx_h), [-1,1])

    # and finally calculate interpolated values
    x0_f = np.float32(x0)
    x1_f = np.float32(x1)
    y0_f = np.float32(y0)
    y1_f = np.float32(y1)
    z0_f = np.float32(z0)
    z1_f = np.float32(z1)

    #First slice XY along Z where z=0
    wa = F.expand_dims(((x1_f - x) * (y1_f - y) * (z1_f - z)), 1)
    wb = F.expand_dims(((x1_f - x) * (y - y0_f) * (z1_f - z)), 1)
    wc = F.expand_dims(((x - x0_f) * (y1_f - y) * (z1_f - z)), 1)
    wd = F.expand_dims(((x - x0_f) * (y - y0_f) * (z1_f - z)), 1)
    # First slice XY along Z where z=1
    we = F.expand_dims(((x1_f - x) * (y1_f - y) * (z - z0_f)), 1)
    wf = F.expand_dims(((x1_f - x) * (y - y0_f) * (z - z0_f)), 1)
    wg = F.expand_dims(((x - x0_f) * (y1_f - y) * (z - z0_f)), 1)
    wh = F.expand_dims(((x - x0_f) * (y - y0_f) * (z - z0_f)), 1)

    """output = wa * Ia
    temp_list = [wb * Ib, wc * Ic, wd * Id,  we * Ie, wf * If, wg * Ig, wh * Ih]
    for item in temp_list:
        output = np.add(output, item)
    """

    output = F.add(wa * Ia, wb * Ib, wc * Ic, wd * Id,  we * Ie, wf * If, wg * Ig, wh * Ih)
    return output


def np_voxel_meshgrid(width, depth, height, homogeneous=False):
    """
    Because 'ij' ordering is used for meshgrid, z_t and x_t are swapped (Think about order in 'xy' VS 'ij'
    The range for the meshgrid depends on the width/depth/height input
    :param width
    :param depth
    :param height
    :param concatenating vector to make 4x4 homogeneous matrix
    :return 3D numpy meshgrid
    """

    z_t, y_t, x_t = np.meshgrid(np.arange(depth),
                                np.arange(height),
                                np.arange(width), indexing='ij')

    #Reshape into a big list of slices one after another along the X,Y,Z direction
    x_t_flat = np.reshape(x_t[::-1], (1, -1))
    y_t_flat = np.reshape(y_t, (1, -1))
    z_t_flat = np.reshape(z_t[::-1], (1, -1)) #Default OpenGL setting it to look towards the negative Z dir

    #Vertical stack to create a (3,N) matrix for X,Y,Z coordinates
    grid = np.concatenate([x_t_flat, y_t_flat, z_t_flat], axis=0)
    if homogeneous:
        ones = np.ones_like(x_t_flat)
        grid = np.concatenate([grid, ones], axis = 0)

    return grid


def np_rotation_around_grid_centroid(azimuth, elevation):
    """
    This function returns a rotation matrix around a center with y-axis being the up vector.
    It first rotates the matrix by the azimuth angle (theta) around y, then around X-axis by elevation angle (gamma)
    return a rotation matrix in homogeneous coordinate
    The default Open GL camera is to looking towards the negative Z direction
    This function is suitable when the silhouette projection is done along the Z direction
    :param azimuth
    :param elevation
    :return Rotation matrix around azimuth and along elevation
    """

    # Convert azimuth to positive rotatation direction (right hand rule)
    # so that azimuth at 0 aligns with X-axis, looking into the negative Z axis
    azimuth = -(azimuth + math.pi * 0.5)

    Rot_Y = np.array([[np.cos(azimuth), 0, np.sin(azimuth), 0],
                     [0, 1, 0, 0],
                     [-np.sin(azimuth), 0, np.cos(azimuth), 0],
                     [0, 0, 0, 1]])


    Rot_Z = np.array([[np.cos(elevation), -np.sin(elevation), 0, 0],
                     [np.sin(elevation), np.cos(elevation), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    R = np.matmul(Rot_Z, Rot_Y)

    return R


def np_resampling(voxel_array, transformation_matrix, size=64, new_size=80, nearest_neighbour= True):
    """
    Batch resampling function
    :param voxel_array: Batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size: Input size of voxel grid
    :param new_size: Output size (usually bigger that input) to make sure the rotated input is not cut off
    :param nearest_neighbour: If true, use nearest neightbour interpolation instead of trilinear
    :return: transformed voxel grids with new_size
    """

    #Resampling a voxel array after rotation using nearest neighbour
    batch_size = voxel_array.shape[0]
    target = np.zeros([new_size, new_size, new_size])
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = np.array([[1,0,0, -size * 0.5],
                  [0,1,0, -size * 0.5],
                  [0,0,1, -size * 0.5],
                  [0,0,0,1]])


    #However, since the rotated grid might be out of bound for the original grid size,
    #move the rotated grid to a new bigger grid
    #Precompute T_inv to save time
    T_inv = np.array([[1,0,0, new_size * 0.5],
                  [0,1,0, new_size * 0.5],
                  [0,0,1, new_size * 0.5],
                  [0,0,0,1]])

    scale_factor = 0.1

    R = np.array([[scale_factor, 0, 0, 0],
                     [0, scale_factor, 0, 0],
                     [0, 0, scale_factor, 0],
                     [0, 0, 0, 1]])

    total_M = T_inv.dot(R).dot(transformation_matrix).dot(T)
    total_M = np.linalg.inv(total_M)

    if nearest_neighbour:
        for u in range(new_size):
            for v in range(new_size):
                for w in range(new_size):
                    #Backward mapping
                    q = np.array([u,v,w,1]).T
                    p = np.matmul(total_M, q)

                    x = np.around(p[0] / p[3]).astype(np.int)
                    y = np.around(p[1] / p[3]).astype(np.int)
                    z = np.around(p[2] / p[3]).astype(np.int)

                    if x >= 0 and y >= 0 and z>= 0 and x < size and y < size and z < size:
                        target[u, v, w] = voxel_array[x, y, z]
    else:
        total_M = total_M[0:3, :] #Ignore the homogenous coordinate so the results are 3D vectors
        grid = np_voxel_meshgrid(new_size, new_size, new_size, homogeneous=True)
        grid_transform = np.matmul(total_M, grid)
        input_transformed = np_interpolate(voxel_array, grid_transform[0, :], grid_transform[1, :],
                                              grid_transform[2, :], [batch_size, new_size, new_size, new_size, 1])
        target= np.reshape(
            input_transformed, target.shape)

    return target


def np_batch_rotation_resampling(voxel_array, view_params, size=64, new_size=128):
    """
    Rotate voxel grids
    :param voxel_array: Voxel grids to rotate. Shape [batch_size, height, width, depth, channel]
    :param view_params: Target pose (azimuth, elevation, scale) to transform to. Shape [batch_size, 3]
    :param size: Size of the input voxel grid
    :param new_size: Output size (usually bigger that input) to make sure the rotated input is not cut off
    :return: transformed voxel grids with new_size
    """
    if view_params.shape[1] == 2:
        M = np_batch_rotation_around_grid_centroid(view_params)
        target = np_batch_resampling(voxel_array, M, size=size, new_size=new_size)
    else:
        M, S = np_batch_rotation_around_grid_centroid(view_params)
        target = np_batch_resampling(voxel_array, M, Scale_matrix=S, size=size, new_size=new_size)
    return target


def np_batch_rotation_around_grid_centroid(view_params):
    """
    This function returns a rotation matrix around a center with y-axis being the up vector.
    It first rotates the matrix by the azimuth angle (theta) around y, then around X-axis by elevation angle (gamma)
    return a rotation matrix in homogenous coordinate
    The default Open GL camera is to looking towards the negative Z direction
    This function is suitable when the silhoutte projection is done along the Z direction
    :param view_params: batch of view parameters. Shape : [batch_size, 2] (azimuth-elevation) or [batch_size, 3] (azimuth-elevation-scale)
    :return: Rotation matrix around azimuth and along elevation
    """
    batch_size = view_params.shape[0]
    azimuth = view_params[:, 0]
    elevation = view_params[:, 1]
    azimuth = -(azimuth + math.pi * 0.5)  # Convert azimuth to positive rotatation direction (right hand rule)
    # so that azimuth at 0 aligns with X-axis, looking into the negative X axis


    batch_Rot_Y = np.identity(4).reshape((1, 4, 4))
    batch_Rot_Z = np.identity(4).reshape((1, 4, 4))
    batch_Rot_Y = np.tile(batch_Rot_Y, (batch_size, 1, 1))
    batch_Rot_Z = np.tile(batch_Rot_Z, (batch_size, 1, 1))

    batch_Rot_Y[:, 0, 0] = np.cos(azimuth)
    batch_Rot_Y[:, 0, 2] = np.sin(azimuth)
    batch_Rot_Y[:, 2, 0] = -np.sin(azimuth)
    batch_Rot_Y[:, 2, 2] = np.cos(azimuth)

    batch_Rot_Z[:, 0, 0] = np.cos(elevation)
    batch_Rot_Z[:, 0, 1] = -np.sin(elevation)
    batch_Rot_Z[:, 1, 0] = np.sin(elevation)
    batch_Rot_Z[:, 1, 1] = np.cos(elevation)

    return np.matmul(batch_Rot_Z, batch_Rot_Y)


def np_batch_resampling(voxel_array, transformation_matrix, size=64, new_size=128, scale_factor = 0.5):
    """
    Batch resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size: Input size of voxel grid
    :param new_size: Output size (usually bigger that input) to make sure the rotated input is not cut off
    :return: transformed voxel grids with new_size
    """
    batch_size = np.shape(voxel_array)[0]

    # Aligning the centroid of the object (voxel grid) to origin for rotation,
    # then move the centroid back to the original position of the grid centroid
    T = np.array([[1, 0, 0, -size * 0.5],
                 [0, 1, 0, -size * 0.5],
                 [0, 0, 1, -size * 0.5],
                 [0, 0, 0, 1]])
    T = np.tile(np.reshape(T, (1, 4, 4)), [batch_size, 1, 1])

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = np.array([[1, 0, 0, new_size * 0.5],
                          [0, 1, 0, new_size * 0.5],
                          [0, 0, 1, new_size * 0.5],
                          [0, 0, 0, 1]])

    T_new_inv = np.tile(np.reshape(T_new_inv, (1, 4, 4)), [batch_size, 1, 1])

    R = np.array([[scale_factor, 0, 0, 0],
                  [0, scale_factor, 0, 0],
                  [0, 0, scale_factor, 0],
                  [0, 0, 0, 1]])
    R = np.tile(np.reshape(R, (1, 4, 4)), [batch_size, 1, 1])

    total_M = np.matmul(np.matmul(np.matmul(T_new_inv, R), transformation_matrix), T)
    total_M = np.linalg.inv(total_M)
    total_M = total_M[:, :3, :]

    grid = np.expand_dims(np_voxel_meshgrid(new_size, new_size, new_size, homogeneous=True), axis = 0)
    grid = np.tile(grid, [batch_size, 1, 1])
    grid_transform = np.matmul(total_M, grid)
    x_s_flat =  grid_transform[:, 0, :].reshape([-1])
    y_s_flat =  grid_transform[:, 1, :].reshape([-1])
    z_s_flat =  grid_transform[:, 2, :].reshape([-1])
    #start = time.time()
    input_transformed = np_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat,
                                          [batch_size, new_size, new_size, new_size, 1])

    target = F.reshape(input_transformed, [batch_size, new_size, new_size, new_size, 1])

    return target
