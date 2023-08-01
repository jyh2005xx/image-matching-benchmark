# Copyright 2020 Google LLC, University of Victoria, Czech Technical University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import os
from utils.io_helper import make_clean_folder, copy_file,read_pfm


def extract_depth(img_path, depth_model, cfg):    
    tmp_folder = os.path.join(os.path.dirname(img_path),'tmp')
    make_clean_folder(tmp_folder)
    copy_file(img_path, tmp_folder)
    depth_model = 'dpt_beit_large_512'
    os.system('python third_party/MiDaS/run.py --model_type {} --input_path {} --output_path {} --model_weights {}'.format(depth_model,tmp_folder,tmp_folder,'./third_party/MiDaS/weights/{}.pt'.format(depth_model)))
    depth_file = os.path.join(tmp_folder,os.path.basename(img_path)[:-4]+'-{}.pfm'.format(depth_model))
    dis = read_pfm(depth_file)[0]
    depth = 1/(dis+1e-8)
    depth[depth<=0] = 0
    return depth

def bilinear_interpolation(query_points,value_map):
    # query_points N by [x,y]
    # value_map H by W 
    query_points_00 = np.concatenate([np.floor(query_points[:,[0]]),np.floor(query_points[:,[1]])],axis=1)
    query_points_10 = np.concatenate([np.floor(query_points[:,[0]])+1,np.floor(query_points[:,[1]])],axis=1)
    query_points_01 = np.concatenate([np.floor(query_points[:,[0]]),np.floor(query_points[:,[1]])+1],axis=1)
    query_points_11 = np.concatenate([np.floor(query_points[:,[0]])+1,np.floor(query_points[:,[1]])+1],axis=1)
    idx = np.stack((
                    np.concatenate([query_points_00[:,[1]],query_points_10[:,[1]],query_points_01[:,[1]],query_points_11[:,[1]]],axis=-1),
                    np.concatenate([query_points_00[:,[0]],query_points_10[:,[0]],query_points_01[:,[0]],query_points_11[:,[0]]],axis=-1),
                    np.zeros((query_points.shape[0],4)))
                    ,axis=0).astype(np.uint)
    query_points_4_neighbour = np.expand_dims(value_map,axis=-1)[tuple(idx)]
    # def get_conf(data):
    #     # data: m by n 
    #     mean = np.mean(data,axis=-1,keepdims=True)
    #     data_centered = (data - mean)/mean
    #     conf = np.sqrt(np.mean(data_centered**2,axis=-1))/0.2
    #     return conf
    dist_x = np.stack((query_points_11[:,0]-query_points[:,0],query_points[:,0]-query_points_01[:,0],query_points_10[:,0]-query_points[:,0],query_points[:,0]-query_points_00[:,0]),axis=-1)
    dist_y = np.stack((query_points_11[:,1]-query_points[:,1],query_points_01[:,1]-query_points[:,1],query_points[:,1]-query_points_10[:,1],query_points[:,1]-query_points_00[:,1]),axis=-1)
    dist_weight = dist_x*dist_y
    query_points_value = np.sum(query_points_4_neighbour*dist_weight,axis=-1)
    return query_points_value

def get_lcoal_varience(query_points,value_map,kernel_size = 4):
    # query_points N by [x,y]
    # value_map H by W 
    
    # offset point N by 2
    query_points_00 = np.concatenate([np.floor(query_points[:,[0]]),np.floor(query_points[:,[1]])],axis=1)

    # kernel K*K by 2
    kernel_y = np.repeat((np.arange(kernel_size)-(kernel_size/2-1))[...,np.newaxis],kernel_size,axis=-1)
    kernel_x = np.repeat((np.arange(kernel_size)-(kernel_size/2-1))[np.newaxis,...],kernel_size,axis= 0)
    kernel_xy = np.stack([kernel_x,kernel_y],axis=-1).reshape(-1,2)

    # kernel points K*K by N by 2
    idx_xy = (kernel_xy.reshape(-1,1,2)+query_points_00.reshape(1,-1,2))

    # clip idx with H and W
    H, W = value_map.shape
    idx_xy[:,:,0] = np.clip(idx_xy[:,:,0],0,W-1)
    idx_xy[:,:,1] = np.clip(idx_xy[:,:,1],0,H-1)
    
    # query on value map
    idx = idx_xy.transpose(2,1,0) #2 by N by K*K
    idx[[0,1],:,:] = idx[[1,0],:,:] # swap x y
    idx = np.concatenate([idx,np.zeros((1,idx.shape[1],idx.shape[2]))],axis=0).astype(np.uint)
    query_values = np.expand_dims(value_map,axis=-1)[tuple(idx)] # N by K*K
    def get_cor(data):
        # data: m by n 
        mean = np.mean(data,axis=-1)
        data_centered = data - mean[...,np.newaxis]
        print(data_centered.shape)
        print(mean.shape)
        conf = np.sqrt(np.mean(data_centered**2,axis=-1))/mean
        return conf
    return get_cor(query_values)

def point_mapping(kp1, d1, K_1, K_2, dR, dt):
    # Normalize keypoints using the calibration data
    C_x = K_1[0, 2]
    C_y = K_1[1, 2]
    f_x = K_1[0, 0]
    f_y = K_1[1, 1]
    kp1n = (kp1 - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    # Append depth to key points
    kp1nd = np.concatenate([kp1n * d1, d1], axis=1)

    # Project points from one image to another image
    kp1np = np.matmul(dR[None], kp1nd[..., None]) + dt[None]

    # Move back to canonical plane
    kp1np = np.squeeze(kp1np[:, 0:2] / kp1np[:, [2]])

    #Undo the normalization of the keypoints using the calibration data
    C_x = K_2[0, 2]
    C_y = K_2[1, 2]
    f_x = K_2[0, 0]
    f_y = K_2[1, 1]
    kp1p = kp1np * np.array([[f_x, f_y]]) + np.array([[C_x, C_y]])

    return kp1p

def visualize_corrs(img1, img2, corrs, c=None, mask=None, label=False, save_file = False):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(figsize=(12, 12),frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = .5
    alpha = 1
    if c is None:
        c = np.repeat(np.array(cols[1])[np.newaxis,...],corrs.shape[0],axis=0)
    else:
        c= np.array(cols[1])[np.newaxis,...]*c[...,np.newaxis]

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color= cols[1],
    )
    

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T

    # plt.plot(
    #     xs, ys,
    #     alpha=alpha,
    #     linestyle="-",
    #     linewidth=lw,
    #     aa=False,
    #     color=c,
    # )
    for _x,_y,_c in zip(xs.T,ys.T,c):
        plt.plot(
            _x, _y,
            alpha=alpha,
            linestyle="-",
            linewidth=lw,
            aa=False,
            color=_c,
        ) 
    plt.scatter(xs, ys)

    if label:
        labels = [str(_s) for _s in range(corrs.shape[0])]
        for _x,_l in zip(_x1,labels):
            plt.text(
                _x[0], _x[1], _l
            )        
        for _x,_l in zip(_x2p,labels):
            plt.text(
                _x[0], _x[1], _l
            )        
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    if save_file:
        plt.savefig('corrs.png')
    else:
        plt.show()

