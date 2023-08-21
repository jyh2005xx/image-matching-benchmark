import numpy as np
import cv2

def get_epi_dist(pt_1,pt_2,F,eps=1e-9):
    # compute distance to epipolar line given F (pixel space)
    # pt_1: N by 2
    # pt_2: N by 2
    # F: 3 by 3
    pt_1_homo = np.concatenate((pt_1,np.ones((pt_1.shape[0],1))),axis=-1)
    pt_2_homo = np.concatenate((pt_2,np.ones((pt_2.shape[0],1))),axis=-1)
    epi_line_2 = np.matmul(pt_1_homo,F.T)
    epi_line_1 = np.matmul(pt_2_homo,F)
    
    dis_epi_line_2 = np.sum(pt_2_homo*epi_line_2,axis=-1)**2/(epi_line_2[:,0]**2+epi_line_2[:,1]**2+eps)
    dis_epi_line_1 = np.sum(pt_2_homo*epi_line_2,axis=-1)**2/(epi_line_1[:,0]**2+epi_line_1[:,1]**2+eps)
    dis = dis_epi_line_1+dis_epi_line_2
    
    return np.sqrt(dis)

def get_rep_error(pt1,d1,pt2,d2,K1,K2,pose_1_2):            
    pt_1_cam = np.matmul(np.linalg.inv(K1),np.concatenate([pt1, np.ones((pt1.shape[0],1))],axis=-1).T).T*d1
    pt_2_cam = np.matmul(np.linalg.inv(K2),np.concatenate([pt2, np.ones((pt2.shape[0],1))],axis=-1).T).T*d2
    pt_1_cam_homo = np.concatenate([pt_1_cam, np.ones((pt_1_cam.shape[0],1))],axis=-1)
    pt_2_cam_homo = np.concatenate([pt_2_cam, np.ones((pt_2_cam.shape[0],1))],axis=-1)
    pt_1_cam_2 = np.matmul(pose_1_2,pt_1_cam_homo.T).T
    pt_2_cam_1 = np.matmul(np.linalg.inv(pose_1_2),pt_2_cam_homo.T).T
    pt_1_img_2 = np.matmul(K2,pt_1_cam_2[:,:3].T).T
    pt_1_img_2 = pt_1_img_2/pt_1_img_2[:,[2]]
    pt_1_img_2 = pt_1_img_2[:,:2]
    pt_2_img_1 = np.matmul(K1,pt_2_cam_1[:,:3].T).T
    pt_2_img_1 = pt_2_img_1/pt_2_img_1[:,[2]]
    pt_2_img_1 = pt_2_img_1[:,:2]
    rep_err_1 = np.sqrt(np.sum((pt1 - pt_2_img_1)**2,axis=-1))
    rep_err_2 = np.sqrt(np.sum((pt2 - pt_1_img_2)**2,axis=-1))
    rep_err = (rep_err_1+rep_err_2)/2
    return rep_err

def depth_sac(corrs, d_1, d_2, d_1_valid, d_2_valid, K_1, K_2, max_iter = 500000, epi_thold = 0.5, rep_thold=10):
    best_E = None
    best_mask = None
    min_sac_sample = 8
    min_dep_sample = 4
    idx = np.arange(corrs.shape[0])
    max_inlier = 0
    d_valid = np.logical_and(d_1_valid, d_2_valid)
    # pt_1_xy_img_homo = np.concatenate((corrs[:,:2],np.ones((corrs.shape[0],1))),axis=-1)
    # pt_2_xy_img_homo = np.concatenate((corrs[:,2:],np.ones((corrs.shape[0],1))),axis=-1)
    for _iter in range(max_iter):
    # for _ in range (10):
        # sample a subset
        sub_idx = np.random.choice(idx,min_sac_sample,replace=False)
        sub_corrs = corrs[sub_idx]
        sub_d_1 = d_1[sub_idx]
        sub_d_2 = d_2[sub_idx]
        sub_d_valid = np.logical_and(d_1_valid[sub_idx],d_2_valid[sub_idx])
        num_valid_dep_points = np.sum(sub_d_valid)
        if num_valid_dep_points<min_dep_sample:
            continue

        # compute F using 8 points
        F, _ = cv2.findFundamentalMat(sub_corrs[:, :2], sub_corrs[:, 2:], cv2.FM_8POINT)

        if F is None:
            continue
        
        sub_epi_err = np.mean(get_epi_dist(sub_corrs[:,:2],sub_corrs[:,2:],F))
        if sub_epi_err>1:
            continue
        # get E
        E = K_2.T @ F @ K_1

        # get pose from E
        sub_pt_norm_1 = np.matmul(np.linalg.inv(K_1),np.concatenate([sub_corrs[:,:2], np.ones((sub_corrs.shape[0],1))],axis=-1).T).T
        sub_pt_norm_2 = np.matmul(np.linalg.inv(K_2),np.concatenate([sub_corrs[:,2:], np.ones((sub_corrs.shape[0],1))],axis=-1).T).T
        points, R, t, mask = cv2.recoverPose(E, sub_pt_norm_1[:,:2], sub_pt_norm_2[:,:2])
        pose_1_2 = np.identity(4)
        pose_1_2[:3,:3] = R
        pose_1_2[:3,[3]] = t

        # solve scale using depth and pose
        sub_pt_norm_1 = sub_pt_norm_1[sub_d_valid]
        sub_pt_norm_2 = sub_pt_norm_2[sub_d_valid]
        sub_d_1 = sub_d_1[sub_d_valid]
        sub_d_2 = sub_d_2[sub_d_valid]

        # solve linear system to get mapping of d1 and d2
        # R[x1,y1,1].T*(a*d1+b)+t = [x2,y2,1].T*(m*d2+n)
        # [R[x1,y1,1].T*d_1,R[x1,y1,1].T,-[x_2,y_2,1].T*d_2,[x_2,y_2,1].T]*[a,b,m,n].T = -t
        # A*[a,b].T = B
        # [a,b].T = A'B
        A = np.concatenate(((np.matmul(R,sub_pt_norm_1.T)*sub_d_1).T.reshape(-1,1),
                                np.matmul(R,sub_pt_norm_1.T).T.reshape(-1,1),
                                -(sub_pt_norm_2*sub_d_2[...,np.newaxis]).reshape(-1,1),
                                -(sub_pt_norm_2).reshape(-1,1)
                                ),axis=-1)
        B = -np.repeat(t.T,sub_pt_norm_1.shape[0],axis=0).reshape(-1,1)
        lin_mapping = np.matmul(np.linalg.pinv(A),B)

        # calculate epipolar distance
        epi_err = get_epi_dist(corrs[:,:2],corrs[:,2:],F)
        epi_mask = epi_err<epi_thold
        
        # calculate reprojection error for all points
        d_1_mapped = d_1*lin_mapping[0,0] + lin_mapping[1,0]
        d_2_mapped = d_2*lin_mapping[2,0] + lin_mapping[3,0]
        rep_err = get_rep_error(corrs[:,:2],d_1_mapped[...,np.newaxis],corrs[:,2:],d_2_mapped[...,np.newaxis],K_1,K_2,pose_1_2)
        rep_mask = np.logical_and(rep_err<rep_thold,d_valid)

        inlier_mask = np.logical_and(epi_mask,rep_mask)
        num_inlier = np.sum(inlier_mask)

        if max_inlier < num_inlier:
            max_inlier = num_inlier
            best_mask = inlier_mask
            best_E = E
            # best_R = R
            # best_t = t
    # print('num_inlier{}'.format(max_inlier))
    # import IPython
    # IPython.embed()
    # assert(0)
    return best_E, best_mask

def run_pair(folder_path,pair,sac_type,depth_extention):
    img1 = pair.split('-')[0]
    img2 = pair.split('-')[1]
    # get kp
    with h5py.File(os.path.join(folder_path,'keypoints.h5'),'r') as f:
        kp1 = np.array(f[img1])
        kp2 = np.array(f[img2])
    # get desc
    with h5py.File(os.path.join(folder_path,'descriptors.h5'),'r') as f:
        desc1 = np.array(f[img1]).astype(np.float32)
        desc2 = np.array(f[img2]).astype(np.float32)
    # get camera para
    with h5py.File(os.path.join(folder_path,'cam_parameters.h5'),'r') as f:
        intrinsic_1 = np.array(f[img1]['intrinsic'])
        extrinsic_1 = np.linalg.inv(np.array(f[img1]['extrinsic']))
        intrinsic_2 = np.array(f[img2]['intrinsic'])
        extrinsic_2 = np.linalg.inv(np.array(f[img2]['extrinsic']))
    gt_rel_pose = np.matmul(np.linalg.inv(extrinsic_2),extrinsic_1)
    gt_R = gt_rel_pose[:3,:3]
    gt_t = gt_rel_pose[:3,3]
    # compute match
    corrs = match(desc1,desc2,kp1,kp2)
    # get depth
    if depth_extention.endswith('h5'):
        with h5py.File(os.path.join(folder_path,'depth',img1+depth_extention),'r') as f:
            img_1_depth = np.array(f['depth'])
        with h5py.File(os.path.join(folder_path,'depth',img2+depth_extention),'r') as f:
            img_2_depth = np.array(f['depth'])
    elif depth_extention.endswith('pfm'):
        img_1_dis_est = read_pfm(os.path.join(folder_path,'depth',img1+depth_extention))[0]
        img_2_dis_est = read_pfm(os.path.join(folder_path,'depth',img2+depth_extention))[0]

        img_1_depth = 1/(img_1_dis_est+1e-8)
        img_1_depth[img_1_depth<=0] = 0
        img_2_depth = 1/(img_2_dis_est+1e-8)
        img_2_depth[img_2_depth<=0] = 0
    # get image
    img_1_rgb = imageio.imread(os.path.join(folder_path,'img',img1+'.jpg'))
    img_2_rgb = imageio.imread(os.path.join(folder_path,'img',img2+'.jpg'))
    pt_1_xy_img = corrs[:,:2]
    pt_2_xy_img = corrs[:,2:]

    pt_1_gt_d = bilinear_interpolation(pt_1_xy_img,img_1_depth)
    pt_2_gt_d = bilinear_interpolation(pt_2_xy_img,img_1_depth)
    pt_1_gt_d_cor = get_lcoal_varience(pt_1_xy_img,img_2_depth)
    pt_2_gt_d_cor = get_lcoal_varience(pt_2_xy_img,img_2_depth)
    # compute pose
    

    if sac_type == 'ransac':
        R,t,rep = RANSAC(corrs,intrinsic_1,intrinsic_2)
    elif sac_type == 'custom_ransac':
        R,t,rep = custom_ransac(corrs,intrinsic_1,intrinsic_2)
    elif sac_type == 'depsac':
        R,t,rep = DEPSAC(corrs,pt_1_gt_d,pt_2_gt_d,intrinsic_1,intrinsic_2,d_1_var=pt_1_gt_d_cor,d_2_var=pt_2_gt_d_cor)

    err_q, err_t = evaluate_R_t(gt_R, gt_t, R, t)
    return err_q, err_t,rep
