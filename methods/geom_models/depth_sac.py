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

def depth_sac(corrs, d_1, d_2, K_1, K_2, max_iter = 500000, epi_thold = 0.3, rep_thold=20, d_1_var=None, d_2_var=None,gt_R=None,gt_t=None,img1=None,img2=None):
    np.random.seed(0)
    min_sample = 8
    num_sample = corrs.shape[0]
    idx = np.arange(num_sample)
    max_inlier = 0
    valid_counter = 0
    update_counter = 0
    best_rep_err =100000
    best_pose = np.eye(4)
    best_E = None
    best_mask = None
    # pt_1_xy_img_homo = np.concatenate((corrs[:,:2],np.ones((corrs.shape[0],1))),axis=-1)
    # pt_2_xy_img_homo = np.concatenate((corrs[:,2:],np.ones((corrs.shape[0],1))),axis=-1)
    for _iter in range (max_iter):
    # for _ in range (10):
        # sample a subset
        sub_idx = np.random.choice(idx,min_sample,replace=False)
        # print(sub_idx)
        sub_corrs = corrs[sub_idx]

        # compute model using 8 points
        F, _ = cv2.findFundamentalMat(sub_corrs[:, :2], sub_corrs[:, 2:], cv2.FM_8POINT)

        if not F is None:
            
            # get inlier using epipolar distance and depth local varience
            dist = get_epi_dist(corrs[:,:2],corrs[:,2:],F)
            if not d_1_var is None:
                d_var_mask = np.logical_and(d_1_var<0.05,d_2_var<0.05)
                epi_mask = dist<epi_thold
                inlier_mask = np.logical_and(epi_mask,d_var_mask)
            else:    
                inlier_mask = dist<epi_thold
            inlier_corrs = corrs[inlier_mask]
            sub_depth_1 = d_1[inlier_mask]
            sub_depth_2 = d_2[inlier_mask]
            num_inlier = np.sum(inlier_mask)
            ####### DEBUG check if ransac with only F works #######
            # if max_inlier < num_inlier:
            #     update_counter = update_counter + 1
            #     max_inlier = num_inlier
            #     print(num_inlier)
            #     E = K_2.T @ F @ K_1
            #     # get normalized points (camera coor)
            #     sub_pt_norm_1 = np.matmul(np.linalg.inv(K_1),np.concatenate([inlier_corrs[:,:2], np.ones((inlier_corrs.shape[0],1))],axis=-1).T).T
            #     sub_pt_norm_2 = np.matmul(np.linalg.inv(K_2),np.concatenate([inlier_corrs[:,2:], np.ones((inlier_corrs.shape[0],1))],axis=-1).T).T
            #     # get pose from E
            #     points, R, t, mask = cv2.recoverPose(E, sub_pt_norm_1[:,:2], sub_pt_norm_2[:,:2])
            #     best_pose = np.eye(4)
            #     best_pose[:3,:3] = R
            #     best_pose[:3,[3]] = t
            #     best_lin_mapping = None
            #     best_mask = inlier_mask
            #     best_F = F

            #     best_rep_err = None
            #     best_point_rep_err = None
            #     best_point_eu_err = None
            #     print(np.max(dist[inlier_mask]))
            # continue
            #######################################################

            # # skip if too little inliers
            if np.sum(inlier_mask)<min_sample+3:
                continue
            valid_counter = valid_counter +1
            # use provied intrinsic get E and normalize points
            E = K_2.T @ F @ K_1
            # get normalized points (camera coor)
            sub_pt_norm_1 = np.matmul(np.linalg.inv(K_1),np.concatenate([inlier_corrs[:,:2], np.ones((inlier_corrs.shape[0],1))],axis=-1).T).T
            sub_pt_norm_2 = np.matmul(np.linalg.inv(K_2),np.concatenate([inlier_corrs[:,2:], np.ones((inlier_corrs.shape[0],1))],axis=-1).T).T
            # get pose from E
            points, R, t, mask = cv2.recoverPose(E, sub_pt_norm_1[:,:2], sub_pt_norm_2[:,:2])
            pose_1_2 = np.eye(4)
            pose_1_2[:3,:3] = R
            pose_1_2[:3,[3]] = t

            # solve linear system to get mapping of d1 and d2
            # R[x1,y1,1].T*(a*d1+b)+t = [x2,y2,1].T*(m*d2+n)
            # [R[x1,y1,1].T*d_1,R[x1,y1,1].T,-[x_2,y_2,1].T*d_2,[x_2,y_2,1].T]*[a,b,m,n].T = -t
            # A*[a,b].T = B
            # [a,b].T = A'B
            A = np.concatenate(((np.matmul(R,sub_pt_norm_1.T)*sub_depth_1).T.reshape(-1,1),
                                 np.matmul(R,sub_pt_norm_1.T).T.reshape(-1,1),
                                 -(sub_pt_norm_2*sub_depth_2[...,np.newaxis]).reshape(-1,1),
                                 -(sub_pt_norm_2).reshape(-1,1)
                                 ),axis=-1)

            B = -np.repeat(t.T,sub_pt_norm_1.shape[0],axis=0).reshape(-1,1)
            lin_mapping = np.matmul(np.linalg.pinv(A),B)

            # A = np.concatenate(((np.matmul(R,sub_pt_norm_1.T)*sub_depth_1).T.reshape(-1,1),
            #                      (np.matmul(R,sub_pt_norm_1.T).T-(sub_pt_norm_2)).reshape(-1,1),
            #                      -(sub_pt_norm_2*sub_depth_2[...,np.newaxis]).reshape(-1,1)
            #                      ),axis=-1)

            # B = -np.repeat(t.T,sub_pt_norm_1.shape[0],axis=0).reshape(-1,1)
            # lin_mapping = np.matmul(np.linalg.pinv(A),B)

            # A = np.concatenate((
            #                      -(sub_pt_norm_2*sub_depth_2[...,np.newaxis]).reshape(-1,1),
            #                      -(sub_pt_norm_2).reshape(-1,1)
            #                      ),axis=-1)

            # B = -((np.matmul(R,sub_pt_norm_1.T)*sub_depth_1).T+t.T).reshape(-1,1)
            # lin_mapping = np.matmul(np.linalg.pinv(A),B)

            # A = np.concatenate(((np.matmul(R,sub_pt_norm_1.T)*sub_depth_1).T.reshape(-1,1),np.matmul(R,sub_pt_norm_1.T).T.reshape(-1,1)),axis=-1)
            # B = (sub_pt_norm_2*sub_depth_2[...,np.newaxis] - t.T).reshape(-1,1)
            # lin_mapping = np.matmul(np.linalg.pinv(A),B)

            # # A = np.concatenate(((np.matmul(R,pt_norm_1.T)*depth_1).T.reshape(-1,1),
            # #                      np.matmul(R,pt_norm_1.T).T.reshape(-1,1),
            # #                      -(pt_norm_2*depth_2[...,np.newaxis]).reshape(-1,1),
            # #                      -(pt_norm_2).reshape(-1,1)
            # #                      ),axis=-1)

            # # B = -np.repeat(t[np.newaxis,...],corrs.shape[0],axis=0).reshape(-1,1)
            # # lin_mapping = np.matmul(np.linalg.pinv(A),B)

            # calculate reprojection error for all points
            sub_depth_1_mapped = sub_depth_1*lin_mapping[0,0] + lin_mapping[1,0]
            sub_depth_2_mapped = sub_depth_2*lin_mapping[2,0] + lin_mapping[3,0]

            # sub_depth_1_mapped = sub_depth_1
            # sub_depth_2_mapped = sub_depth_2
            rep_err = get_rep_error(inlier_corrs[:,:2],sub_depth_1_mapped[...,np.newaxis],inlier_corrs[:,2:],sub_depth_2_mapped[...,np.newaxis],K_1,K_2,pose_1_2)

            num_rep_inlier = np.sum(rep_err<rep_thold)

            # print(np.mean(rep_err))
            # if max_inlier < num_rep_inlier:
            if np.mean(rep_err)<best_rep_err:
                update_counter = update_counter + 1
                max_inlier = num_rep_inlier
                best_mask = inlier_mask
                best_pose = pose_1_2
                best_lin_mapping = lin_mapping
                best_rep_err = np.mean(rep_err)
                best_point_rep_err = rep_err
                best_F = F
                best_E = E

    return best_E,best_mask


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
