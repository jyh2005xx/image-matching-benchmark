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

from utils.stereo_helper import normalize_keypoints
from methods.geom_models.common import _fail, _preprocess_kp_depth
from methods.geom_models.depth_sac import depth_sac

def _depth_estimate_E_with_intrinsics(cfg, matches, kps1, kps2, deps1, deps2,
                                    dep_vars1, dep_vars2, calib1, calib2):
    '''Estimate the Essential matrix from correspondences with depth. 
    Assumes known intrinsics.
    '''

    cur_key = 'config_{}_{}'.format(cfg.dataset, cfg.task)
    geom = cfg.method_dict[cur_key]['geom']
    if geom['method'].lower() != 'custom-depth-sac':
        raise ValueError('Unknown method [{}] to estimate E'.format(
            geom['method'].lower()
        ))

    is_valid, matches, kp1, kp2, dep1, dep2, dep_var1, dep_var2 = _preprocess_kp_depth(matches, kps1, kps2, 
                                                       deps1, deps2, 
                                                       dep_vars1, dep_vars2, 
                                                       5)
    if not is_valid:
        return _fail()

    corrs = np.concatenate([kp1, kp2], axis=-1)
    E, mask_E = depth_sac(corrs, dep1, dep2, dep_var1,dep_var2, calib1['K'], calib2['K'], 
                                    geom['max_iter'], geom['threshold'],
                                    geom['rep_threshold'])

    if E is None:
        return _fail()
    
    mask_E = mask_E.astype(bool).flatten()

    indices = matches[:, mask_E]
    return E, indices

def estimate_essential(cfg,
                       matches,
                       kps1,
                       kps2,
                       calib1,
                       calib2,
                       scales1=None,
                       scales2=None,
                       ori1=None,
                       ori2=None,
                       descs1=None,
                       descs2=None,
                       dep1 = None,
                       dep2 = None,
                       dep_conf1 = None,
                       dep_conf2 = None):
    '''Estimate the Essential matrix from correspondences.

    Common entry point for all methods. Currently uses OpenCV to estimate E,
    with or without assuming known intrinsics.
    '''
    # Estimate E with 8 points + depth verification(assumes known intrinsics)
    return _depth_estimate_E_with_intrinsics(cfg, matches, kps1, kps2,
                                               dep1, dep2, 
                                               dep_conf1, dep_conf2,
                                               calib1, calib2)

