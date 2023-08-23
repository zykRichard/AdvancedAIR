import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, box_utils

import json
import os
import math

from mmdet3d.core import (
    Box3DMode,
    LiDARInstance3DBoxes,
)

def test_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names

    logger.info('*************** EPOCH %s Generate results *****************' % epoch_id)

    model.eval()
    #progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    total_pred_objects = 0
    total_frame = len(dataloader)
    for i, batch_dict in enumerate(tqdm.tqdm(dataloader)):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)


        #convert 
        pred_dict = pred_dicts[0] #bs=1
        pred_boxes = pred_dict['pred_boxes'] #[x,y,z,l,w,h,r]
        total_pred_objects += len(pred_boxes)
        if len(pred_boxes) == 0:
            result_dict = {
                'boxes_3d': np.zeros((1, 8, 3)).tolist(),
                'labels_3d': np.zeros((1)).tolist(),
                'scores_3d': np.zeros((1)).tolist(),
                'ab_cost': 0,
            }
            print(batch_dict['frame_id'][0] + ' .  zero prediction!')
        else:     
            pred_labels = pred_dict['pred_labels'].cpu().tolist() #[1,2,3] - [Car, Ped, Cyc] .  [1,2,3,4] - ['Car', 'Van', 'Bus', 'Truck']
            class_mapping = {
                # 1: 2,
                # 2: 1,
                # 3: 3,
                1: 2,
                2: 2,
                3: 2,
                4: 2
            }
            mapped_labels = [class_mapping[label] for label in pred_labels]
            # import pdb
            # pdb.set_trace()
            lidar_box = LiDARInstance3DBoxes(pred_boxes)
            corner_3d = lidar_box.corners


            result_dict = {
                'boxes_3d': corner_3d.cpu().tolist(),
                'labels_3d': mapped_labels,
                'scores_3d': pred_dict['pred_scores'].cpu().tolist(),
                'ab_cost': 0,
            }

        json_filename = os.path.join(result_dir, batch_dict['frame_id'][0]  + ".json")
        with open(json_filename, 'w') as f:
            json.dump(result_dict, f)

    
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (total_frame, total_pred_objects / max(1, total_frame)))
    logger.info('****************Evaluation done.*****************')
    return 1


def test_one_epoch_vic(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    # dataset = dataloader.dataset
    # class_names = dataset.class_names

    logger.info('*************** EPOCH %s Generate results *****************' % epoch_id)

    model.eval()
    #progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    for VICFrame, label, filt in tqdm(dataloader):

        batch_dict = {}

        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)


        #convert 
        pred_dict = pred_dicts[0] #bs=1
        pred_boxes = pred_dict['pred_boxes'] #[x,y,z,l,w,h,r]
        if len(pred_boxes) == 0:
            result_dict = {
                'boxes_3d': np.zeros((1, 8, 3)).tolist(),
                'labels_3d': np.zeros((1)).tolist(),
                'scores_3d': np.zeros((1)).tolist(),
                'ab_cost': 0,
            }
            print(batch_dict['frame_id'][0] + ' .  zero prediction!')
        else:     
            pred_labels = pred_dict['pred_labels'].cpu().tolist() #[1,2,3] - [Car, Ped, Cyc]
            class_mapping = {
                1: 2,
                2: 2,
                3: 2,
                4: 2,
            }
            mapped_labels = [class_mapping[label] for label in pred_labels]
            # import pdb
            # pdb.set_trace()
            lidar_box = LiDARInstance3DBoxes(pred_boxes)
            corner_3d = lidar_box.corners


            result_dict = {
                'boxes_3d': corner_3d.cpu().tolist(),
                'labels_3d': mapped_labels,
                'scores_3d': pred_dict['pred_scores'].cpu().tolist(),
                'ab_cost': 0,
            }

        json_filename = os.path.join(result_dir, batch_dict['frame_id'][0]  + ".json")
        with open(json_filename, 'w') as f:
            json.dump(result_dict, f)

    logger.info('****************Evaluation done.*****************')
    return 1


if __name__ == '__main__':
    pass
