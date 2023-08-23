import pickle
import time
import datetime
import argparse
import numpy as np
import torch
import tqdm
import os
from pathlib import Path
import json
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils.ensemble_boxes_wbf_3d import weighted_nms
import glob

from pcdet.utils.ensemble_boxes_wbf_3d import weighted_nms

from mmdet3d.core import (
    Box3DMode,
    LiDARInstance3DBoxes,
)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='', help='specify the config for training')

    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument("--result_a", type=str, default='/')
    parser.add_argument("--result_b", type=str, default='')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg



def wbf(preds, conf_type='avg', weights=None, iou_thr=0.45):
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Motorcycle', 'Truck']
    detections = []
    gt_boxes = None
    FileNum = len(preds)
    total_detection_num = 0
    for i in tqdm.tqdm(range(len(preds[0]))):
        token = preds[0][i]['frame_id']
        box_list = []
        box_score_list = []
        box_label_list = []
        for iter in range(FileNum):
            pred = preds[iter]
            if pred[i]['name'].shape[0] == 0:
                pred[i]['name'] = np.array(['Car'])
                pred[i]['boxes_lidar'] = np.zeros((1, 7))
                pred[i]['score'] = np.zeros(1)

            if (isinstance(pred[i]['boxes_lidar'], torch.Tensor)):
                bbox = pred[i]['boxes_lidar'].cpu().numpy()
            else:
                bbox = pred[i]['boxes_lidar']

            if (isinstance(pred[i]['score'], torch.Tensor)):
                box_score = pred[i]['score'].cpu().numpy()
            else:
                box_score = pred[i]['score']
            

            box_label = np.stack([CLASS_NAMES.index(ele) for ele in pred[i]['name']])

            box_list.append(bbox)
            box_score_list.append(box_score)
            box_label_list.append(box_label)

        boxes, scores, labels = weighted_nms(box_list, box_score_list, box_label_list, weights=weights,
                                             conf_type=conf_type, iou_thr=iou_thr, skip_box_thr=0.0)

        labels = np.array([CLASS_NAMES[ele] for ele in labels.astype('int64')])
        output = {
            'boxes_3d': boxes,
            'score': scores,
            'name': labels,
            'frame_id': token
        }
        detections.append(output)
        total_detection_num += len(output['name'])
    print('total ensemble number of prediction: {}'.format(total_detection_num))
    return detections


def tta_wbf(preds, conf_type='avg', weights=None, iou_thr=0.45):

    box_list = []
    box_score_list = []
    box_label_list = []
    for i in range(len(preds)):
        pred = preds[i]

        if pred['labels_3d'][0] == 0:
            pred['pred_boxes'] = np.zeros((1, 7)).tolist()
            print('zero_prediction!')


        bbox = pred['pred_boxes']
        box_score = pred['scores_3d']
        box_label = pred['labels_3d']

        box_list.append(bbox)
        box_score_list.append(box_score)
        box_label_list.append(box_label)

    boxes, scores, labels = weighted_nms(box_list, box_score_list, box_label_list, weights=weights,
                                         conf_type=conf_type, iou_thr=iou_thr, skip_box_thr=0.0)

    output = {
        'pred_boxes': boxes,
        'pred_scores': scores,
        'pred_labels': labels,
    }
    return [output]


if __name__ == '__main__':
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'model_ensemble' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir)


    if args.result_a:
        pred_a_list = os.listdir(args.result_a)
        print('load {} results form {}'.format(len(pred_a_list), args.result_a))
    if args.result_b:
        pred_b_list = os.listdir(args.result_b)
        print('load {} results form {}'.format(len(pred_b_list), args.result_b))
    assert len(pred_a_list) == len(pred_b_list)

    weights = [1] * 2
    
    cnt_pred_a = 0
    cnt_pred_b = 0
    cnt_pred_ensemble = 0
    for pred in tqdm.tqdm(pred_a_list):
        pred_dicts = []

        pred_a_path = os.path.join(args.result_a, pred)
        pred_b_path = os.path.join(args.result_b, pred)

        with open(pred_a_path, 'r') as f:
            preda = json.load(f)
            pred_dicts.append(preda)
            cnt_pred_a += len(preda['labels_3d'])
        with open(pred_b_path, 'r') as f:
            predb = json.load(f)
            pred_dicts.append(predb)
            cnt_pred_b += len(predb['labels_3d'])

        pred_dicts = tta_wbf(pred_dicts, iou_thr=0.45)

        pred_dict = pred_dicts[0] #bs=1
        pred_boxes = pred_dict['pred_boxes'] #[x,y,z,l,w,h,r]
        cnt_pred_ensemble += len(pred_boxes)
        lidar_box = LiDARInstance3DBoxes(pred_boxes)
        corner_3d = lidar_box.corners    

        result_dict = {
            'boxes_3d': corner_3d.cpu().tolist(),
            'labels_3d': pred_dict['pred_labels'].tolist(),
            'scores_3d': pred_dict['pred_scores'].tolist(),
            'ab_cost': 0,
        }

        json_filename = os.path.join(output_dir, pred)
        with open(json_filename, 'w') as f:
            json.dump(result_dict, f)
    
    print('{}, {}, {}'.format(cnt_pred_a/len(pred_b_list), cnt_pred_b/len(pred_b_list), cnt_pred_ensemble/len(pred_b_list)))