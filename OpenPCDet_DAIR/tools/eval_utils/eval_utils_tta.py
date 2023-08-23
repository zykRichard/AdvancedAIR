import pickle
import time
import os
import numpy as np
import torch
import tqdm
import json
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.utils.ensemble_boxes_wbf_3d import weighted_nms
from mmdet3d.core import (
    Box3DMode,
    LiDARInstance3DBoxes,
)

def random_world_flip(box_preds, params, reverse = False):
    if reverse:
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
    else:
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
    return box_preds

def random_world_rotation(box_preds, params, reverse = False):
    if reverse:
        noise_rotation = -params
    else:
        noise_rotation = params

    angle = torch.tensor([noise_rotation]).to(box_preds.device)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(1)
    ones = angle.new_ones(1)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(3, 3).float()
    box_preds[:, :3] = torch.matmul(box_preds[:, :3], rot_matrix)
    box_preds[:, 6] += noise_rotation
    return box_preds

def tta_wbf(preds, conf_type='avg', weights=None, iou_thr=0.45):

    box_list = []
    box_score_list = []
    box_label_list = []
    for i in range(len(preds)):
        pred = preds[i]

        if isinstance(pred['pred_boxes'], torch.Tensor):
            bbox = pred['pred_boxes'].cpu().numpy()
        else:
            bbox = pred['pred_boxes']

        if isinstance(pred['pred_scores'], torch.Tensor):
            box_score = pred['pred_scores'].cpu().numpy()
        else:
            box_score = pred['pred_scores']

        if isinstance(pred['pred_labels'], torch.Tensor):
            box_label = pred['pred_labels'].cpu().numpy()
        else:
            box_label = pred['pred_labels']

        box_list.append(bbox)
        box_score_list.append(box_score)
        box_label_list.append(box_label)

    boxes, scores, labels = weighted_nms(box_list, box_score_list, box_label_list, weights=weights,
                                         conf_type=conf_type, iou_thr=iou_thr, skip_box_thr=0.0)

    output = {
        'pred_boxes': torch.tensor(boxes),
        'pred_scores': torch.tensor(scores),
        'pred_labels': torch.tensor(labels).long(),
    }
    return [output]

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            # do tta
            if cfg.DATA_CONFIG.TEST_AUGMENTOR.ENABLE:
                tta_configs = cfg.DATA_CONFIG.TEST_AUGMENTOR.AUG_CONFIG_LIST
                aug_types = [ele['NAME'] for ele in tta_configs]
                jmax, kmax = 1, 1
                if 'test_world_flip' in aug_types:
                    flip_list = tta_configs[aug_types.index('test_world_flip')]['ALONG_AXIS_LIST']
                    jmax = len(flip_list)+1
                if 'test_world_rotation' in aug_types:
                    rot_list = tta_configs[aug_types.index('test_world_rotation')]['WORLD_ROT_ANGLE']
                    kmax = len(rot_list)

                for j in range(jmax):
                    for k in range(kmax):
                        pred_dict = pred_dicts[j*kmax+k]
                        if 'test_world_rotation' in aug_types:
                            rot = rot_list[k]
                            pred_dict['pred_boxes'] = random_world_rotation(pred_dict['pred_boxes'], rot, reverse=True)
                        if 'test_world_flip' in aug_types:
                            if j == 0:
                                continue
                            flip = flip_list[j-1]
                            pred_dict['pred_boxes'] = random_world_flip(pred_dict['pred_boxes'], flip, reverse=True)
                pred_dicts = tta_wbf(pred_dicts, iou_thr=0.45) #0.45
                # visualization(batch_dict['points'], pred_dicts[0]["pred_boxes"], batch_dict['frame_id'], 'debug_kitti', batch_dict['gt_boxes'][0][:,:-1])

        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def submit_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s Generate results *****************' % epoch_id)

    model.eval()

    total_pred_objects = 0
    total_frame = len(dataloader)
    for i, batch_dict in enumerate(tqdm.tqdm(dataloader)):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            # do tta
            if cfg.DATA_CONFIG.TEST_AUGMENTOR.ENABLE:
                tta_configs = cfg.DATA_CONFIG.TEST_AUGMENTOR.AUG_CONFIG_LIST
                aug_types = [ele['NAME'] for ele in tta_configs]
                jmax, kmax = 1, 1
                if 'test_world_flip' in aug_types:
                    flip_list = tta_configs[aug_types.index('test_world_flip')]['ALONG_AXIS_LIST']
                    jmax = len(flip_list)+1
                if 'test_world_rotation' in aug_types:
                    rot_list = tta_configs[aug_types.index('test_world_rotation')]['WORLD_ROT_ANGLE']
                    kmax = len(rot_list)

                for j in range(jmax):
                    for k in range(kmax):
                        pred_dict = pred_dicts[j*kmax+k]
                        if 'test_world_rotation' in aug_types:
                            rot = rot_list[k]
                            pred_dict['pred_boxes'] = random_world_rotation(pred_dict['pred_boxes'], rot, reverse=True)
                        if 'test_world_flip' in aug_types:
                            if j == 0:
                                continue
                            flip = flip_list[j-1]
                            pred_dict['pred_boxes'] = random_world_flip(pred_dict['pred_boxes'], flip, reverse=True)
                pred_dicts = tta_wbf(pred_dicts, iou_thr=0.45) #0.45
                # visualization(batch_dict['points'], pred_dicts[0]["pred_boxes"], batch_dict['frame_id'], 'debug_kitti', batch_dict['gt_boxes'][0][:,:-1])


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
                'pred_boxes':  pred_boxes.cpu().tolist() #for model ensemble mid-results
            }

        json_filename = os.path.join(result_dir, batch_dict['frame_id'][0]  + ".json")
        with open(json_filename, 'w') as f:
            json.dump(result_dict, f)

    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (total_frame, total_pred_objects / max(1, total_frame)))
    logger.info('****************Evaluation done.*****************')
    return 1

if __name__ == '__main__':
    pass
