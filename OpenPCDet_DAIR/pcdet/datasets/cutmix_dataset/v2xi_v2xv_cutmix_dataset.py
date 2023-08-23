import os.path
import numpy as np
import pickle
import copy
from pathlib import Path

from pcdet.datasets import CutMixDatasetTemplate
from pcdet.utils import box_utils
from ..kitti import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, calibration_kitti

class V2XCutMixDataset(CutMixDatasetTemplate):
    def __init__(self, dataset_cfg=None, training=True, dataset_names=None, logger=None):
        super().__init__(dataset_cfg, training, dataset_names, logger)

        self.v2xi_infos = []
        self.v2xv_infos = []
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # for v2xi
        self.root_split_path_source = self.root_path_source / ('training' if self.split != 'test' else 'testing')
        self.include_v2xi_data(self.mode)
        # for v2xv
        self.root_split_path_target = self.root_path_target / ('training' if self.split != 'test' else 'testing')
        self.include_v2xv_data(self.mode)

    def include_v2xi_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading V2XIDataset dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg['V2XIDataset'].INFO_PATH[mode]:
            info_path = self.root_path_source / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.v2xi_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for V2XI dataset: %d' % (len(kitti_infos)))

    def include_v2xv_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading V2XVDataset dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg['V2XVDataset'].INFO_PATH[mode]:
            info_path = self.root_path_target / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.v2xv_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for V2XV dataset: %d' % (len(kitti_infos)))


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return (len(self.v2xi_infos) + len(self.v2xv_infos)) * self.total_epochs
        return len(self.v2xi_infos) + len(self.v2xv_infos)

    def __getitem__(self, index):

        prob = np.random.random(1)
        if prob < self.dataset_cfg.CUTMIX_PROB:
            v2xi_infos = copy.deepcopy(self.v2xi_infos[index % len(self.v2xi_infos)])
            v2xv_infos = copy.deepcopy(self.v2xv_infos[index % len(self.v2xv_infos)])

            # for v2xi
            sample_idx = v2xi_infos['point_cloud']['lidar_idx']
            img_shape = v2xi_infos['image']['image_shape']
            calib = self.get_calib(sample_idx, src=1)
            get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            v2xi_input_dict = {
                'frame_id': sample_idx,
                'calib': calib,
            }

            if 'annos' in v2xi_infos:
                annos = v2xi_infos['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
                    
                v2xi_input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list:
                    v2xi_input_dict['gt_boxes2d'] = annos["bbox"]

            if "points" in get_item_list:
                points = self.get_lidar(sample_idx, src=1)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

                if self.dataset_cfg.get('IMAGE_MASK', None):

                    # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % sample_idx)
                    # mask_file = lidar_file.replace('velodyne', 'mask2d')
                    # assert mask_file.exists()
                    # mask2d = np.fromfile(str(mask_file)).reshape(-1,2)[:,0].astype('uint8')
                    N, D = points.shape
                    mask2d = np.random.randint(0,3,N)
                    mask = np.eye(3, dtype='uint8')[mask2d]
                    points = np.hstack([points, mask])

                v2xi_input_dict['points'] = points
            v2xi_input_dict['calib'] = calib

            # for v2xv
            sample_idx = v2xv_infos['point_cloud']['lidar_idx']
            img_shape = v2xv_infos['image']['image_shape']
            calib = self.get_calib(sample_idx, src=0)
            get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            v2xv_input_dict = {
                'frame_id': sample_idx,
                'calib': calib,
            }

            if 'annos' in v2xv_infos:
                annos = v2xv_infos['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
                    
                v2xv_input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list:
                    v2xv_input_dict['gt_boxes2d'] = annos["bbox"]


            if "points" in get_item_list:
                points = self.get_lidar(sample_idx, src=0)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

                if self.dataset_cfg.get('IMAGE_MASK', None):

                    # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % sample_idx)
                    # mask_file = lidar_file.replace('velodyne', 'mask2d')
                    # assert mask_file.exists()
                    # mask2d = np.fromfile(str(mask_file)).reshape(-1,2)[:,0].astype('uint8')
                    N, D = points.shape
                    mask2d = np.random.randint(0,3,N)
                    mask = np.eye(3, dtype='uint8')[mask2d]
                    points = np.hstack([points, mask])

                v2xv_input_dict['points'] = points
            v2xv_input_dict['calib'] = calib

            #data_dict = self.prepare_ori_data(v2xi_input_dict, source=True)
            data_dict = self.prepare_data(v2xi_input_dict, v2xv_input_dict)


        else:
            if index < len(self.v2xi_infos):
                v2xi_infos = copy.deepcopy(self.v2xi_infos[index])
                sample_idx = v2xi_infos['point_cloud']['lidar_idx']
                img_shape = v2xi_infos['image']['image_shape']
                calib = self.get_calib(sample_idx, src=1)
                get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
                v2xi_input_dict = {
                    'frame_id': sample_idx,
                    'calib': calib,
                }

                if 'annos' in v2xi_infos:
                    annos = v2xi_infos['annos']
                    annos = common_utils.drop_info_with_name(annos, name='DontCare')
                    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                    gt_names = annos['name']
                    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
                        
                    v2xi_input_dict.update({
                        'gt_names': gt_names,
                        'gt_boxes': gt_boxes_lidar
                    })
                    if "gt_boxes2d" in get_item_list:
                        v2xi_input_dict['gt_boxes2d'] = annos["bbox"]

                if "points" in get_item_list:
                    points = self.get_lidar(sample_idx, src=1)
                    if self.dataset_cfg.FOV_POINTS_ONLY:
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]

                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

                    if self.dataset_cfg.get('IMAGE_MASK', None):

                        # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % sample_idx)
                        # mask_file = lidar_file.replace('velodyne', 'mask2d')
                        # assert mask_file.exists()
                        # mask2d = np.fromfile(str(mask_file)).reshape(-1,2)[:,0].astype('uint8')
                        N, D = points.shape
                        mask2d = np.random.randint(0,3,N)
                        mask = np.eye(3, dtype='uint8')[mask2d]
                        points = np.hstack([points, mask])

                    v2xi_input_dict['points'] = points
                
                v2xi_input_dict['calib'] = calib
                data_dict = self.prepare_ori_data(v2xi_input_dict, source=True)

            else:
                v2xv_infos = copy.deepcopy(self.v2xv_infos[index - len(self.v2xi_infos)])
                sample_idx = v2xv_infos['point_cloud']['lidar_idx']
                img_shape = v2xv_infos['image']['image_shape']
                calib = self.get_calib(sample_idx, src=0)
                get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
                v2xv_input_dict = {
                    'frame_id': sample_idx,
                    'calib': calib,
                }

                if 'annos' in v2xv_infos:
                    annos = v2xv_infos['annos']
                    annos = common_utils.drop_info_with_name(annos, name='DontCare')
                    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                    gt_names = annos['name']
                    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
                        
                    v2xv_input_dict.update({
                        'gt_names': gt_names,
                        'gt_boxes': gt_boxes_lidar
                    })
                    if "gt_boxes2d" in get_item_list:
                        v2xv_input_dict['gt_boxes2d'] = annos["bbox"]

                if "points" in get_item_list:
                    points = self.get_lidar(sample_idx, src=0)
                    if self.dataset_cfg.FOV_POINTS_ONLY:
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]

                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

                    if self.dataset_cfg.get('IMAGE_MASK', None):

                        # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % sample_idx)
                        # mask_file = lidar_file.replace('velodyne', 'mask2d')
                        # assert mask_file.exists()
                        # mask2d = np.fromfile(str(mask_file)).reshape(-1,2)[:,0].astype('uint8')
                        N, D = points.shape
                        mask2d = np.random.randint(0,3,N)
                        mask = np.eye(3, dtype='uint8')[mask2d]
                        points = np.hstack([points, mask])

                    v2xv_input_dict['points'] = points
                v2xv_input_dict['calib'] = calib
                data_dict = self.prepare_ori_data(v2xv_input_dict, source=False)

        return data_dict

    def get_lidar(self, idx, src):
        if src:
            lidar_file = self.root_split_path_source / 'velodyne' / ('%s.bin' % idx)
        else:
            lidar_file = self.root_split_path_target / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


    def get_calib(self, idx, src):
        if src:
            calib_file = self.root_split_path_source / 'calib' / ('%s.txt' % idx)
        else:
            calib_file = self.root_split_path_target / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag
