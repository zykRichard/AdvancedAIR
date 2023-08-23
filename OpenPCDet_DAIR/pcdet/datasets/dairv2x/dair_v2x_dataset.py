import os.path as osp
from functools import cmp_to_key
import logging

logger = logging.getLogger(__name__)

from base_dataset import DAIRV2XDataset, get_annos, build_path_to_info
from .dataset_utils import load_json, InfFrame, VehFrame, VICFrame, Label
from .dataset_utils import Filter, RectFilter, id_cmp, id_to_str, get_trans, box_translation
import pdb
from ..dataset import DatasetTemplate

class VICDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.path = root_path
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        sensortype = "lidar"
        self.inf_path2info = build_path_to_info(
            "infrastructure-side",
            load_json(osp.join(root_path, "infrastructure-side/data_info.json")),
            sensortype,
        )
        self.veh_path2info = build_path_to_info(
            "vehicle-side",
            load_json(osp.join(root_path, "vehicle-side/data_info.json")),
            sensortype,
        )

        frame_pairs = load_json(osp.join(root_path, "cooperative/data_info.json"))
        split_path = osp.join(root_path, 'data/split_datas/cooperative-split-data.json')
        frame_pairs = self.get_split(split_path, split, frame_pairs)

        extended_range = self.point_cloud_range

        self.data = []
        self.inf_frames = {}
        self.veh_frames = {}

        for elem in frame_pairs:
            if sensortype == "lidar":
                inf_frame = self.inf_path2info[elem["infrastructure_pointcloud_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_pointcloud_path"]]
            elif sensortype == "camera":
                inf_frame = self.inf_path2info[elem["infrastructure_image_path"]]
                veh_frame = self.veh_path2info[elem["vehicle_image_path"]]
                get_annos(root_path, "infrastructure-side", inf_frame, "camera")
                get_annos(root_path, "vehicle-side", veh_frame, "camera")

            inf_frame = InfFrame(root_path + "/infrastructure-side/", inf_frame)
            veh_frame = VehFrame(root_path + "/vehicle-side/", veh_frame)
            if not inf_frame["batch_id"] in self.inf_frames:
                self.inf_frames[inf_frame["batch_id"]] = [inf_frame]
            else:
                self.inf_frames[inf_frame["batch_id"]].append(inf_frame)
            if not veh_frame["batch_id"] in self.veh_frames:
                self.veh_frames[veh_frame["batch_id"]] = [veh_frame]
            else:
                self.veh_frames[veh_frame["batch_id"]].append(veh_frame)
            vic_frame = VICFrame(root_path, elem, veh_frame, inf_frame, 0)

            # filter in world coordinate
            if extended_range is not None:
                trans = vic_frame.transform(from_coord="Vehicle_lidar", to_coord="World")
                filt_world = RectFilter(trans(extended_range)[0])

            trans_1 = vic_frame.transform("World", "Vehicle_lidar")
            if osp.isfile(osp.join(root_path, elem["cooperative_label_path"])):
                label_v = Label(osp.join(root_path, elem["cooperative_label_path"]), filt_world)
                label_v["boxes_3d"] = trans_1(label_v["boxes_3d"])
            else:
                label_v = None
            filt = RectFilter(extended_range[0])
            tup = (
                vic_frame,
                label_v,
                filt,
            )
            self.data.append(tup)

    def query_veh_segment(self, frame, sensortype="lidar", previous_only=False):
        segment = self.veh_frames[frame.batch_id]
        return [f for f in segment if f.id[sensortype] < frame.id[sensortype] or not previous_only]

    def query_inf_segment(self, frame, sensortype="lidar", previous_only=False):
        segment = self.inf_frames[frame.batch_id]
        return [f for f in segment if f.id[sensortype] < frame.id[sensortype] or not previous_only]

    def get_split(self, split_path, split, frame_pairs):
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test", "test_A"]:
            split_data = split_data["cooperative_split"][split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for frame_pair in frame_pairs:
            veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
            if veh_frame_idx in split_data:
                frame_pairs_split.append(frame_pair)
        return frame_pairs_split

    def __getitem__(self, index):
        raise NotImplementedError



if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np

    input = "../data/cooperative-vehicle-infrastructure/"
    split = "val"
    sensortype = "camera"
    box_range = np.array([-10, -49.68, -3, 79.12, 49.68, 1])
    indexs = [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
    extended_range = np.array([[box_range[index] for index in indexs]])
    dataset = VICDataset(input, split, sensortype, extended_range=extended_range)

    for VICFrame_data, label, filt in tqdm(dataset):
        veh_image_path = VICFrame_data.vehicle_frame()["image_path"][-10:-4]
        inf_image_path = VICFrame_data.infrastructure_frame()["image_path"][-10:-4]
        print(veh_image_path, inf_image_path)
