import os.path as osp
import logging

logger = logging.getLogger(__name__)
import numpy as np

from dataset.dataset_utils import save_pkl, load_pkl, read_jpg
from v2x_utils import mkdir
from model_utils import init_model, inference_detector, inference_mono_3d_detector
from base_model import BaseModel
from mmdet3d_anymodel_anymodality_late import LateFusionVeh, LateFusionInf
import os, json

class SingleSide(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--config-path", type=str, default="")
        parser.add_argument("--model-path", type=str, default="")
        parser.add_argument("--sensor-type", type=str, default="lidar")
        parser.add_argument("--overwrite-cache", action="store_true")

    def __init__(self, args):
        super().__init__()
        self.model = None
        self.args = args
        mkdir(osp.join(args.output, "preds"))

    def pred(self, frame, pred_filter):
        id = frame.id["camera"]
        if self.args.dataset == "dair-v2x-i":
            input_path = osp.join(self.args.input, "infrastructure-side")
        elif self.args.dataset == "dair-v2x-v":
            input_path = osp.join(self.args.input, "vehicle-side")
        path = osp.join(self.args.output, "preds", id + ".pkl")
        if not osp.exists(path) or self.args.overwrite_cache:
            logger.debug("prediction not found, predicting...")
            if self.model is None:
                raise Exception

            if self.args.sensortype == "lidar":
                result, _ = inference_detector(self.model, frame.point_cloud(data_format="file"))

            elif self.args.sensortype == "camera":
                image = osp.join(input_path, frame["image_path"])
                # tmp = "../cache/tmps_i/" + frame.id + ".jpg"  # TODO
                # if not osp.exists(tmp):
                #     import mmcv

                # mmcv.tmp = mmcv.imwrite(image, tmp)
                annos = osp.join(input_path, "annos", id + ".json")

                result, _ = inference_mono_3d_detector(self.model, image, annos)

                # hard code by yuhb
                for ii in range(len(result[0]["labels_3d"])):
                    result[0]["labels_3d"][ii] = 2

            if len(result[0]["boxes_3d"].tensor) == 0:
                box = np.zeros((1, 8, 3))
                score = np.zeros(1)
                label = np.zeros(1)
            else:
                box = result[0]["boxes_3d"].corners.numpy()
                score = result[0]["scores_3d"].numpy()
                label = result[0]["labels_3d"].numpy()

            remain = []
            for i in range(box.shape[0]):
                if pred_filter(box[i]):
                    remain.append(i)
            if len(remain) >= 1:
                box = box[remain]
                score = score[remain]
                label = label[remain]
            else:
                box = np.zeros((1, 8, 3))
                score = np.zeros(1)
                label = np.zeros(1)
            pred_dict = {
                "boxes_3d": box,
                "scores_3d": score,
                "labels_3d": label,
            }
            save_pkl(pred_dict, path)
        else:
            pred_dict = load_pkl(path)
        return pred_dict

    def forward(self, frame, pred_filter):
        try:
            pred_dict = self.pred(frame, pred_filter)
        except Exception:
            logger.info("building model")
            self.model = init_model(
                self.args.config_path,
                self.args.model_path,
                device=self.args.device,
            )
            pred_dict = self.pred(frame, pred_filter)
        return pred_dict


class InfOnly(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--no-comp", action="store_true")
        parser.add_argument("--overwrite-cache", action="store_true")

    def __init__(self, args, pipe):
        super().__init__()
        self.model = LateFusionInf(args, pipe)
        self.pipe = pipe

    def forward(self, vic_frame, filt, offset, *args):
        self.model(
            vic_frame.infrastructure_frame(),
            vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar"),
            filt,
        )
        pred = np.array(self.pipe.receive("boxes"))
        return {
            "boxes_3d": pred,
            "labels_3d": np.array(self.pipe.receive("label")),
            "scores_3d": np.array(self.pipe.receive("score")),
        }

class OfflineInfOnly(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--no-comp", action="store_true")
        parser.add_argument("--overwrite-cache", action="store_true")
        parser.add_argument("--veh-result-path", type=str, default="")
        parser.add_argument("--inf-result-path", type=str, default="")
        parser.add_argument("--data-info-path", type=str, default="")

    def __init__(self, args, pipe):
        super().__init__()
        self.pipe = pipe
        self.inf_result_path = args.inf_result_path
        print(args.inf_result_path, 'model: OfflineInfOnly init()')
        if args.inf_result_path == "":
            self.inf_model = LateFusionInf(args, pipe)

        if args.inf_result_path:
            inf_results_path = os.listdir(args.inf_result_path)
            inf_results_path.sort()
            inf_resuts = {}
            for inf_result in inf_results_path:
                inf_result_path = osp.join(args.inf_result_path, inf_result)
                with open(inf_result_path, 'r') as f:
                    inf = json.load(f)
                inf_id = inf_result.split('.')[0]
                inf_resuts[inf_id] = inf
            self.inf_results = inf_resuts
            print('load {} inf results from {}'.format(len(self.inf_results ), args.inf_result_path))

            veh2inf = {}
            with open(args.data_info_path, 'r') as f:
                data_info_c = json.load(f)
            for data in data_info_c:
                veh_id = os.path.basename(data['vehicle_pointcloud_path']).split('.')[0]
                inf_id = os.path.basename(data['infrastructure_pointcloud_path']).split('.')[0]
                veh2inf[veh_id] = inf_id
            self.veh2inf = veh2inf


    def forward(self, vic_frame, filt, offset, *args):
        veh_id = vic_frame.vehicle_frame().id["lidar"]
        if self.inf_result_path == "":
            id_inf = self.inf_model(
                vic_frame.infrastructure_frame(),
                vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar"),
                filt,
            )
            return {
                "boxes_3d": np.array(self.pipe.receive("boxes")),
                "labels_3d": np.array(self.pipe.receive("label")),
                "scores_3d": np.array(self.pipe.receive("score")),
            }
        else:
            # loading offline inf_results
            inf_id = self.veh2inf[veh_id]
            trans = vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
            pred_dict_inf = self.inf_results[inf_id]

            self.pipe.send("boxes", pred_dict_inf["boxes_3d"])
            self.pipe.send("score", pred_dict_inf["scores_3d"])
            self.pipe.send("label", pred_dict_inf["labels_3d"])

            return {
                "boxes_3d": np.array(trans(np.array(pred_dict_inf["boxes_3d"]))),
                "labels_3d": np.array(pred_dict_inf["labels_3d"]),
                "scores_3d": np.array(pred_dict_inf["scores_3d"]),
            }

class VehOnly(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--overwrite-cache", action="store_true")

    def __init__(self, args, pipe):
        super().__init__()
        self.model = LateFusionVeh(args)
        self.pipe = pipe

    def forward(self, vic_frame, filt, *args):
        pred = self.model(vic_frame.vehicle_frame(), None, filt)[0]
        return {
            "boxes_3d": np.array(pred["boxes_3d"]),
            "labels_3d": np.array(pred["labels_3d"]),
            "scores_3d": np.array(pred["scores_3d"]),
        }

class OfflineVehOnly(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--overwrite-cache", action="store_true")
        parser.add_argument("--veh-result-path", type=str, default="")
        parser.add_argument("--inf-result-path", type=str, default="")
        parser.add_argument("--data-info-path", type=str, default="")

    def __init__(self, args, pipe):
        super().__init__()
        #self.model = LateFusionVeh(args)
        self.pipe = pipe
        self.veh_result_path = args.veh_result_path
        if args.veh_result_path == "":
            self.veh_model = LateFusionVeh(args)

        if args.veh_result_path:
            veh_results_path = os.listdir(args.veh_result_path)
            veh_results_path.sort()
            veh_resuts = {}
            for veh_result in veh_results_path:
                veh_result_path = osp.join(args.veh_result_path, veh_result)
                with open(veh_result_path, 'r') as f:
                    veh = json.load(f)
                veh_id = veh_result.split('.')[0]
                veh_resuts[veh_id] = veh
            self.veh_results = veh_resuts
            print('load {} veh results from {}'.format(len(self.veh_results ), args.veh_result_path))

    def forward(self, vic_frame, filt, *args):

        if self.veh_result_path == "":
            pred_dict, veh_id = self.veh_model(vic_frame.vehicle_frame(), None, filt)
        else:
            #loading offline inf_results
            veh_id = vic_frame.vehicle_frame().id["lidar"]
            pred_dict = self.veh_results[veh_id]

        return {
            "boxes_3d": np.array(pred_dict["boxes_3d"]),
            "labels_3d": np.array(pred_dict["labels_3d"]),
            "scores_3d": np.array(pred_dict["scores_3d"]),
        }

class InfGTOnly(BaseModel):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--inf-config-path", type=str, default="")
        parser.add_argument("--inf-model-path", type=str, default="")
        parser.add_argument("--veh-config-path", type=str, default="")
        parser.add_argument("--veh-model-path", type=str, default="")
        parser.add_argument("--no-comp", action="store_true")
        parser.add_argument("--overwrite-cache", action="store_true")
        parser.add_argument("--veh-result-path", type=str, default="")
        parser.add_argument("--inf-result-path", type=str, default="")
        parser.add_argument("--data-info-path", type=str, default="")

    def __init__(self, args, pipe):
        super().__init__()
        self.model = LateFusionInf(args, pipe)
        self.pipe = pipe
        print('model: InfGTonly init()')

    def forward(self, vic_frame, filt, label, offset, *args):

        # self.model(
        #     vic_frame.infrastructure_frame(),
        #     vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar"),
        #     filt,
        # )
        # model_pred = {
        #     "boxes_3d": self.pipe.receive("boxes"),
        #     "labels_3d": self.pipe.receive("label"),
        #     "scores_3d": self.pipe.receive("score"),
        # }
        #loading inf gt
        trans = vic_frame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
        pred_dict_inf = label.pop('label_i')

        # self.pipe.send("boxes", pred_dict_inf["boxes_3d"])
        # self.pipe.send("score", pred_dict_inf["scores_3d"])
        # self.pipe.send("label", pred_dict_inf["labels_3d"])

        trans_box = trans(pred_dict_inf["boxes_3d"])
        label_i = {
            "boxes_3d": trans_box,
            "labels_3d": pred_dict_inf["labels_3d"],
            "scores_3d": pred_dict_inf["scores_3d"],
        }
        
        # label_i_trans = {
        #     "boxes_3d": trans_box.tolist(),
        #     "labels_3d": pred_dict_inf["labels_3d"].tolist(),
        #     "scores_3d": pred_dict_inf["scores_3d"].tolist(),
        # }

        # label_c = {
        #     "boxes_3d": label["boxes_3d"].tolist(),
        #     "labels_3d": label["labels_3d"].tolist(),
        #     "scores_3d": label["scores_3d"].tolist(),   
        # }

        # save_pth = './temp_result'
        # os.makedirs(save_pth, exist_ok=1)
        # with open(os.path.join(save_pth, 'modelpred.json'), 'w') as f:
        #     json.dump(model_pred, f)
        # with open(os.path.join(save_pth, 'labeli.json'), 'w') as f:
        #     json.dump(label_i, f)
        # with open(os.path.join(save_pth, 'labelitrans.json'), 'w') as f:
        #     json.dump(label_i_trans, f)
        # with open(os.path.join(save_pth, 'labelc.json'), 'w') as f:
        #     json.dump(label_c, f)


        return label_i
