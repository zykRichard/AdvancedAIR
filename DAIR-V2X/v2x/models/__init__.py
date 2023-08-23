import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from detection_models import *

SUPPROTED_MODELS = {
    "single_side": SingleSide,
    "late_fusion": LateFusion,
    "early_fusion": EarlyFusion,
    "veh_only": VehOnly,
    "inf_only": InfOnly,
    "offline_fusion": OfflineLateFusion,
    "offline_inf_only": OfflineInfOnly,
    "offline_veh_only": OfflineVehOnly,
    "offline_inf_gt": OfflineLateFusionInfGT,
    "infgt_only": InfGTOnly
}
