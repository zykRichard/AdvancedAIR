DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure"
OUTPUT="../cache/offlinefusion_temp_val"
rm -r $OUTPUT
# rm -r ../cache/tmps
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar

# INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
# INFRA_CONFIG_NAME="trainval_config_i.py"
# INFRA_MODEL_NAME="vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"

INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillar_trainval"
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="epoch_20.pth"

VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

# VEHICLE_RESULT_PATH="../output/v2x_models/voxel_rcnn_car/v2xve80/test"
# VEHICLE_RESULT_PATH="../output/v2x_models/voxel_rcnn_car/v2xve80_v2xce80_score0.1_test/test"
# VEHICLE_RESULT_PATH="../output/v2x_models/centerpoint_4class/v2xve80_4class/test"
# VEHICLE_RESULT_PATH="../output/v2x_models/pv_rcnn_plusplus_resnet/v2xv_e80/test"
# VEHICLE_RESULT_PATH="../output/v2x_models/pointpillar_4class/v2xve80/test"
# VEHICLE_RESULT_PATH="../output/v2x_models/centerpoint_car/v2xve80/test"
VEHICLE_RESULT_PATH="../output/v2x_models/model_ensemble/second4clss_centerpointpaintingfix"

# INF_RESULT_PATH="../output/v2xi_models/second_finetune/waymopre_secondcar_wide/val_i"

SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"
DATA_INFO_PATH="../data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative_i/data_info.json"

# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=$1
FUSION_METHOD=$2
DELAY_K=$3
EXTEND_RANGE_START=$4
EXTEND_RANGE_END=$5
TIME_COMPENSATION=$6
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-sync \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --veh-result-path $VEHICLE_RESULT_PATH \
  # --inf-result-path $INF_RESULT_PATH \
  --data-info-path $DATA_INFO_PATH \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION