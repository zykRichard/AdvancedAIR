DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure-testA"
OUTPUT="../cache/vic-late-lidar"
rm -r $OUTPUT
rm -r ../cache
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/test
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar

INFRA_MODEL_PATH="../work_dirs/second_v2x_i2"
INFRA_CONFIG_NAME="second_v2x_i2.py"
INFRA_MODEL_NAME="latest.pth"

VEHICLE_MODEL_PATH="../work_dirs/second_v2x_v2"
VEHICLE_CONFIG_NAME="second_v2x_v2.py"
VEHICLE_MODEL_NAME="latest.pth"


SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"
#SPLIT_DATA_PATH="../data/split_datas/example-cooperative-split-data.json"

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
  --split test_A \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION