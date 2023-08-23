# README

Implementation of paper "Addressing Class Imbalance in VIC3D Object Detection: A Model Ensemble and Multi-modal Fusion Innovation"



### Preparation

1. Config environment of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 

2. Running the following instruction in directory `OpenPCDet_DAIR`   :

   ```bash
   python -m pcdet.datasets.v2x.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/v2x_v_dataset.yaml
   ```

   to generate *info* files of dataset.



### Training 

1. Training vehicle-side detection model [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) 

   ```
   cd OpenPCDet_DAIR
   bash scripts/dist_train.sh 4  cfgs/v2x_models/centerpoint_car.yaml
   ```

   Please set arg `TEST_AUGMENTOR` as True in file `centerpoint_car.yaml`. Then:

   ```bash
   python tta_test.py –cfg_file cfgs/v2x_models/centerpoint_car.yaml
   ```

2. Training vehicle-side detection model [Second](https://www.mdpi.com/1424-8220/18/10/3337)

   ```bash
   bash scripts/dist_train.sh 4  cfgs/v2x_models/second_4class.yaml
   ```

   Please set arg `TEST_AUGMENTOR` as True in file `second_4class.yaml`. Then:

   ```bash
   Python tta_test.py –cfg_file cfgs/v2x_models/second_4class.yaml
   ```

3. Merging CenterPoint and Second

   ```bash
   Python model_ensemble.py –-result_a <Model A dir> –result_b <Model B dir>
   ```

4. Training infrastructure-side detection model PointPillar

​		Please refer to （https://drive.google.com/file/d/1BO5dbqmLjC3gTjvQTyfEjhIikFz2P_Om/view?usp=sharing）

5. Merging

   Please set argument `VEHICLE_RESULT_PATH` as the file directory in step3, then :

   ```bash
   cd ${dair-v2x_root}/dair-v2x/v2x
   bash scripts/test_lidar_offline_late_fusion.sh 0 offline_fusion 0 0 100
   ```



### Inference

1. Preparation

   Download DAIR-V2X dataset first, then make a new file directory and transmit into kitti format:

   ```bash
   cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training
   
   python tools/dataset_converter/get_fusion_data_info.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training
   
   rm ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json
   
   mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/fusion_data_info.json ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json
   
   # Kitti Format
   cd ${dair-v2x_root}/dair-v2x
   python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training \
       --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training \
       --split-path ./data/split_datas/cooperative-split-data.json \
       --label-type lidar --sensor-view cooperative --no-classmerge
   
   ```

2. Centerpoint_pointpainting Inference:

   Unzip the result of Painting (painted_lidar_003_fix.zip), then move to `./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/training`

   Then, begin model inference by TTA:

   ```bash
   cd OpenPCDet_DAIR/tools
   python tta_test.py  --cfg_file cfgs/v2x_models/centerpoint_pointpainting.yaml --extra_tag v2xve80_trainval_tta 
   --ckpt ../weight/v2xv_models/centerpoint_pointpainitng/ckpt/checkpoint_epoch_80.pth 
   ```

   

3. Second Inference

   Begin Second inference by TTA :

   ```bash
   cd OpenPCDet_DAIR/tools
   python tta_test.py  --cfg_file cfgs/v2x_models/second_4class.yaml --extra_tag v2xve40_trainval_tta 
   --ckpt ../weight/v2xv_models/second_4class /ckpt/checkpoint_epoch_40.pth 
   ```

   

4. Merging Vehicle-side Inference Result

   ```bash
   cd OpenPCDet_DAIR/tools
   Python model_ensemble.py --cfg_file cfgs/v2x_models/centerpoint_pointpainting.yaml \
   --extra_tag centerpointpainting_second4class_testb \
   --result_a ../output/v2x_models/centerpoint_pointpainting/v2xve80_trainval_tta/testb_v_tta_ensemble \
   --result_b ../output/v2x_models/second_4class/v2xve40_trainval_tta/testb_v_tta_ensembleensemble \
   ```

5. Cooperation Inference

6. First, modify the parameter `VEHICLE_RESULT_PATH` in the `v2x/scripts/test_lidar_offline_late_fusion.sh` file to the folder where the results are stored in Step 4, i.e., `"/Yourpath/OpenPCDet/output/v2x_models/model_ensemble/centerpointpainting_second4class_testb"`

   Then, execute the following command:

   ```bash
   cd DAIR-V2X/v2x
   bash scripts/test_lidar_offline_late_fusion.sh 0 offline_fusion 0 0 100
   ```

   