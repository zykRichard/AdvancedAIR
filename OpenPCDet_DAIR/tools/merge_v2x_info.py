import os
import pickle


v2xi_root = '/home/lixiang/OpenPCDet/data/cooperative-vehicle-infrastructure/infrastructure-side'
v2xv_root = '/home/lixiang/OpenPCDet/data/cooperative-vehicle-infrastructure/vehicle-side'
v2x_root = '/home/lixiang/OpenPCDet/data/cooperative-vehicle-infrastructure/cooperative'

v2xi_info_path = os.path.join(v2xi_root, 'kitti_infos_trainval.pkl')
v2xv_info_path = os.path.join(v2xv_root, 'kitti_infos_trainval.pkl')
v2x_info_path = os.path.join(v2xv_root, 'v2x_infos_trainval.pkl')


print('---------------Merge v2x-i & v2x-v data-info---------------')

with open(v2xv_info_path, 'rb') as f:
    infos = pickle.load(f)
v2xv_infos = [info for info in infos if info['annos']!={}]
for info in v2xv_infos:
    info['src'] = 0 #0 for vehicle side, 1 for inf side
print('v2xv info: {}'.format(len(v2xv_infos)))

with open(v2xi_info_path, 'rb') as f:
    infos = pickle.load(f)
v2xi_infos = [info for info in infos if info['annos']!={}]
for info in v2xi_infos:
    info['src'] = 1
print('v2xi info: {}'.format(len(v2xi_infos)))

v2x_infos = v2xv_infos + v2xi_infos

with open(v2x_info_path, 'wb') as f:
    pickle.dump(v2x_infos, f)
print('Merge process finished: {}'.format(len(v2x_infos)))