import os
import numpy as np
import pickle

result_root = '/home/lixiang/DAIR-V2X/cache/vic-late-lidar/result'
results = os.listdir(result_root)

for result in results:
    print(result)
    result_path = os.path.join(result_root, result)
    with open(result_path, 'rb') as f:
        res = pickle.load(f)
    print(res)
