from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import (
                convertdetectiondict2listoflist,
                    )
from tqdm import tqdm
import pickle, re
import numpy as np
import pandas as pd

meta_columns = ['head', 'neck', 'handl', 'handr', 'hip', 'footl', 'footr', 'tl', 'tr', 'bl', 'br'] 
col = []

for id in range(2):
    if id == 1:
        for i in meta_columns[:-4]:
            col.append('x_' + i + str(id))
            col.append('y_' + i + str(id))
            col.append('conf_' + i + str(id))
    else:
        for i in meta_columns:
            col.append('x_' + i + str(id))
            col.append('y_' + i + str(id))
            col.append('conf_' + i + str(id))
 

with open('MP-TestDLC_resnet50_PoCNov1shuffle1_50000_full.pickle', "rb") as file:
    data = pickle.load(file)

header = data.pop("metadata")
all_jointnames = header["all_joints_names"]

numjoints = len(all_jointnames)
bpts = range(numjoints)
frame_names = list(data)
frames = [int(re.findall(r"\d+", name)[0]) for name in frame_names]

#n = 0
#ind = frames.index(n)


df = pd.DataFrame(index=frames, columns=col)

for n in tqdm(frames):
#for n in range(2):
    dets = convertdetectiondict2listoflist(data[frame_names[n]], bpts)
    # print(n)
    for m in range(11):
        for p in range(3):
           # print(n, m - 11, p)
            if m < 7:
                try:
                    df.iloc[n, 3*m + p] = dets[m][0][p]  
                    df.iloc[n, (3*m + p) + 33] = dets[m][1][p]  
                    # print('succes')
                except:
                    # print('fail')
                    pass
            else:
                try:
                    df.iloc[n, 3*m + p] = dets[m][0][p]  
                except:
                    pass

df.to_csv('detections.csv')