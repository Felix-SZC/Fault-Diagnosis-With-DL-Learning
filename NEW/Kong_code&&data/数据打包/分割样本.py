# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:49:58 2025

@author: kongz
"""

import numpy as np
from scipy import io

all_data = np.load('all_data.npy')

all_label = np.load('all_label.npy')

#1正常
#2轴承内圈
#3轴承外圈
#4轴承滚子
#5轴承组合
#6齿根裂纹
#7齿轮断齿
#8齿轮点蚀
#9齿轮缺齿

list = ['1正常',
'2轴承内圈',
'3轴承外圈',
'4轴承滚子',
'5轴承组合',
'6齿根裂纹',
'7齿轮断齿',
'8齿轮点蚀',
'9齿轮缺齿']

#取第二个传感器
sensor_len = int(all_data.shape[1] / 9)

all_data = all_data[:,sensor_len:(sensor_len+sensor_len)]

#取前48秒数据
all_data = all_data[:,0:25600*48]

sensor_len = 25600*48

reshape_all_data = np.reshape(all_data,shape=[9,4*sensor_len]) #9条数据

reshape_all_data = np.reshape(reshape_all_data,shape=[-1,4096])

#io.savemat('data_sample.mat', {'data': reshape_all_data})

reshape_all_label = np.empty([reshape_all_data.shape[0]],dtype=object)

for i in range(reshape_all_label.shape[0]):
    j = i//1200 
    reshape_all_label[i] = list[j]



io.savemat('data_sample.mat', {'data': reshape_all_data})

