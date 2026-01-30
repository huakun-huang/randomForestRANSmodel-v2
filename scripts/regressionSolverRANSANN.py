# -*- coding: utf-8 -*-

import torch
import torchvision
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import random, sys, os
import math as mt
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.append('/mnt/d/Codes/PINNv/scripts')

sys.path.append('/mnt/d/Codes/PINNv/include/')
#import ANNModel as Ah
#import dataModel as dm
#import CNNModel as CN
import hhkNNModel as hm
import hhkNN as hn

import DUTModel as md


# [1]  load the input features
def loadTrainData():
    data = np.loadtxt('inputFeatures.txt')
    return data

"""
def loadPtModel(version):
    model_path = '/mnt/d/machineLearningVersion/CNN/ReNet/critical5/0.8585/hhk_PINNmodel.pt'  
    model = hm.model()   
    ANNs = hn.model_load2(model, modelName=model_path) #torch.jit.load(model_path)
    #ANNs = torch.jit.load(model_path)
    return ANNs

"""
    
#[2] read critical
def readCritical(env, files):
    # 获取环境变量中的路径列表
    paths = os.getenv(env)
    if not paths:
        print("错误：环境变量 PINN_INCLUDE 未设置")
        exit(1)
    # 分割路径（自动适配系统分隔符，如 Windows 用 ;，Linux/macOS 用 :）
    path_list = paths.split(os.pathsep)
    target_file = files
    # 遍历路径搜索文件
    found = False
    words='uu'
    for path in path_list:
        if not path:  # 跳过空路径
            continue
        full_path = Path(path) / target_file
        if full_path.is_file():
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()  
                    words = first_line.split()            
                    if len(words) >= 3:
                        print('Critical:', words[2])                   
                    else:
                        print("错误：第一行不足三个字符串")
                        os.exit(0)
                    found = True
                    f.close()
                    break  # 找到后退出循环
            except Exception as e:
                print(f"读取文件 {full_path} 失败：{str(e)}")
    if not found:
        print(f"错误：在所有路径中未找到文件 {target_file}")
        exit(1)
    return float(words[2])

#Test GPU
def GPUTest():
    if torch.cuda.is_available():
        return 1
    else:
        return 0

# [2]  load the model and predict
def predict(inputFeatures, version):
    """
    CNN = False
    regModel = loadPtModel(version)
    CNN = True

    data = torch.from_numpy(inputFeatures)
    data = data.to(torch.float32)
    
    runGPU = GPUTest()
    if runGPU:
        data = data.cuda()
        device = torch.device('cuda')
        regModel.to(device)
        
    regModel.eval()      
    with torch.no_grad():
        ResponsesPred = regModel(data)
            
    
    if runGPU:
        ResponsesPred = ResponsesPred.cpu()
    
    
    ResponsesPred = ResponsesPred.clamp(min=0)
    
    critical=readCritical("PINN_INCLUDE", 'Critical.H')
    critical = 5
    
    with open('outputResult.txt', 'w') as f:
        for row in ResponsesPred:
            f.write('\t'.join(f"{val*critical:.6f}" for val in row) + '\n')
    f.close()
    
    print('Machine learning is done')
    """
    #训练结束后
    ResponsesPred = md.predictResMLP(inputFeatures)
    
    critical=readCritical("PINN_INCLUDE", 'Critical.H')
    
    # 1. 负值归零（向量化）
    ResponsesPred = np.maximum(ResponsesPred, 0.0)
    
    
    with open('outputResult.txt', 'w') as f:
        for row in ResponsesPred:
            f.write('\t'.join(f"{val*critical:.12f}" for val in row) + '\n')
    f.close()
    
    print('Machine learning is done')
    return ResponsesPred
    

# [3] run the regression solver
def main():
    parser = argparse.ArgumentParser(description='Predict using the ANN method.')
    parser.add_argument('--v', type=int, nargs='+', required=True, help='Select a version of ANN')
    args = parser.parse_args()

    inputFeatures = loadTrainData()
    print("load data is done")
    version = args.v[0] #input('Please select a version number, 2-old, 3-new-part, 4-new_partial, 5-CNN: ')
    y_pred = predict(inputFeatures, int(version))

if __name__ == '__main__':
    main()