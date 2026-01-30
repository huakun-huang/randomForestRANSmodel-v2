# -*- coding: utf-8 -*-

## Import system modules
# sci computing
import numpy as np
# sklearn importing
from sklearn.ensemble._forest import RandomForestRegressor
import threading


#save model
import pickle

import argparse

import time, os
from pathlib import Path

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

def findFile(env, files):
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
    full_path='0'
    for path in path_list:
        if not path:  # 跳过空路径
            continue
        full_path = Path(path) / target_file
        if full_path.is_file():
            found = True
            break  # 找到后退出循环
    if not found:
        print(f"错误：在所有路径中未找到文件 {target_file}")
        exit(1)
    return full_path

critical=readCritical("PINN_INCLUDE", 'Critical.H')

# [1]  load the input features

def loadTrainData():
    #cols=[0, 1,2,3,4,5,6,7,8,9,10,56,57]
    data = np.loadtxt('inputFeatures.txt')
    #data = np.delete(data, [58], axis=1) #去掉温度比值，即最后一个特征
    #data = data[:, cols]
    return data


# [2]  load the model and predict
def loadModel(n):
    print('Load model---PINN_RANS_Model.pkl version-', n)
    """
    #model_path = '/mnt/d/machineLearningVersion/v1_SUSTECH_26_08_2024/PINN_RANS_Model.pkl'
    if n==1:
        model_path = '/home/hhk/OpenFOAM/PINN_RANS_Modelold.pkl'
        print('Select method in: /home/hhk/OpenFOAM/PINN_RANS_Modelold.pkl')
    elif n==2:
        model_path = '/mnt/d/machineLearningVersion/randomForestV2/PINN_RANS_Model.pkl'
        print('Select method in: /mnt/d/machineLearningVersion/randomForestV2/PINN_RANS_Model.pkl')
    elif n==3:
        model_path = '/mnt/d/machineLearningVersion/randomForestV3/PINN_RANS_Model.pkl'
        print('Select method in: /mnt/d/machineLearningVersion/randomForestV3/PINN_RANS_Model.pkl')
    else:
        raise Exception('Invalid version of method. Application abort')
    """
    #model_path = findFile('PINN_MODEL', 'PINN_RANS_Model-c5.pkl')
    model_path = findFile('PINN_MODEL', 'PINN_RANS_Model.pkl')
    print(model_path)
    with open(model_path, "rb") as f:
        regModel = pickle.load(f)
    return regModel
    
def predict(inputFeatures, regModel):
    print('Predict data')
    ResponsesPred = regModel.predict(inputFeatures)
    return ResponsesPred*critical

def output(ResponsesPred):
    print('Output outputResult.txt')
    with open('outputResult.txt', 'w') as f:
        for prediction in ResponsesPred:
                 f.write(" ".join(map(str, prediction.tolist())) + "\n")
    """ 
    with open('outputResult.txt', 'w') as f:
        for xi in range(ResponsesPred.shape[0]):
            for yi in range(ResponsesPred.shape[1]):
                f.write("%g\t" % ResponsesPred[xi][yi])
            f.write('\n')
    """
    f.close()
    print('Machine learning is done')   

# [3] run the regression solver

#print('done')
def main():
    parser = argparse.ArgumentParser(description='Predict using the random forest method.')
    parser.add_argument('--v', type=int, nargs='+', required=True, help='Select a version of random forest')
    args = parser.parse_args()
    
    inputFeatures = loadTrainData()
    version = args.v[0] #input('Please select a version number, 1-old, 2-new: ')
    regModel = loadModel(int(version))
    ResponsesPred = predict(inputFeatures, regModel)
    output(ResponsesPred)

if __name__ == '__main__':
    main()