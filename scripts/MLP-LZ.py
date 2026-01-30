import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import pickle
import joblib
from pathlib import Path

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

def predict():
    path = '/mnt/d/machineLearningVersion/CNN/singleCNN/MLP/Luze-MLP/'
    pickle_out_pre = open(path+'dy_model_pre.pickle','rb')
    model_pre_pickle = pickle.load(pickle_out_pre)

    scaler_x = joblib.load(path+'x_scaler.pkl')
    scaler_y = joblib.load(path+'y_scaler.pkl') 
    
    data = np.loadtxt('inputFeatures.txt')
    data_x=scaler_x.transform(data)
    
    Y_pre=model_pre_pickle.predict(data_x)
    ResponsesPred = scaler_y.inverse_transform(Y_pre)
    
    critical=readCritical("PINN_INCLUDE", 'Critical.H')
    
    with open('outputResult.txt', 'w') as f:
        for row in ResponsesPred:
            f.write('\t'.join(f"{val*critical:.6f}" for val in row) + '\n')
    f.close()


predict()



