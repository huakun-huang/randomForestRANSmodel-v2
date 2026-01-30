################################################################################
# Output the inputFeatures.txt
# Output the ratio.txt
# random
################################################################################

import numpy as np
import time, random, os, sys
from pathlib import Path

def readCritical():
    # 获取环境变量中的路径列表
    paths = os.getenv("PINN_INCLUDE")
    if not paths:
        print("错误：环境变量 PINN_INCLUDE 未设置")
        exit(1)
    # 分割路径（自动适配系统分隔符，如 Windows 用 ;，Linux/macOS 用 :）
    path_list = paths.split(os.pathsep)
    target_file = 'Critical.H'
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

def check_directory_exists(dir_path, create):
    if os.path.isdir(dir_path):
        return True
    else:
        if create:
            os.system('mkdir ' + dir_path)
        else:
            print(dir_path+' does not exist')
        return False

# test dataRANS
dirs = 'dataRANS'
exists = check_directory_exists(dirs, True) 

#Test RANS folder
exists = check_directory_exists('RANS', False)
if exists==False:
    print('RANS folder does not exist')
exists2 = check_directory_exists('Fidelity', False)
if exists2==False:
    print('Fidelity folder does not exist')

if exists and exists2:
    ras = np.loadtxt('RANS/energy.txt')
    les = np.loadtxt('Fidelity/energy.txt')
    input = np.loadtxt('RANS/inputFeatures.txt')
    
    print('output features', input.shape[1])
    print('RANS cells:', input.shape[0])
    print('Fidelity cells:', les.shape[0])

    Pk_ras = ras[:,0]
    Pk_les = les[:,0]

    #delete odd data
    critical = readCritical() #阈值取2比较好

###############################################
    size = 1
    try:
        size = np.loadtxt('size.txt')
    except (IOError, ValueError) as e:
        size = 1
###############################################
    maxmumCell = len(les[:,1])
    trainSampleRow = random.sample(range(0, maxmumCell), int(len(les[:,1])*size))
    print('Random output results')
    badCells=0
    with open('dataRANS/inputFeatures.txt', 'w') as fi:
        with open('dataRANS/ratio.txt', 'w') as f:
          for n in range(len(trainSampleRow)):
           #for n in range(len(Pk_ras)):
            row = trainSampleRow[n]
            try:
                Pk = Pk_les[row]/max(Pk_ras[row], 1e-15)
                Dk = les[:,1][row]/max(ras[:,1][row], 1e-15)
                Pw = les[:,2][row]/max(ras[:,2][row], 1e-15)
                Dw = les[:,3][row]/max(ras[:,3][row], 1e-15)
                if Pk<0 or Dk<0 or Pw<0 or Dw<0 :
                    badCells=badCells+1
                    #f.write(repr(Pk)+'\t')
                    #f.write(repr(Dk)+'\t')
                    #f.write(repr(Pw)+'\t')
                    #f.write(repr(Dw)+'\n')
                    continue
                if(Pk>critical or Dk>critical or Pw>critical or Dw>critical):
                    badCells=badCells+1
                    continue
                    f.write(repr(critical)+'\t')
                    f.write(repr(critical)+'\t')
                    f.write(repr(critical)+'\t')
                    f.write(repr(critical)+'\n')
                    for k in range(input.shape[1]):
                        fi.write(repr(input[row][k])+'\t')
                    fi.write('\n')
                else:
                    f.write(repr(Pk/critical)+'\t')
                    f.write(repr(Dk/critical)+'\t')
                    f.write(repr(Pw/critical)+'\t')
                    f.write(repr(Dw/critical)+'\n')
                    for k in range(input.shape[1]):
                        fi.write(repr(input[row][k])+'\t')
                    fi.write('\n')
            except:
                continue
        f.close()
    fi.close()
    print('badCells', badCells)
else:
    print('Please check your folder')
