# -*- coding: utf-8 -*-

import torch
import torchvision
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math as mt
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


import sys, os
sys.path.append('/home/hhk/OpenFOAM/PINNv/include')

import errorPlot as ep

#Test GPU
def GPUTest():
    if torch.cuda.is_available():
        return 1
    else:
        return 0
    
#GPU information output
def GPUInfo():
    print("GPU 设备数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"device {i} 名称: {torch.cuda.get_device_name(i)}")
        print(f"device {i} 计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"device {i} 总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

def get_all_directories(path='.'):
    # 获取当前目录中的所有文件和文件夹
    items = os.listdir(path)
    
    # 只保留文件夹
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    
    return directories

#load inputFeature and outputFeatures
def dataInput():
    trainFolder = get_all_directories('../dataBase_v2_multil')
    
    for n in range(len(trainFolder)):  
        folder = trainFolder[n]
        print('handle in '+folder)
        input = np.loadtxt('../dataBase_v2_multil/'+folder+'/dataRANS/inputFeatures.txt')
        data = np.loadtxt('../dataBase_v2_multil/'+folder+'/dataRANS/ratio.txt')
    
        if n==0:
            input_ = input
            data_ = data
        else:    
            input_ = arrayAppend(input_, input)  
            data_ = arrayAppend(data_, data) 
        
    trainSampleRow = random.sample(range(0, input_.shape[0]-1), int(input_.shape[0]*0.96)) 
    trainSampleRow = sorted(trainSampleRow)
    k = 0
    testSample=[]
    for n in range(input_.shape[0]):
        if n!=trainSampleRow[k]:
            testSample.append(n)
        elif n== trainSampleRow[k]:
            k = k+1
            if k >= len(trainSampleRow):
                break    
        
    writeFile("dataRANS/train.txt", input_, trainSampleRow)    
    writeFile("dataRANS/trainRes.txt", data_, trainSampleRow)
      
    writeFile("dataRANS/test.txt", input_, testSample)
    writeFile("dataRANS/testRes.txt", data_, testSample) 

# write file
def writeFile(fileName, input, rows):
    k=0
    with open(fileName, 'w') as f:
        for n in range(len(rows)):
            for m in range(input.shape[1]):
                f.write(repr(input[rows[n]][m])+'\t');
            f.write('\n');
    f.close()

def writeTest(fileName, input):
    with open(fileName, 'w') as f:
        for n in range(input.shape[0]):
            for m in range(input.shape[1]):
                f.write(repr(input[n][m])+'\t')
            f.write('\n')
    f.close()

# append aaray
def arrayAppend(a, b):
    rows = a.shape[0]+b.shape[0]
    colus = a.shape[1]
    temp = np.zeros((rows, colus))
    for n in range(temp.shape[0]):
        for m in range(temp.shape[1]):
            if(n<a.shape[0]):
                temp[n][m] = a[n][m]
            else:
                temp[n][m] = b[n-a.shape[0]][m]
    return temp
 

#################### define a ResidualBlock for CNN ###################
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.dropout = torch.nn.Dropout2d(dropout_prob)  # 添加 Dropout
        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # 应用 Dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualBlockANN(torch.nn.Module):
    def __init__(self, in_features, out_features, pout):
        super(ResidualBlockANN, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=pout)
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features)
            )

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

#################### define a net which combines CNN and ANN ###################
class Net(torch.nn.Module):
    def __init__(self, layers_config, input_size, hidden_size=80, num_layers=10, output_size=4, dropout_prob=0.5):
        super().__init__()
        self.layers = []
        self.layersANN = torch.nn.ModuleList()
        in_channels = layers_config[0]['in_channels']
        k=0
        for config in layers_config:
            self.layers.append(torch.nn.Conv2d(in_channels, config['out_channels'], 
                  kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding']))
            self.layers.append(ResidualBlock(config['out_channels'], 
                  config['out_channels'], dropout_prob=config['dropout_prob']))
            if k %2==0:
                self.layers.append(torch.nn.MaxPool2d(kernel_size=1, stride=2))  #池化层
                k = k+1
            in_channels = config['out_channels']
        self.layers = torch.nn.ModuleList(self.layers)
        #self.fc = torch.nn.Linear(in_channels * 4 * 4, 4)
        self.fc = torch.nn.Linear(in_channels * 4 * 4, input_size)
        self.dropout = torch.nn.Dropout(0.2)
        
        # hidden layer
        for kk in range(num_layers - 1):
            self.layersANN.append(ResidualBlockANN(input_size, hidden_size, dropout_prob))
            input_size = hidden_size
            
        # output layer
        self.output_layers = torch.nn.Linear(hidden_size, output_size)
        

    def forward(self, xb):
        for layer in self.layers:
            xb = layer(xb)
        xb = F.adaptive_avg_pool2d(xb, 4)
        xb = xb.view(xb.size(0), -1)
        xb = self.dropout(xb)
        xb = self.fc(xb)
        for layer in self.layersANN:
            xb = layer(xb)
        xb = self.output_layers(xb)             
        xb = torch.tanh(xb)+1
        return xb

#### get data ##############################
def get_data(train_ds, valid_ds, bs):
    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
    
############### compute error ############################
def error(outputs, targets, error, runGPU=False):
    if runGPU:
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().numpy()
    else:
        outputs = outputs.detach().numpy()
        targets = targets.numpy()
    for n in range(outputs.shape[0]):
        for m in range(outputs.shape[1]):
            er = ((outputs[n][m]-targets[n][m])*(outputs[n][m]-targets[n][m]))**0.5/max(((targets[n][m])**2)**0.5, 1e-5)
            error.append(er)
    return error

############## write error ################################
def writeError(error):
    with open('error.txt', 'a+') as f:
        for n in range(len(error)):
            f.write(repr(error[n])+'\n')
    f.close() 

##################### data ####################    
def preprocess(x, y):
    #                   row cls
    return x.view(-1, 1, 1, x.shape[1]), y

################## wrapped ####################
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def loadTrainingData(newData):
    if newData:
        dataInput()
    trainFeatures =  np.loadtxt("dataRANS/train.txt")
    trainResponses = np.loadtxt("dataRANS/trainRes.txt")       
    return trainFeatures, trainResponses
    
def loadTestData():
    testFeatures = np.loadtxt("dataRANS/test.txt")
    testResponses = np.loadtxt("dataRANS/testRes.txt")
    return testFeatures, testResponses

# Pass an optimizer for training set
# https://blog.csdn.net/weixin_37993251/article/details/88916913
def loss_batch(model, loss_func, xb, yb, opt=None):
    runGPU = GPUTest()
    if runGPU:
        xb, yb = xb.cuda(), yb.cuda()
    pred = model(xb)
    loss = loss_func(pred, yb)
    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()
    else:
        errors=[]
        errors =  error(pred, yb, errors, runGPU)
        writeError(errors)
        
    return loss.item(), len(xb)

################## model save ################### 
def model_save(train_dl, model):
    for xb, yb in train_dl:
      xb1 = xb
      break
    runGPU = GPUTest()
    if runGPU:
        xb1 = xb.cpu().numpy()
        torch.save(model.module.state_dict(), 'hhk_PINNmodel.pt')
    else:
        torch.save(model.module.state_dict(), 'hhk_PINNmodel.pt')
        #torch.save(model.state_dict(), 'hhk_PINNmodel.pt')
    #traced_script_module = torch.jit.trace(model, xb1)
    #traced_script_module.save('hhk_PINNmodel.pt')
    
################## model load ###################    
def model_load(model): 
    runGPU = GPUTest()
    model_path = '/mnt/d/machineLearningVersion/CNN-ANN/hhk_PINNmodel.pt'
    if runGPU:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load('/mnt/d/machineLearningVersion/CNN-ANN/hhk_PINNmodel.pt'))
        #model.load_state_dict(torch.load('hhk_PINNmodel.pt'))
    return model

################## model load ###################    
def model_loadAgain(model): 
    runGPU = GPUTest()
    model_path = '/mnt/d/machineLearningVersion/CNN-ANN/hhk_PINNmodel.pt'
    if runGPU:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = torch.nn.DataParallel(model)
    return model

################# fit ############################
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    loss_max=100
    scheduler = StepLR(opt, step_size=30, gamma=0.1)
    runGPU = GPUTest()
    for epoch in range(epochs):
        model.train()
        try:
            os.remove('error.txt')
        except FileNotFoundError:
            print('not found error.txt')
        for xb, yb in train_dl:
            if runGPU:
                xb, yb = xb.cuda(), yb.cuda()
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb, in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}, loss {val_loss}, learning rate {scheduler.get_last_lr()}")
        if(val_loss<loss_max):
            model_save(train_dl, model)
            loss_max = val_loss
            ep.plotError(1)
            
################# define model ############################
def defineModel(newData, layers_config, bs, loss_func, epochs, lrs, dropout, hiddenSize, layerNum):
    #load data   
    trainFeatures, trainResponses = loadTrainingData(newData)
    
    inputSize = trainFeatures.shape[1]
    outputSize = trainResponses.shape[1]
    
    trainFeatures = torch.from_numpy(trainFeatures)
    trainResponses = torch.from_numpy(trainResponses)

    testFeatures, testResponses = loadTestData()
    testFeatures = torch.from_numpy(testFeatures)
    testResponses = torch.from_numpy(testResponses)

    testResponses = testResponses.to(torch.float32)
    testFeatures = testFeatures.to(torch.float32)

    trainFeatures = trainFeatures.to(torch.float32)
    trainResponses = trainResponses.to(torch.float32)

    ####################################
    #hyperparameter
    """
    #batch size
    bs = 100

    #loss_function
    loss_func = torch.nn.HuberLoss(delta=0.3) 

    #epochs
    epochs = 60

    #learning rate
    lrs = 0.01
    
    #dropout
    dropout = 0.2
    
    #hidden layer unit
    hiddenSize = 80
    
    #hidden layer num
    layerNum = 10
    """
    #####################################

    train_ds = Data.TensorDataset(trainFeatures, trainResponses)
    valid_ds = Data.TensorDataset(testFeatures, testResponses)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    
    """
    layers_config = [
        {'in_channels': 1, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dropout_prob': dropout},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dropout_prob': dropout},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dropout_prob': dropout},
        {'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dropout_prob': dropout},
    ]
    """

    model = Net(layers_config, inputSize, hiddenSize, layerNum, outputSize, dropout)
    
    if newData==0:
        model = model_load(model)
        
    runGPU = GPUTest()
    #model = torch.nn.DataParallel(model)
    if runGPU:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        
    return model, loss_func, lrs, epochs, train_dl, valid_dl

def train(newData, layers_config, bs, loss_func, epochs, lrs, dropout, hiddenSize, layerNum):
    
    regModel, loss_func, lrs, epochs, train_dl, valid_dl = defineModel(newData, layers_config,
         bs, loss_func, epochs, lrs, dropout, hiddenSize, layerNum)
         
    opt = optim.SGD(regModel.parameters(), lr=lrs, momentum=0.9)
    
    GPUInfo()
    
    fit(epochs, regModel, loss_func, opt, train_dl, valid_dl)
    
    
                       
    