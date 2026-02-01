## Import system modules
# sci computing
import numpy as np
# sklearn importing
from sklearn.ensemble._forest import RandomForestRegressor
# plotting
import matplotlib.pyplot as plt  # for plotting
#import matplotlib as mp

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#save model
import pickle

import time, random, os, sys

import math as mt

filter = 0.03

def error(outputs, targets):
    errors=[]
    for n in range(outputs.shape[0]):
        for m in range(outputs.shape[1]):
            er = ((outputs[n][m]-targets[n][m])*(outputs[n][m]-targets[n][m]))**0.5/max(((targets[n][m])**2)**0.5, 1e-5)
            errors.append(er)
    return errors

def writeError(error):
    with open('error.txt', 'w') as f:
        for n in range(len(error)):
            f.write(repr(error[n])+'\n')
    f.close() 

def accuracy(y_true, y_pre):
    score = 0
    for n in range(y_true.shape[0]):
        for m in range(y_true.shape[1]):
            score = score + (y_true[n][m]-y_pre[n][m])*(y_true[n][m]-y_pre[n][m])
    score = mt.sqrt(score/(y_true.shape[0]*y_true.shape[1]))
    errors = error(y_pre, y_true)
    writeError(errors)
    return score

def arrayAppend(a, b):
    temp = np.vstack((a, b))
    return temp

def get_all_directories(path='.'):
    # 获取当前目录中的所有文件和文件夹
    items = os.listdir(path)
    
    # 只保留文件夹
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    
    return directories
      
#load inputFeature and outputFeatures
skipFolder=['periodicHill', 
            'cylinder2000-2D-WALE', 
            'cylinder2000-2D-IDDES',
            'Impingement_plane_11000_H6'            
            'Impingement_round_70000_H6',
            'T3B-'
            ] 

def dataInput(folders):
    trainFolder = folders
    featuresAll=0
    for n in range(len(trainFolder)):
        skip = False
        for mn in range(len(skipFolder)):
            if trainFolder[n]==skipFolder[mn]:
                skip = True                
                break
        if skip:
            continue
        folder = trainFolder[n]
        input = np.loadtxt('../dataBase_v2_multil/'+folder+'/dataRANS/inputFeatures.txt')
        data = np.loadtxt('../dataBase_v2_multil/'+folder+'/dataRANS/ratio.txt')
        
        print('handle in '+folder)
        if n==0: #18
            input_ = input
            data_ = data
            featuresAll = input.shape[1]
            print(featuresAll)
        else: 
            if featuresAll !=  input.shape[1]:
                print(folder, 'features is not equal to', featuresAll, 'it is ', input.shape[1])
                sys.exit(0)               
            input_ = arrayAppend(input_, input)  
            data_ = arrayAppend(data_, data) 
    #input_ = np.delete(input_, [58], axis=1)  #不考虑温度比值          
    return input_, data_    
     
            
def loadTrainingData():
    trainFolder = get_all_directories('../dataBase_v2_multil')
    X, Y = dataInput(trainFolder)
    trainFeatures, testFeatures, trainResponses, testResponses = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    #testFeatures, testResponses = dataInput(testFolder, 0)
    
    #cols=[0, 1,2,3,4,5,6,7,8,9,10,56,57]
    #trainFeatures =  trainFeatures[:, cols]
    #testFeatures = testFeatures[:, cols]
    return trainFeatures, trainResponses, testFeatures, testResponses
    


    
def randomForest(trainFeatures, trainResponses, testFeatures, testResponses, score0, maxFeatures = 'log2', nTree=200):
    ## Settings of random forests regressor
    regModel = RandomForestRegressor(n_estimators=nTree, 
                                     max_features=maxFeatures,
                                     min_samples_leaf=4,
                                     min_samples_split=2,                                     
                                     n_jobs=8, 
                                     oob_score=True)    
    ## Train the random forests regressor
    regModel.fit(trainFeatures, trainResponses)
    importances = regModel.feature_importances_
    
    ## Prediction
    testResponsesPred = regModel.predict(testFeatures)
    score2 = regModel.score(testFeatures, testResponses)
    oob_error = 1-regModel.oob_score_
    score = accuracy(testResponses, testResponsesPred)
    print('score:', score2, 'oob_error:', oob_error)
    #  save model
    if(score<score0):
        with open("PINN_RANS_Model.pkl", "wb") as f:
            pickle.dump(regModel, f)
    return score, score2, importances

def output(testResponsesPred):
    fid=open('data_predict.txt','w')
    for n in range(testResponsesPred.shape[0]):
      for m in range(testResponsesPred.shape[1]):
         fid.write("%e  " % testResponsesPred[n][m])
      fid.write("\n")
    fid.close()

def write():
    try:
        #old importance message
        imp = np.loadtxt('structure.txt')
        impCols=[]
        for nn in range(len(imp)): 
            if imp[nn]>filter:
                impCols.append(nn)
        index=0
        print(impCols)
        importance=np.loadtxt('structureVS.txt')
        with open('structure.txt', 'w') as f:
            f.write('#nTree: '+repr(9)+'\n');
            f.write('#max_feature '+repr(5)+'\n')
            for mn in range(len(importance)):
                if index==0:
                    for kkk in range(impCols[index]):
                        f.write('0\n')
                    index = index +1
                else:
                    for kkks in range(impCols[index]-impCols[index-1]-1): 
                        f.write('0\n')
                    index = index +1                                
                f.write(repr(importance[mn])+'\n')
            for vs in range(len(imp)-impCols[index-1]-1):
                f.write('0\n')
        f.close()
    except:
        print('haha') 

def miniTrain(trainFeatures,trainResponses,testFeatures,testResponses,score0, max_Feature, nTree=200):
    bestScore = score0
    print('Mini loop starts')
    for ml in range(1):
        score, score2, importance = randomForest(trainFeatures, trainResponses, testFeatures, testResponses, score0, max_Feature, nTree)
        if(score<bestScore):
            bestScore = score   
            with open('structureVS.txt', 'w') as f:
                f.write('#nTree: '+repr(nTree)+'\n');
                f.write('#max_feature '+repr(max_Feature)+'\n')
                for mn in range(importance.shape[0]):
                    f.write(repr(importance[mn])+'\n')
            f.close()
            #write()
            
        with open('fit.txt', 'a+') as f:
            f.write(repr(max_Feature)+' '+repr(nTree)+' '+repr(score)+'\n')
        f.close()
        print('maxFreature: '+repr(max_Feature)+' nTree = '+repr(nTree)+' | score = '+repr(score)+' | best score = '+repr(bestScore));
        if(score<0.05):
            break
        if(score2<0):
            break
    print('Mini loop ends')
    return bestScore

#os.system('mkdir dataRANS')       
trainFeatures, trainResponses, testFeatures, testResponses = loadTrainingData()

score0 = 0.2
score = 1

#通过理论公式评估随机深林的特征数
feature = 1 + mt.log(trainFeatures.shape[1], 2)
print('随机深林理论特征大小：', feature)
print('缩减特征数到：', testFeatures.shape[1])
score = miniTrain(trainFeatures,trainResponses,testFeatures,testResponses,score0, 9, 390)

""""
for n in range(1):
    for m in range(7, 14):
        for k in range(40):
            nTree = 200+50*k
            score = miniTrain(trainFeatures,trainResponses,testFeatures,testResponses,score0, m, nTree)
            score0 = score
            if nTree>600:
                break
            if(score<0.05):
                break
        if(score<0.05):
            break
    if(score<0.05):
        break                      
"""            
            