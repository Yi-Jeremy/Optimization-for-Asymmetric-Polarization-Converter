# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 11:09
# @Author  : Chuqiao Yi
# @File    : preprocess.py
# @Software: PyCharm
import os
import numpy as np
import h5py
import pandas as pd
import mat73

def convertData(yy_file,yx_file,save_dir):
    Tyy_test,Tyx_test=yy_file,yx_file
    length=Tyy_test.shape[0]
    cou=0
    test=[]
    for Tyy,Tyx in zip(Tyy_test,Tyx_test):
        Tyy,Tyx=mat73.loadmat(Tyy),mat73.loadmat(Tyx)
        help=[]
        help.extend([Tyy['Ha'].tolist(),Tyy['Hc'].tolist(),Tyy['Ka'].tolist(),Tyy['n'].tolist()/100])
        help.extend(Tyy['my2'].tolist()[171:89:-1]) # 波长对应从430nm-550nm，共80个点
        help.extend(Tyx['my2'].tolist()[171:89:-1])
        test.append(help)
        print(f'count {cou}/{length}')

        cou+=1
    test=np.array(test)
    np.save(save_dir,test)

root=r'D:\PyPro\PolarNet\bi-network\data\dataset'
file_name=os.listdir('dataset/Tyx1')
Tyx_file_name=list(map(lambda x:root+'\Tyx1\\'+x,file_name))
Tyy_file_name=list(map(lambda x:root+'\Tyy1\\'+x,file_name))
Tyy_file_name=np.array(Tyy_file_name)
Tyx_file_name=np.array(Tyx_file_name)
index=np.arange(0,10000)
np.random.shuffle(index)
Tyx_training=Tyx_file_name[index[:8000]]
Tyy_training=Tyy_file_name[index[:8000]]

Tyx_validation=Tyx_file_name[index[8000:9000]]
Tyy_validation=Tyy_file_name[index[8000:9000]]

Tyx_test=Tyx_file_name[index[9000:]]
Tyy_test=Tyy_file_name[index[9000:]]

colName=['Ha','Hc','Ka','n','Tyy','Tyx']
with open('parametername.txt','w') as f:
    for name in colName:
        f.write(name+'\n')

convertData(Tyy_training,Tyx_training,r'D:\PyPro\PolarNet\bi-network\data\trainingData\training1.npy')
convertData(Tyy_validation,Tyx_validation,r'D:\PyPro\PolarNet\bi-network\data\validationData\validation1.npy')
convertData(Tyy_test,Tyx_test,r'D:\PyPro\PolarNet\bi-network\data\testData\testdata1.npy')
# solarData=pd.DataFrame(np.zeros(shape=(Tyy_test.shape[0],200*2+4)))


