# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 9:48
# @Author  : Chuqiao Yi
# @File    : nn.py
# @Software: PyCharm
from tensorflow.keras.layers import Input,Dense,BatchNormalization,GRU,Reshape,LeakyReLU,Add
from tensorflow.keras.regularizers import l1_l2
from DirectNN.NTNLayer import TNLayer
from tensorflow.keras import Model
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger,TensorBoard
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import numpy as np
import tensorflow as tf
AV_GPU_NUM=tf.config.experimental.list_physical_devices('GPU') # Avaiable GPU nums
class nn:
    def __init__(self,input_dim=4,output_dim=82,learningRate=0.005):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.learningRate=learningRate

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        layer_1 = Dense(10, activation='tanh')(input_layer)
        layer_2 = Dense(25, activation='relu')(layer_1)
        # layer_2=Reshape(target_shape=(5,5))(layer_2)
        # gru_layer=GRU(50,kernel_regularizer=l1_l2(l1=0.05,l2=0.05))(layer_2)
        layer_3 = Dense(144, activation='relu')(layer_2)
        bt_layer = BatchNormalization()(layer_3)
        # ntn_layer_yy=TNLayer()(bt_layer)
        # ntn_layer_yy=TNLayer()(ntn_layer_yy)
        ntn_layer_yy = Dense(200, activation='relu')(bt_layer)
        add_layer_yy = ntn_layer_yy
        ntn_layer_yy = Dense(500, activation='relu')(ntn_layer_yy)
        # ntn_layer_yy = Dense(500, activation='relu')(ntn_layer_yy)
        # ntn_layer_yy = Dense(500, activation='relu')(ntn_layer_yy)
        ntn_layer_yy = Dense(200, activation='relu')(ntn_layer_yy)
        ntn_layer_yy = Add()([add_layer_yy, ntn_layer_yy])
        # ntn_layer_yx=TNLayer()(bt_layer)
        # ntn_layer_yx=TNLayer()(ntn_layer_yx)
        ntn_layer_yx = Dense(200, activation='relu')(bt_layer)
        add_layer_yx = ntn_layer_yx
        ntn_layer_yx = Dense(500, activation='relu')(ntn_layer_yx)
        ntn_layer_yx = Dense(500, activation='relu')(ntn_layer_yx)
        # ntn_layer_yx = Dense(500, activation='relu')(ntn_layer_yx)
        # ntn_layer_yx = Dense(500, activation='relu')(ntn_layer_yx)
        ntn_layer_yx = Dense(200, activation='relu')(ntn_layer_yx)
        ntn_layer_yx = Add()([ntn_layer_yx, add_layer_yx])
        output_layer_yy = Dense(self.output_dim)(ntn_layer_yy)
        output_layer_yy = LeakyReLU(name='yy_out')(output_layer_yy)
        output_layer_yx = Dense(self.output_dim)(ntn_layer_yx)
        output_layer_yx = LeakyReLU(name='yx_out')(output_layer_yx)

        self.model = Model(input_layer, [output_layer_yy, output_layer_yx])
        self.model.compile(optimizer=Adamax(self.learningRate), loss='mape', metrics=['mape', 'mse', 'mae'])
        return self.model

    def train_model(self):
        if len(AV_GPU_NUM)==0:
            self.model = self.build_model()
        else:
            strategy = tf.distribute.MirroredStrategy([f'/gpu:{i}' for i in range(len(AV_GPU_NUM)//2)])
            with strategy.scope():
                self.model=self.build_model()

        time=datetime.now()
        time=f'{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'
        csv_logger=CSVLogger(r'D:\PyPro\PolarNet\bi-network\result-log'+'//'+time+'.csv',separator=',',append=False)
        tsb=TensorBoard(r'D:\PyPro\PolarNet\bi-network\result-log'+'//'+time+'-logs',histogram_freq=5,write_graph=True)
        self.load_data()
        # batch_size better 64
        history=self.model.fit(x=self.training_structure,y=[self.training_yy,self.training_yx],validation_split=0.1,
                  batch_size=256,epochs=3500,callbacks=[csv_logger,tsb],shuffle=True)
        self.test_model()
        # yy_mse=str(int(self.yy_mse*1000000)/1000000).replace('.','_')
        # yx_mse = str(int(self.yx_mse * 1000000) / 1000000).replace('.', '_')
        yy_mape=str(int(self.yy_mape*1000000)/1000000).replace('.','_')
        yx_mape=str(int(self.yx_mape*1000000)/1000000).replace('.','_')

        # mse=str(self.model_mse).replace('.','_')
        self.model.save(r'D:\PyPro\PolarNet\bi-network\result-log'+'//'+
                        time+f'_yy_mape_{yy_mape}_yx_mape_{yx_mape}.h5')

    def test_model(self,path=None):
        if path is None:
            test_predict=self.model.predict(self.test_structure)
        else:
            try:
                from tensorflow.keras.models import load_model
            except:
                from tensorflow.keras.models import load_model
            model=load_model(path)
            test_predict = model.predict(self.test_structure)
        self.yy_mape = np.mean(np.abs(test_predict[0] - self.test_yy)/self.test_yy)*100
        self.yx_mape = np.mean(np.abs(test_predict[1] - self.test_yx)/self.test_yx)*100
        self.model_mape = int((self.yy_mape + self.yx_mape) / 2 * 1000000) / 1000000
        # self.yy_mse = np.power(test_predict[0] - self.test_yy, 2).mean()
        # self.yx_mse = np.power(test_predict[1] - self.test_yx, 2).mean()
        # self.model_mse = int((self.yy_mse + self.yx_mse) / 2 * 1000000) / 1000000

    def load_data(self, down_sample=True):
        training=np.load(r'D:\PyPro\PolarNet\bi-network\data\trainingData\training.npy')
        validation=np.load(r'D:\PyPro\PolarNet\bi-network\data\validationData\validation.npy')
        self.training_structure=training[:,:4]
        self.training_structure[:,:3]/=100
        self.validation_structure=validation[:,:4]
        self.validation_structure[:,:3]/=100
        self.training_structure=np.vstack([self.training_structure,self.validation_structure])
        self.training_yy=training[:,4:82+4]
        self.training_yx=training[:,82+4:]
        self.validation_yy=validation[:,4:82+4]
        self.validation_yx=validation[:,82+4:]
        self.training_yy = np.vstack([self.training_yy, self.validation_yy])
        self.training_yx = np.vstack([self.training_yx, self.validation_yx])
        test=np.load(r'D:\PyPro\PolarNet\bi-network\data\testData\testdata.npy')
        self.test_structure=test[:,:4]
        self.test_structure[:,:3]/=100
        self.test_yy=test[:,4:82+4]
        self.test_yx=test[:,82+4:]
        if down_sample:
            self.training_yy,self.training_yx=self.training_yy[:,::3],self.training_yx[:,::3]
            self.test_yy,self.test_yx=self.test_yy[:,::3],self.test_yx[:,::3]




if __name__=='__main__':
    down_sample=True
    dn=nn()
    model=dn.build_model()
    # history=dn.train_model()
    from plots.plotFig import plotFigre
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    model82=load_model(r'D:\PyPro\PolarNet\bi-network\result-log\2020_12_17_19_51_33_yy_mape_6_562062_yx_mape_8_204816.h5')
    model28 = load_model(r'D:\PyPro\PolarNet\bi-network\result-log\2020_12_17_21_38_9_yy_mape_3_312393_yx_mape_4_141227.h5')
    dn.load_data(down_sample=True)
    num=999
    predict_spec_82=model28.predict(dn.test_structure[num,:].reshape((1,4)))
    real_spec_82=(dn.test_yy[num,:],dn.test_yx[num,:])
    l_pre=plotFigre(predict_spec_82[1][0],xlabel='Wave points', ylabel='Spectrum',figNum=0)
    l_real=plotFigre(real_spec_82[1], xlabel='Wave points', ylabel='Spectrum', figNum=0)
    plt.legend([l_pre,l_real],['Predict', 'Observation'],fontsize=25)
    plt.title('Tyx',fontsize=25)


















