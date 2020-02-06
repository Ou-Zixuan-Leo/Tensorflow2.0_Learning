# @FileName: Linear_Mul_nn.py
# @Author : Ou Zixuan
# @Time : 2020/2/6 

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''PART1'''
data = pd.read_csv('Advertising.csv') #广告数据集

data.head()

plt.scatter(data.TV,data.sales)

plt.scatter(data.radio,data.sales)

plt.scatter(data.newspaper,data.sales)


x = data.iloc[:,1:-1]  #提取所有行，第一列至倒数第二列的信息
y = data.iloc[:,-1]    #提取所有行，最后一列的信息

'''PART2'''
#建立具有一层hidden layer和一层output layyer的model
model = tf.keras.Sequential(
        [
        tf.keras.layers.Dense(10,input_shape=(3,),activation = 'relu'), 
        tf.keras.layers.Dense(1)
        ]
)

model.summary()    #理解维度！！！，第L层的W和b的个数与前一层的输入特征数及本层的units有关

model.compile(optimizer='adam',    #设置优化方法
              loss = 'mse',        #损失函数定义 均方差 mean square error
)

history = model.fit(x,y,epochs=200)  #训练模型

'''PART3'''
test = data.iloc[:10,1:-1]   #这里直接用原有的数据作为测试样本，检验模型
model.predict(test)

#绘图
plt.plot(history.epoch,history.history.get('loss'),label='train_loss') 
plt.legend()   #用于插入图例





