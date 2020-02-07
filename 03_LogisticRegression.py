# @FileName: Linear_Mul_nn.py
# @Author : Ou Zixuan
# @Time : 2020/2/6

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("dataset/credit-a.csv")
data.head()

x = data.iloc[:,:-1]
y = data.iloc[:,-1].replace(-1,0)    #将-1的label替换成0



model = tf.keras.Sequential([                                            #注意使用方括号
        tf.keras.layers.Dense(8,input_shape=(15,),activation = 'relu'),  #输入层
        tf.keras.layers.Dense(16,activation='relu'),                     #中间隐藏层 
        tf.keras.layers.Dense(1,activation='sigmoid')                    #输出层
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc']
              
)

histiory = model.fit(x,y,epochs = 100)

plt.plot(histiory.epoch,histiory.history.get('acc'))
