#mnist数据集分类例子，包括数字序列形式的NN、独热编码形式的NN、独热编码形式的多层NN，共三个例子

import tensorflow as tf  
import pandas as pd
import matplotlib.pyplot as plt


(train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()

#数据查看（维度、特征）
#数据预处理（归一化，扁平化处理）
#模型建立
#模型优化方式
#模型训练
#模型预测 /评估  model.evaluate 

train_image.shape

train_label.shape

test_image.shape

plt.imshow(train_image[1])

train_image = train_image/255 
test_image = test_image/255   #对灰度数据进行归一化处理



#PART1 数字序列
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #对二维数据进行扁平化处理，使神经网络的Dense可用
model.add(tf.keras.layers.Dense(128,activation='relu'))  #第一层hidden layer
model.add(tf.keras.layers.Dense(10,activation='softmax'))  #10个分类，因此最后一层为10个单元，且为softmax的激活

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', #数字序列的交叉熵
              metrics = ['acc']
)


model.fit(train_image,train_label,epochs = 10)

model.evaluate(test_image,test_label) #测试集模型验证



#PART2 独热编码
train_label_onehot = tf.keras.utils.to_categorical(train_label)  #数字序列转独热编码

train_label_onehot[-1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #对二维数据进行扁平化处理，使神经网络的Dense可用
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))  #10个分类，因此最后一层为10个单元，且为softmax的激活

model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
              loss = 'categorical_crossentropy',   #独热编码交叉熵
              metrics = ['acc']
)

model.fit(train_image,train_label_onehot,epochs = 10)

test_label_onehot = tf.keras.utils.to_categorical(test_label)

model.evaluate(test_image,test_label_onehot)



