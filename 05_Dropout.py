
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




(train_image,train_label),(test_image,test_label) = tf.keras.datasets.fashion_mnist.load_data()


train_image.shape


train_image = train_image/255
test_image = test_image/255


train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)

#边训练边测试，最后生成相应图像
train_label_onehot.shape



#PART1 多层神经网络模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()


model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['acc']
             )

history = model.fit(train_image,train_label_onehot,epochs=10,
         validation_data = (test_image,test_label_onehot)   #同时进行验证，并打印验证集数据
)

plt.plot(history.epoch,history.history.get('loss'),label='train_loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.legend()   #用于插入图例


plt.plot(history.epoch,history.history.get('acc'),label='train_acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()   #用于插入图例



#PART2 使用Dropout后的多层神经网络模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))                    #dropout比例为0.3，表示30%的单元被随机丢弃
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['acc']
             )

history = model.fit(train_image,train_label_onehot,epochs=10,
         validation_data = (test_image,test_label_onehot)
)

plt.plot(history.epoch,history.history.get('acc'),label='train_acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()   #用于插入图例



