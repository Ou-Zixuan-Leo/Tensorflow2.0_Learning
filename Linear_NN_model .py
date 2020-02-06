
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
'''
get_ipython().run_line_magic('matplotlib', 'inline')
#此语句用在jupyter notebook下，使图像可以生成，当使用pycharm的时候不需要
'''

data = pd.read_csv('Income1.csv') #注意路径最前面要加上一点 ./,也可以不加

plt.scatter(data.Education,data.Income)

x = data.Education
y = data.Income

model = tf.keras.Sequential()  #开始建立一个序列模型

model.add(tf.keras.layers.Dense(1,input_shape=(1,)))  #维度（输出，输入）
#推测 输出维度k1对Trainable params的影响为2*k1，因为softmax会使W和b参数成倍数增长
#而input_shape的维度k2对Trainable params为1+k2，因为每多一个维度相当于多一个输入特征w

model.summary()


'''以上的代码建立好了一个model，以下开始设置model的优化方法1'''
model.compile(optimizer='adam',   #优化方法
              loss = 'mse'        #损失函数定义 均方差 mean square error
)

history = model.fit(x,y,epochs=5000)  #开始训练并记录   

model.predict(pd.Series([20]))

model.predict(x)





