import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import  TensorBoard

tb_cb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
            write_graph=True,
            write_grads=True,
            write_images=True)

model = Sequential()
activation = 'tanh'
model.add(Dense(9, input_dim=1, activation=activation))
model.add(Dense(1, activation="softplus"))

model.compile(loss='logcosh',
              optimizer='adam',
              metrics=['mse'])

# from ann_visualizer.visualize import ann_viz
#
# ann_viz(model, title="My first neural network")

SIZE = 100

rng = random.sample(list(range(1, SIZE*100)), SIZE)
np.random.shuffle(rng)
x = np.array(rng).reshape(-1,1)
y = np.array([xi**2 for xi in x])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

scalarX = MinMaxScaler()
scalarX.fit(x)
x_train = scalarX.transform(x_train)

scalarY = MinMaxScaler()
scalarY.fit(y)
y_train = scalarY.transform(y_train)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=1500, callbacks=[tb_cb])

x_test = scalarX.transform(x_test)
y_test = scalarY.transform(y_test)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(loss_and_metrics)

predict_x = np.array(range(1, SIZE*100))
predict = scalarX.transform(predict_x.reshape(-1, 1))
classes = model.predict(predict)
classes = scalarY.inverse_transform(classes)

import matplotlib.pyplot as plt

plt.plot(predict_x, predict_x**2, 'r--', predict_x, classes.ravel(), 'b--')
plt.show()