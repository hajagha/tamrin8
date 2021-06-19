# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:08:34 2021

@author: amir
"""


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense



dataset = loadtxt('diabet.csv', delimiter=',')



X = dataset[:,0:8]
y = dataset[:,8]


model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] ) 
history=model.fit(X, y, epochs=50, batch_size=10)




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model.fit(X, y, epochs=50, batch_size=10)


import matplotlib.pyplot as plt


plt.plot(history.history['loss'], label="Training loss")
plt.plot(history.history['accuracy'], label="accuracy")
plt.legend()
plt.title("Training vs accuracy  with ADAM Optimizer")
plt.show()



plt.plot(history2.history['loss'], label="Training loss")
plt.plot(history2.history['accuracy'], label="accuracy")
plt.legend()
plt.title("Training vs accuracy with SGD Optimizer")
plt.show()