
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Model
from functions import *
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
new_path = "D:\\Data\\archive\\numpy"
test_x_n = new_path+"\\test_x_n2.npy"

test_y_hot = new_path+"\\test_y_hot2.npy"
test_x = np.load(test_x_n)
test_y = np.load(test_y_hot)
print(test_x.shape)
print(test_y.shape)
new_path = "D:\\Data\\archive"
model_path = new_path+"\\VGG16_cross_validation_v6.h5"
model = tf.keras.models.load_model(model_path)
# model = Model(inputs = model.input, outputs = model.layers[-2].output)
metrics = [tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
model.compile(loss = 'CategoricalCrossentropy',optimizer=Adam(learning_rate=0.00001, decay=0.0001),metrics=metrics)
history = model.evaluate(test_x, test_y,verbose=1,batch_size=16)
print(history.history)