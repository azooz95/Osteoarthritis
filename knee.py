import os
from sklearn import metrics 
import tensorflow as tf 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D,LSTM
import cv2 as cv
import pandas as pd
from tensorflow.keras.optimizers import Adam
from numba import cuda 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.backend import dropout
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from functions import *
from itertools import chain
from tensorflow.keras.utils import plot_model
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], True)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

def upload_list(path):
  total = list(np.concatenate([[path+"/"+j.split('(')[0]+'/'+j for j in os.listdir(path+'/'+x)] for x in os.listdir(path)]).flat)
  return total

def extracting_labels(paths):
  labels = [x.split('/')[-1].split('(')[0] for x in paths]
  return labels

def converting_to_number(labels, label):
  labels = [label.index(x) for x in labels]
  return labels
  
def hot_encoding(labels):
  return tf.keras.utils.to_categorical(labels)

def loading_data(paths):
  data = [np.asarray(Image.open(x).resize((224,224))) for x in paths]
  return data

def fine_tune(model,n):
  model.trainable = True
  if n != 0:
    for layer in model.layers[:n]:
      layer.trainable =  False
  return model
  
def gussain_blur_with_weited(img,wight,target=224):
  im = cv.GaussianBlur(img,(0,0),target/60)
  image = cv.addWeighted(img,wight,im,-wight, 128)
  return image

def histogram_equlization(t):
  data = []
  for i in t:
    if np.mean(i)>=200:
      clahe = cv.createCLAHE(clipLimit=15, tileGridSize=(8,8))
      print(i,np.mean(i))
    elif np.mean(i)>=180 and np.mean(i)<200:
      clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8,8))
    elif np.mean(i)>=160 and np.mean(i)<180:
      clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    else:
      clahe = cv.createCLAHE(clipLimit=0.05, tileGridSize=(8,8))
    x = clahe.apply(i)
    x = np.repeat(x[..., np.newaxis], 3, -1)
    data.append(x)
  c = np.array(data)
  return c

def ensamble(paths):
  all_models = []
  for model_path in paths:
      model = tf.keras.models.load_model(model_path)
      all_models.append(model)
      print('loaded:', model_path)

  for index,model in enumerate(all_models):
    model._name = 'mod_'+str(index)
    for layer in model.layers:
      layer._name = layer._name+str(index)
      layer.trainable = False

  ensemble_visible = [model.input for model in all_models]
  ensemble_outputs = [model.output for model in all_models]
  merge = tf.keras.layers.concatenate(ensemble_outputs)
  output = tf.keras.layers.Dense(5, activation='softmax')(merge)
  model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
  return model

# path_train = "D:\\Data\\archive\\aug_data\\train"
# path_test = "D:\\Data\\archive\\aug_data\\test"
# path_validation = "D:\\Data\\archive\\val"
# path_auto = "D:\\Data\\archive\\auto_test"

# new_path = "D:\\Data\\archive\\numpy"

# train_y_number = new_path+"\\train_y_numberV2"
# test_y_number = new_path+"\\test_y_numberV2"
path = "D:\\numpy"
train_xp = path + "\\train_x_n.npy"
test_xp = path + "\\test_x_n.npy"
train_yp = path+"\\train_y_hot.npy"
test_yp = path+"\\test_y_hot.npy"
train_y_p = path + "\\train_y_number.npy"
test_y_p = path + "\\test_y_number.npy"

train_x = np.load(train_xp)
test_x = np.load(test_xp)
train_y = np.load(train_yp)
test_y = np.load(test_yp)
train_y_ = np.load(train_y_p)
test_y_ = np.load(test_y_p)

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape, train_y_.shape, test_y_.shape)
# train_y_ = np.load(train_y_number)
# train_y_ = np.load(test_y_number)

# train_paths = upload_list(path_train)
# test_paths = upload_list(path_test)
# valid = upload_list(path_validation)
# auto = upload_list(path_auto)

# label = os.listdir(path_train)

# train_paths = shuffle(train_paths,random_state=52)
# test_paths = shuffle(test_paths,random_state=52)
# valid_paths = shuffle(valid,random_state=52)
# auto_paths = shuffle(auto,random_state=52)

# total_path = train_paths+test_paths+auto_paths+valid_paths
# d = {'Severe':[],'Moderate':[],'Minimal':[],'Doubtful':[], "Healthy knee":[]}
# for i in total_path:
#   saved_data_train = "D:\\Data\\archive\\aug_data1\\dataset"
#   if not os.path.isdir(saved_data_train):
#     os.mkdir(saved_data_train)
#   if len(d[i.split('/')[-1].split('.')[0].split('(')[0]]) == 100:
#     continue
#   file = i.split('/')[-1].split('.')[0].split('(')[0]
#   if not os.path.isdir(saved_data_train+'\\'+ file):
#     os.mkdir(saved_data_train+'\\'+ file)

#   path_of_each_class = saved_data_train+'\\'+ file
#   img = Image.open(i)
#   number = len(d[i.split('/')[-1].split('.')[0].split('(')[0]])
#   img.save(path_of_each_class+'\\'+file+"("+str(number)+'.png')
#   d[i.split('/')[-1].split('.')[0].split('(')[0]].append(i)
# total_path = [d[i] for i in d]
# total_path = list(chain.from_iterable(total_path))
# total_path = shuffle(total_path,random_state=52)

# precentage = round(len(total_path) * 0.2)
# train_paths = total_path[precentage:]
# test_paths = total_path[:precentage]

# print(len(train_paths))
# train_y = extracting_labels(train_paths)
# test_y = extracting_labels(test_paths)

# train_y_ = converting_to_number(train_y, label)
# test_y_ = converting_to_number(test_y, label)

# np.save(train_y_number, train_y)
# np.save(test_y_number, test_y)

# train_y_hot = new_path+"\\train_y_hotV2"
# test_y_hot = new_path+"\\test_y_hotV2"

# train_y = np.load(train_y_hot)
# test_y = np.load(test_y_hot)
# train_y = hot_encoding(train_y_)
# test_y = hot_encoding(test_y_)

# print(train_y.shape)
# np.save(train_y_hot, train_y)
# np.save(test_y_hot, test_y)

# train_x_n = new_path+"\\train_x_nV2"
# test_x_n = new_path+"\\test_x_nV2"


# train_x = np.load(train_x_n)
# test_x = np.load(test_x_n)
# train_y = np.load(train_y_hot)
# test_y = np.load(test_y_hot)
# train_x = loading_data(train_paths)
# test_x = loading_data(test_paths)
# train_x = np.array(train_x)
# test_x = np.array(test_x)


# train_x = histogram_equlization(train_x)
# test_x = histogram_equlization(test_x)
print(train_x.shape)
if test_x.shape[-1] !=3:
    train_x =  np.repeat(train_x[..., np.newaxis], 3, -1)
    test_x = np.repeat(test_x[..., np.newaxis], 3, -1)

# train_x = np.load(train_x_n)
# test_x = np.load(test_x_n)

kernel = np.array([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])

# train_x = train_x*1/255
# a = train_x[0]
plt.imshow(train_x[0])
plt.title('orginial')
plt.show()
train_x = np.array([cv.filter2D(i,-1,kernel) for i in train_x])
plt.imshow(train_x[0])
plt.title("sharp")
plt.show()

# test_x = np.array([cv.filter2D(i,-1,kernel) for i in test_x])
# np.save(train_x_n, train_x)
# np.save(test_x_n, test_x)
# plt.subplot(1,2,1)
# plt.imshow(train_x[0])
# plt.subplot(1,2,2)
# plt.imshow(a)
# plt.show()

# train_x = [np.uint8((np.random.normal(0.1,0.005 ** 0.5,x.shape)+x)*255) for x in train_x]
# train_x = np.array(train_x)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes= np.unique(train_y_),
                                                 y= train_y_)
class_weights = dict(enumerate(class_weights))
print(class_weights)
# model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)

# x = tf.keras.layers.Conv2D(filters= 36, kernel_size= 3, padding= "same")(model.output)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 36, kernel_size= 3, padding= "same")(model.output)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, padding= "same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, padding= "same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)
# x = GlobalAveragePooling2D()(x)

# x = Dense(64,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
# x = Dropout(0.2)(x)
# x = Dense(32,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
# x = tf.keras.layers.Conv2D(filters= 1024, kernel_size= 3, padding= "same")(model.output)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 256, kernel_size= 3, padding= "same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 64, kernel_size= 3, padding= "same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, padding= "same")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Activation("relu")(x)

# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# predction = Dense(5,activation='softmax')(x)

model = tf.keras.applications.VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = Dense(64,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = Dropout(0.5)(x)
x = Dense(64,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = Dense(32,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
x = Dropout(0.2)(x)
x = Dense(32,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
# x = Dense(128,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
# x = Dense(64,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
# x = Dropout(0.2)(x)
# x = Dense(32,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
predction = Dense(5,activation='softmax')(x)
new_path = "D:\\Data\\archive"
# paths = [new_path+"\\VGG19v2.h5", new_path+"\\VGG16v7.h5"]
# model = ensamble(paths)
tf.random.set_seed(52)

print(model.summary())
model = fine_tune(model,0)
checkpoint_path = "D:\\Data\\archive\\VGG16_extr.h5"
callback = [
          ModelCheckpoint(checkpoint_path,
                          monitor = 'val_accuracy',
                          verbose = 1,
                          save_weights_only=True,
                          save_best_only = True,
                          mode="max"),
        EarlyStopping(monitor='val_loss',
                      patience=5,
                      verbose=0),
        ReduceLROnPlateau(monitor='val_loss',
                          patience=3,
                          verbose=1)
]
metrics = [tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
# model = Model(inputs=model.input, outputs=predction)
model.compile(loss = 'CategoricalCrossentropy',optimizer=Adam(learning_rate=0.00001, decay=0.0001),metrics=metrics)
batch_size = 16
epochs = 100
new_path = "D:\\Data\\archive"
model_path = new_path+"\\VGG16_extr.h5"
# number_folds = 10
# if number_folds:
#   kf = KFold(n_splits=number_folds,shuffle=True, random_state=52)
#   data_input = train_x
#   data_target = train_y
#   for index,(train, test) in enumerate(kf.split(data_input,data_target)):
#       print("folds: " + str(index))
#       history = model.fit(data_input[train],data_target[train],
#                                   validation_data = (data_input[test],data_target[test]),
#                                   verbose = 1,
#                                   epochs = epochs,
#                                   batch_size= batch_size,
#                                   callbacks=[callback])
#       pf = pd.DataFrame(history.history)
#       history_result = new_path+"\\VGG16_cv"+str(index)+"v6.csv"
#       with open(history_result,mode='w') as f:
#         pf.to_csv(f)
#   model.save(model_path)
print(train_y.shape)
print(train_x.shape)
print(test_y.shape)
print(test_x.shape)
# tf.compat.v1.disable_eager_execution()
plot_model(model, to_file='VGG16_extr.png', show_shapes=True)

history = model.fit([train_x,train_x], train_y,
                    validation_data = ([test_x,test_x],test_y),
                    verbose = 1,
                    epochs = epochs,
                    batch_size= batch_size,
                    callbacks=[callback])

model_path = new_path+"\\ensabmle.h5"
history_result = new_path+"\\ensabmle.csv"
pf = pd.DataFrame(history.history)
with open(history_result,mode='w') as f:
  pf.to_csv(f)
# history_result = "/content/drive/MyDrive/knee/DenseNet169.csv"
# pf = pd.DataFrame(history.history)
# with open(history_result,mode='w') as f:
#   pf.to_csv(f)
model.save(model_path)
