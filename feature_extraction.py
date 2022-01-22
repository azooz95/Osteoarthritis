from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Model
from functions import *
import os
import numpy as np

new_path = "D:\\Data\\archive\\numpy"

train_y_number = new_path+"\\train_y_number.npy"
test_y_number = new_path+"\\test_y_number.npy"
train_x_n = new_path+"\\train_x_n.npy"
test_x_n = new_path+"\\test_x_n.npy"

train_y = np.load(train_y_number)
test_y = np.load(test_y_number)
train_x = np.load(train_x_n)
test_x = np.load(test_x_n)

new_path = "D:\\Data\\archive"

# ResNet50
# model_path = new_path+"\\ResNet50v2.h5"
# model = tf.keras.models.load_model(model_path)
# model = Model(inputs = model.input, outputs = model.layers[-2].output)

# x1 = model.predict(train_x,batch_size=16,verbose=1)
# print(x1.shape)
# x_t1 = model.predict(test_x,batch_size=16,verbose=1)

# VGG16
model_path = new_path+"\\VGG16v7.h5"
model = tf.keras.models.load_model(model_path)
model = Model(inputs = model.input, outputs = model.layers[-2].output)

x2 = model.predict(train_x,batch_size=16,verbose=1)
x_t2 = model.predict(test_x,batch_size=16,verbose=1)
print(x2.shape)
# VGG19
# model_path = new_path+"\\VGG19.h5"
# model = tf.keras.models.load_model(model_path)
# model = Model(inputs = model.input, outputs = model.layers[-2].output)

# x3 = model.predict(train_x,batch_size=16,verbose=1)
# x_t3 = model.predict(test_x,batch_size=16,verbose=1)
# print(x3.shape)

# combining data
# sumX = np.concatenate((x1, x2,x3), axis=0)
# sumX_t = np.concatenate((x_t1, x_t2,x_t3), axis=0)
# print("total train",sumX.shape)
# print("total test",sumX_t.shape)
# t_y = np.tile(train_y, 3)
# ts_y = np.tile(test_y, 3)
label = np.unique(train_y).tolist()
t_y = converting_to_number(train_y,label)
ts_y = converting_to_number(test_y,label)
t_y = np.array(t_y)
ts_y = np.array(ts_y)
print(t_y.shape)



# clf = RandomForestClassifier(n_estimators=600)
# clf.fit(x2,t_y)
# predidct_y = clf.predict(x_t2)
# print("Accuracy:",metrics.accuracy_score(ts_y, predidct_y))

# from sklearn import svm

# clf = svm.SVC(kernel="poly") # Linear Kernel
# clf.fit(x2,t_y)
# predidct_y = clf.predict(x_t2)
# print("Accuracy:",metrics.accuracy_score(ts_y, predidct_y))

# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=7)
# clf.fit(x2,t_y)
# predidct_y = clf.predict(x_t2)
# print("Accuracy:",metrics.accuracy_score(ts_y, predidct_y))

# from sklearn.ensemble import VotingClassifier
# rd = RandomForestClassifier(n_estimators=600)
# sv = svm.SVC(kernel="rbf") 
# kn = KNeighborsClassifier(n_neighbors=20)

# model = VotingClassifier(estimators=[('rd', rd), ('sv', sv),('kn', kn)], voting='hard')
# model.fit(x2,t_y)
# predidct_y = model.predict(x_t2)
# print("Voting Accuracy:",metrics.accuracy_score(ts_y, predidct_y))

# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier(loss="hinge", penalty="l2")
# clf.fit(x2,t_y)
# predidct_y = clf.predict(x_t2)
# print("Accuracy:",metrics.accuracy_score(ts_y, predidct_y))

# from sklearn.ensemble import BaggingClassifier
# clf = BaggingClassifier(base_estimator=RandomForestClassifier(),n_estimators=600, random_state=0)
# clf.fit(x2,t_y)
# predidct_y = clf.predict(x_t2)
# print("Accuracy:",metrics.accuracy_score(ts_y, predidct_y))


from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=5).fit(x2)
labels = gmm.predict(x_t2)
print(gmm.score(x_t2))
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

# # grid search 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
# from pprint import pprint
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]# Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)

# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
# rf.fit(x2, t_y)
# pprint(rf_random.best_params_)

