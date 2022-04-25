# %reset
import numpy as np 
from mnist import MNIST # require `pip install python-mnist` #get dataset form MNIST database
# https://pypi.python.org/pypi/python-mnist/

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time
import os
print(os.listdir('/'))
# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/

#get the dataset (training data, test data)
mndata = MNIST('/Documents/ML_sample/K_nearest_neighbours') # path to your MNIST folder 
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)
print("Get data done!")
start_time = time.time() #mark the starting time
clf = neighbors.KNeighborsClassifier(n_neighbors = 5, weights="distance", p = 2) #call model
clf.fit(X_train, y_train) #fit traning data
print("Training done!")
y_pred = clf.predict(X_test)
end_time = time.time() #mark the ending time
print("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("Running time: %.2f (s)" % (end_time - start_time))