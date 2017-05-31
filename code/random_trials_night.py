
# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
from time import time
from collections import Counter


# In[2]:

d = {}
bus_stop_names = ["Medavakkam", "Mogappair East", "Pari Salai"]
for it in xrange(1, len(bus_stop_names)+1):
    d[it] = bus_stop_names[it-1]


# In[3]:

def image_resize(im):
    im = sc.imresize(im, (32, 32, 3))
    return im


# In[4]:

X_train = np.array(image_resize(plt.imread("./TrainDataNight/Train1.jpg")).flatten().astype("float32"))
for i in range(2, 271):
    if i % 25 == 0:
        print "Reading train image " + str(i)
    img = plt.imread("./TrainDataNight/Train" + str(i) + ".jpg")
    X_train = np.vstack((X_train,image_resize(img).flatten().astype("float32")))
print len(X_train)


# In[5]:

X_train = X_train / 255.0


# In[6]:

y_train = []
for bus_stop in range(1, 4):
    for train_images in range(90):
        y_train.append(bus_stop)
y_train = np.array(y_train)


# In[7]:

def compute_distances(X_train, X):
    X = np.array(X)
    X = X.astype("float32")
    X /= 255.0
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((np.square(X).sum(axis=1, keepdims=True)) - (2*X.dot(X_train.T)) + (np.square(X_train).sum(axis=1)))
    return dists


# In[8]:

for n_images in [2, 3, 5, 7, 9, 11]:
    t1 = time()
    for iterx in range(100):
        print "Iter: " + str(iterx)
        class_name = np.random.randint(1, 4)
        print "Class name: " + str(d[class_name]), "; No. of Images: " + str(n_images)
        print
        X_test = np.zeros((n_images, np.prod(X_train.shape[1:])))
        x = 0
        if class_name == 1:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestDataNight/Medavakkam/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 2:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestDataNight/Mogappair East/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 3:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestDataNight/Pari Salai/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1

        dis = compute_distances(X_train, X_test)

        for k in range(1, 6):
            count = 0
            correct_classes = []
            for i in range(dis.shape[0]):
                l = y_train[np.argsort(dis[i, :])].flatten()
                closest_y = l[:k]
                correct_classes.append(Counter(closest_y).most_common(1)[0][0])
            correct_classes = np.array(correct_classes)
            #print l[:10]
            for v in range(correct_classes.shape[0]):
                if correct_classes[v] == class_name:
                    count += 1
	    print "k = " + str(k)
            print "Predicted as: ",
            for cc in correct_classes:
                print d[cc] + ", ",
            print
            print "Groundtruth : " + str(d[class_name])
            accuracy = float(count) / dis.shape[0]
            print "Accuracy: " + str(accuracy)
            print
        #print
    t2 = time()
    print "Time Taken: " + str(t2-t1)
    print
