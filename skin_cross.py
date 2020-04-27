import cv2
import os
import sys
from tqdm import tqdm  
import numpy as np
import csv
#import argparse
import random


# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import pickle
from sklearn.metrics import confusion_matrix

TRAIN_DIR= "/home/roozbeh/Skin/"
Labels = "/home/roozbeh/HAM10000_metadata.csv"
# ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

def target_choose(value):
    if value=='bkl':
        out = 0
    if value=='nv':
        out = 1
    if value=='df':
        out = 2
    if value=='mel':
        out = 3
    if value=='vasc':
        out = 4
    if value=='bcc':
        out = 5
    if value=='akiec':
        out = 6
    return out
X_training_data = []
target = []
IMG_SIZE = 50

with open('train2.pickle', 'rb') as f:
    X_train, y_train = pickle.load(f)


#### More complesx model ########### #################
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        
        # first (and only) set of FC => RELU layers

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        
        # use a *softmax* activation for single-label classification
        # and *sigmoid* activation for multi-label classification
        model.add(Dense(classes))
        model.add(Activation(finalAct))
 
        # return the constructed network architecture
        return model
		
# Done with model
import matplotlib
##matplotlib.use("Agg")
 
##import matplotlib.pyplot as plt
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 180
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (IMG_SIZE, IMG_SIZE, 3)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(y_train)

aug = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,height_shift_range=0.2, horizontal_flip=True)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    ##    plt.show()
    plt.savefig('images/plot1.png', format='png')
	
import time
start_time = time.time()

#model = SmallerVGGNet.build(
##      width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
#      depth=IMAGE_DIMS[2], classes=7,
#      finalAct="sigmoid")
 
# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience= 40)
logger = callbacks.CSVLogger('images/log.csv', separator=',', append=False)
callback=[early_stop, logger]

import numpy as np
speciv = []
sensitiv = []
accu = []
observ = []
accuy =[]
X= X_train
y= y_train
X_train = None
y_train = None
n= X.shape[0]
from sklearn.model_selection import KFold
#kf = KFold(n, n_splits=10, shuffle=True)
kf = KFold(n_splits=10, shuffle=True)
issues = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

#for iteration, data in enumerate(kf, start=1):
for train_index, test_index in kf.split(X):

	trainX = X[train_index]
	testX = X[test_index]
	trainY = y[train_index] 
	testY = y[test_index]

	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=7, finalAct="sigmoid")
	# Compile model
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])	# Fit the model
	model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, callbacks= callback,verbose=2)
	# evaluate the model
	y_pred = model.predict(testX)
	Y_predict = np.round(y_pred)
	Y_predict2 = Y_predict.reshape(Y_predict.shape[0]*Y_predict.shape[1],)
	testY2 = testY.reshape(testY.shape[0]*testY.shape[1],)
	testY3 = [int(numeric_string) for numeric_string in testY2]
	Y_predict3= [int(numeric_string) for numeric_string in Y_predict2]
	cnf_matrix = confusion_matrix(testY3, Y_predict3)
	Sensitivity= cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])*100
	Specificity= cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[0,1])*100
	accy = (cnf_matrix[1,1]+cnf_matrix[0,0])/np.sum(cnf_matrix)*100

	for i in range(7):
	       accu.append(np.sum([int(numeric_string) for numeric_string in testY[:,i]]==Y_predict[:,i])/Y_predict.shape[0]*100)
	       observ.append(np.sum([int(numeric_string) for numeric_string in testY[:,i]]))
	#accu.append(Accuracy)
	speciv.append(Specificity)
	sensitiv.append(Sensitivity)
	accuy.append(accy)
	print("Accuracy= ", accy)	

rsd = (time.time() - start_time)/3600.0

print("--- %s hours ---" %rsd)

with open('cv.pickle', 'wb') as f:
    pickle.dump([accuy, accu, speciv, sensitiv, observ], f, protocol=4)	



#issues = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

'''
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target_labels)
'''
