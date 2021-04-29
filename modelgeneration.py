#!/usr/bin/env python3

import numpy as np 
import glob
import cv2
import os
from features_extraction import feature_extractor
#from plots import plot_graph
import pandas as pd


#print(os.listdir("dataset/"))

#Resize images to
SIZE = 128

#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
for directory_path in glob.glob("dataset/train/*"):
    label = directory_path.split("/")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) #Reading color images
        img = cv2.resize(img, (SIZE, SIZE)) #Resize images
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #Optional step. Change BGR to RGB
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)



test_images = []
test_labels = [] 
for directory_path in glob.glob("dataset/validation/*"):
    fruit_label = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #Optional
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)




from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0







#Extract features from training images
image_features = feature_extractor(x_train)
c=list(image_features)
aux=list(image_features.columns)

#Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 20, random_state = 0)
RF_model.fit(X_for_RF, y_train)


#*************************************************
#popularity = []
#print('%f'%np.average(RF_model.feature_importances_[:128*128*3]))
#popularity.append('%f'%np.sum(RF_model.feature_importances_[:128*128*3]))
#for i in range(1,len(c)):
    #print('%f'%np.average(RF_model.feature_importances_[128*128*3*i:128*128*3*(i+1)]))
    #popularity.append('%f'%np.sum(RF_model.feature_importances_[128*128*3*i:128*128*3*(i+1)]))

#myvar = pd.Series(popularity, index = aux,dtype='float')
#myvar=myvar.sort_values(ascending=True)

#plot_graph(myvar.index,myvar.values*100) 
#************************************************ 




 

#Predict on test
test_prediction = RF_model.predict(test_for_RF)
#Inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))



    




import pickle

filename = "pi_model"
pickle.dump(RF_model, open(filename, 'wb'))






#import random
#n=random.randint(0, x_test.shape[0]-1) #Select the index of image to be loaded for testing
#img = x_test[n]
#plt.imshow(img)
#Predict
#input_img_for_RF= input_processing(img)
#img_prediction = mymodel.predict(input_img_for_RF)
#img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
#print("The prediction for this image is: ", img_prediction)
#print("The actual label for this image is: ", test_labels[n])



