#!/usr/bin/python

from imutils.video import VideoStream
from imutils.video import FPS
from Input_prep import input_processing
from real_time_object_detection import main
#import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import imutils
import time
import cv2



if __name__ == "__main__":	
	aux=0
	filename = "pi_model"
	RF_model = pickle.load(open('/home/zizou/Ml_proto1/'+filename, 'rb'))

#image processing
	aux=main(aux)
	time.sleep(1)




	input_img=input_processing(aux)



#Scaling
	test_labels = np.array(['chicAndclassic','casual','sporty'])
	from sklearn import preprocessing
	le = preprocessing.LabelEncoder()
	le.fit(test_labels)
	test_labels_encoded = le.transform(test_labels)



#prediction
	img_prediction = RF_model.predict(input_img)
	img_prediction = le.inverse_transform([img_prediction])






	print(img_prediction[0])
