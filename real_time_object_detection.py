#!/usr/bin/python

from imutils.video import VideoStream
from imutils.video import FPS
from Input_prep import input_processing
#import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import imutils
import time
import cv2


# construct the argument parse and parse the arguments
def main(output):
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt",
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model",
		help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.6,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	#event = threading.Event()
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	args["prototxt"]="/home/zizou/Ml_proto2/MobileNetSSD_deploy.prototxt.txt"
	args["model"]="/home/zizou/Ml_proto2/MobileNetSSD_deploy.caffemodel"
# load our serialized model from disk
#print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
#print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(0.5)
	fps = FPS().start()
	#output=0
	exit=0
	cpt=0
# loop over the frames from the video stream
	while cpt<51:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
		output=0
		frame = vs.read()
		frame = imutils.resize(frame, width=720)

	# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
		net.setInput(blob)
		detections = net.forward()

	# loop over the detections
		for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
			confidence = detections[0, 0, i, 2]
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
		
			if (confidence > args["confidence"]) & (str(CLASSES[idx]) == "person"):
				confidence = detections[0, 0, i, 2]
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				
				
			
			
			
				
	# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
		if (key == ord("q")) | (exit == 1):
			break
		    
	
		output=frame
	# update the FPS counter
		fps.update()
		#print(cpt)
		cpt=cpt+1

	fps.stop()
	
	

# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	return output
	

# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
