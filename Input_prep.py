#!/usr/bin/env python3

import numpy as np 
import cv2
from features_extraction import feature_extractor



def input_processing(img):
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)
    input_img_features = feature_extractor(img_input)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF   = np.reshape(input_img_features, (img_input.shape[0], -1))
    
    return input_img_for_RF
    
