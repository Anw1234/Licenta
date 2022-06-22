# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:05:52 2022

@author: oteaa
"""

# importing the required library
import cv2

# loading the image to detect
image_to_detect = cv2.imread('media/trump-and-benjamin.jpg')
cv2.imshow("Initial image",image_to_detect)

#  load the pretrained haar classifier model
face_detection_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# detect all face locations using the haar classifier
all_face_locations = face_detection_classifier.detectMultiScale(image_to_detect)

# print the number of faces detected
print('There are {} number of faces in this image'.format(len(all_face_locations)))

# looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    # splitting the tuple to get the four position values
    x,y,width,height = current_face_location
    # start coordinates
    left_x, left_y = x,y
    # end coordinates
    right_x, right_y = x+width, y+height
    # printing the location of current face
    print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
    # slicing the current face from main image
    current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
    # showing the current face with dynamic title
    cv2.imshow("Face Number "+str(index+1),current_face_image)
    # draw bounding box around the faces
    cv2.rectangle(image_to_detect, (left_x,left_y),(right_x,right_y),(0,255,0),2)
    
# show the image
cv2.imshow("Haar Cascade image",image_to_detect)
# keep the window waiting until we press a key
cv2.waitKey(0)
# close all windows
cv2.destroyAllWindows
    
    
    