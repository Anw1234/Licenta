# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:02:25 2022

@author: oteaa
"""

# importing the required library
import cv2

# Get the video 
webcam_video_stream = cv2.VideoCapture(0)

# load the pretrained haar classifier model
face_detection_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# initialize the array variable to hold all face locations in the frame
all_face_locations = []

while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25, fy=0.25)
    # detect all face locations using the haar classifier
    all_face_locations = face_detection_classifier.detectMultiScale(current_frame_small)
    
    # looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        # splitting the tuple to get the four position values
        x,y,width,height = current_face_location
        left_pos = x
        top_pos = y
        right_pos = x+width
        bottom_pos = y+height
        # correct the coordinates location to the change the while resizing to 1/4 size inside the loop
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        # printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        # draw rectangle around the face detecteed
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,225),2)
        
    # showing the current face with rectangle drawn
    cv2.imshow("Haar Webcam",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release the video   
webcam_video_stream.release()
# close all opencv windows open
cv2.destroyAllWindows()
        