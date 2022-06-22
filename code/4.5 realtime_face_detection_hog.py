# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:02:25 2022

@author: oteaa
"""

# importing the required library
import cv2
import dlib

# Get the video 
video_stream = cv2.VideoCapture(0)

# load the pretrained HOG model
face_detection_classifier = dlib.get_frontal_face_detector()

# initialize the array variable to hold all face locations in the frame
all_face_locations = []

while True:
    # get the current frame from the video stream as an image
    ret, current_frame = video_stream.read()
    # create a grayscale image to pass into the dlib HOG detector
    current_frame_to_detect_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame_to_detect_gray,(0,0),fx=0.25, fy=0.25)
    # detect all face locations using the HOG classifier
    all_face_locations = face_detection_classifier(current_frame_small,1)
    
    # looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        
        # start and end coordinates
        left_pos, top_pos, right_pos, bottom_pos = current_face_location.left(),current_face_location.top(),current_face_location.right(),current_face_location.bottom()
      
        # correct the co-ordinate location to the change the while resizing to 1/4 size inside the loop
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        # printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        # draw rectangle around the face detecteed
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,225),2)
        
    # showing the current face with rectangle drawn
    cv2.imshow("Real-time face detection using HOG",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release the video   
video_stream.release()
# close all opencv windows open
cv2.destroyAllWindows()
        