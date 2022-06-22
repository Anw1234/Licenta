# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:38:52 2022

@author: oteaa
"""

# importing the required libraries
import cv2
import face_recognition

# loading the image to detect
original_image = cv2.imread('media/trump-and-benjamin2.jpg')

# load the sample images and get the 128 face embeddings from them
benjamin_image = face_recognition.load_image_file('media/Benjamin.jpg')
benjamin_face_encodings = face_recognition.face_encodings(benjamin_image)[0]

trump_image = face_recognition.load_image_file('media/Trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

# save the encodings and the corresponding labels in separate arrays in the same order
known_face_encodings = [benjamin_face_encodings,trump_face_encodings]
# array to hold the labels
known_face_names = ["Benjamin Netanyahu","Donald Trump"]

# load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('media/trump-and-benjamin2.jpg')

# detect all faces in the image
# arguments are image,no_of_times_to_upsample, model
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')
# detect face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)


# print the number of faces detected
print('There are {} number of faces in this image'.format(len(all_face_locations)))

# looping through each face location and the face encodings found in the unknown image
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    # splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    # find all the matches and get the list of matches
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    # string to hold the label
    name_of_person = 'Unknown face'
    # check if the all_matches have at least one item
    # if yes,get the index number of face that is located in the first index of all_matches
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
    # draw a rectangle around the face
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    
    # display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image,name_of_person,(left_pos,bottom_pos),font,0.5,(0,0,0),1)
    
    # display the imaage
    cv2.imshow("Faces Identified",original_image)
    
    
    
    
    