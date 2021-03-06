 # -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:02:25 2022

@author: oteaa
"""

# importing the required libraries
import cv2
import face_recognition
import sqlite3
# Import datetime class from datetime module
from datetime import datetime

conn = sqlite3.connect('facerecdb.db',detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

# Get the webcam 
webcam_video_stream = cv2.VideoCapture(0)

# load the sample images and get the 128 face embeddings from them
benjamin_image = face_recognition.load_image_file('media/Benjamin.jpg')
benjamin_face_encodings = face_recognition.face_encodings(benjamin_image)[0]

trump_image = face_recognition.load_image_file('media/Trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

andrei_image = face_recognition.load_image_file('media/Andrei.jpg')
andrei_face_encodings = face_recognition.face_encodings(andrei_image)[0]

# save the encodings and the corresponding labels in separate arrays in the same order
known_face_encodings = [benjamin_face_encodings,trump_face_encodings,andrei_face_encodings]
known_face_names = ["Benjamin Netanyahu","Donald Trump","Andrei Otea"]

# initialize the array variable to hold all face locations , encodings and names in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    # resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25, fy=0.25)
    # arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model="hog")
    
    # detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)

    # looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        # splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        # change the position maginitude to fit the actual size video frame
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        
        # find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        # string to hold the label
        name_of_person = 'Unknown face'
        # check if the all_matches have at least one item
        # if yes,get the index number of face that is located in the first index of all_matches
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            # printing the location of current face
            print('Found face of {} at top:{},right:{},bottom:{},left:{}'.format(name_of_person,top_pos,right_pos,bottom_pos,left_pos))
            # returns current date and time
            temp = datetime.now()
            now = temp.strftime("%a %b %d %H:%M:%S %Y")
            print("now = ", now)
            
        def create_table():
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS FaceRecgIntel (Name TEXT, Top REAL, Right REAL, Bottom REAL, Left REAL,DateTime TEXT)')
            c.close()
                
        def data_entry():
            c = conn.cursor()
            c.execute("INSERT INTO FaceRecgIntel (Name, Top, Right, Bottom, Left, DateTime) VALUES(?, ?, ?, ?, ?, ?)", (name_of_person, top_pos, right_pos, bottom_pos, left_pos, now))
            conn.commit()  
            c.close()
                
        create_table()
        data_entry()
 
        # draw a rectangle around the face
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        # display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,name_of_person,(left_pos,bottom_pos),font,0.5,(0,0,0),1)
        
    # display the imaage
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release the stream and camera    
webcam_video_stream.release()
# close all opencv windows open
cv2.destroyAllWindows()
        