import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

akshay_image = face_recognition.load_image_file("Photos/Akshay_kumar.jpg")
akshay_encoding = face_recognition.face_encodings(akshay_image)[0]

modi_image = face_recognition.load_image_file("Photos/Narendra_modi.jpg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

priyanka_image = face_recognition.load_image_file("Photos/Priyanka_chopra.jpg")
priyanka_encoding = face_recognition.face_encodings(priyanka_image)[0]

tata_image = face_recognition.load_image_file("Photos/Ratan_tata.jpg")
tata_encoding = face_recognition.face_encodings(tata_image)[0]

kohli_image = face_recognition.load_image_file("Photos/Virat_kohli.jpg")
kohli_encoding = face_recognition.face_encodings(kohli_image)[0]

known_face_encoding = [ akshay_encoding, modi_encoding, priyanka_encoding, tata_encoding, kohli_encoding]

known_face_names = ["akshay", "modi", "priyanka", "tata", "kohli"]

students = known_face_names.copy()

face_location = []
face_encoding = []
face_names = []
s=True

now=datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame =cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encoding = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
