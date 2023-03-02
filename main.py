import face_Recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture = cv2.VideoCapture(0)

alex_image = face_Recognition.load_image_file("photos/Alex.jpg")
alex_encoding = face_Recognition.face_encodings(alex_image)[0]

den_image = face_Recognition.load_image_file("photos/Dennial.jpg")
den_encoding = face_Recognition.face_encodings(den_image)[0]

el_image = face_Recognition.load_image_file("photos/Elon.jpg")
el_encoding = face_Recognition.face_encodings(el_image)[0]

jel_image = face_Recognition.load_image_file("photos/Jellon.jpg")
jel_encoding = face_Recognition.face_encodings(jel_image)[0]

known_face_encoding = [
    alex_encoding,
    den_encoding,
    el_encoding,
    jel_encoding

]

known_face_names = [
    "Alex",
    "Dennial"
    "Elon",
    "Jellon"
]

student = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now =datetime.now()
current_date = now.strftime("%y-%m-%d")

f=open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations= face_Recognition.face_locations(rgb_small_frame)
        face_encodings = face_Recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_Recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_Recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in employees:
                    employees.remove(name)
                    print(employees)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendence system",frame)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindow()
f.close()
