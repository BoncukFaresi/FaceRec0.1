import numpy as np
import cv2
import pickle

#programın suratı tanıması için gerekli parametreler
face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')#suratları yakalamk için hazır içerik
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels={}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items() }

cap = cv2.VideoCapture(0)#yakalancak kamera seçiliyor

while True:
    ret, frame=cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h)in faces :
        #print(x,y,w,h)#roi = region of interest
        roi_gray =gray[y:y+h,x:x+w]#[x:x+yükseklik , y:y+yükseklik]
        roi_color=frame[y:y+h,x:x+w]


        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke = 2
            cv2.putText(frame,name.replace("-"," "),(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item ="10.png"
        cv2.imwrite(img_item,roi_color)
        #dörtgen çizerek suratı belirtme
        color = (0,255,0) #BGR
        stroke = 2 #kalınlık
        end_cord_x = x + w #dörtkenin x için bitme koordinatı
        end_cord_y = y + h #dörtkenin y için bitme koordinatı
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)#dötgenin çizimi

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
