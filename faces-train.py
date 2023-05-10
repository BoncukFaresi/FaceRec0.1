import os
import cv2
from PIL import Image
import numpy as np
import pickle

#Ana konumu faces-train.py dosyasının konumu olarak atıyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#eğitim için kullanılıcak resimlerin onumunu belirliyoruz
image_dir= os.path.join(BASE_DIR , "images")

face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')#ilgi alanını belirlemek için yüz detektörümüzü tekrar çağırıyoruz
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids= {}
y_labels = []
x_train = []

for root,dirts,files in os.walk(image_dir):#bütün resimlere ulaşmak için döngü kullanıyoruz
    for file in files:
        if file.endswith("png")or file.endswith("jpg"):#png veya jpg bile biten dosyaları arıyoruz
            path = os.path.join(root,file)#dosyanın konuunu almamızı sağlayan fonksiyon
            #ulaştığımız resimlerin sahiplerinin isimleri ile isimlendirdiğimiz klasörleri başlık olark alıyoruz
            label =os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label,path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            #y_labels.append(label)# labeller için numaralar
            #x_train.append(path) #resimi kontrol edip NUMPY dizisi ,GRİ
            pil_image = Image.open(path).convert("L") # grayscale
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8") #resimi sayısal veri dizisine dönüştürdük
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)

#training
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
