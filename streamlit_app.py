from flask import Flask,render_template,Response
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image


app=Flask(__name__)


model=load_model("model.h5")
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'mask', 1:'without mask'}
color_dict={0:(0, 255, 0),1:(0,0,255)}

size = 4
st.write(''' # Mask Detector by Anastasiia ''')
img_file_buffer = st.camera_input('We need your picture for our prediction')


if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    im = np.array(img) 
    im = im[:, :, ::-1].copy() 
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f] 

        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224, 224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1, 224, 224, 3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        st.write('Thank you for testing my service!')
        st.image(im_pil)

        st.write('Want to see more? Check out my [portfolio](https://avganshina.github.io/anastasiaganshina/peronalprojects.html)!')


