from flask import Flask,render_template,Response
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app=Flask(__name__)


model=load_model("model.h5")
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'mask', 1:'without mask'}
color_dict={0:(0, 255, 0),1:(0,0,255)}

size = 4
webcam = cv2.VideoCapture(0)


def generate_frames():

    while True:
        success, frame = webcam.read()
        if not success:
            break

        else:
 
            im = cv2.flip(frame, 1) 
            
 
            # Resize the image to speed up detection
            mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
            # detect MultiScale / faces 
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
                #Save just the rectangle faces in SubRecFaces
                face_img = im[y:y+h, x:x+w]
                resized=cv2.resize(face_img,(224, 224))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1, 224, 224, 3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)
                #print(result)
                
                label=np.argmax(result,axis=1)[0]
            
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            ret, buffer=cv2.imencode('.jpg', im)
            im = buffer.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
