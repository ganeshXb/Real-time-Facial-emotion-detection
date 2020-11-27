
from keras.models import load_model
from time import sleep 
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.style as mpl
import matplotlib.pyplot as plt
import time
import rectangle 


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('weights.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']   

capture = cv2.VideoCapture(0) 

primary_label,secondary_label,inverse_label = '','',''
fig = plt.figure()
ax= fig.add_subplot(111)
ax.set_title("Emotion Data Analysis - 5 Emotions")
i = 0
t = []
emotion_neutral = []
emotion_angry = []
emotion_happy = []
emotion_sad = []
emotion_surprise = []
while True:
    # grab single frame, returns True
    ret, frame = capture.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)  
    cv2.rectangle(frame,(0,0),(250,200),(30,30,30),cv2.FILLED)
    cv2.putText(frame,'S T A T I S T I C S', (25,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1) 
    
    #Draw rectangle on face detected
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(180,180,0),1)
        #rectangle
        reactangle.drawborder(frame,(x,y),(x+w,y+h),(255,128,0),1,1,20)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  
        
        if np.sum([roi_gray]) !=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make prediction on the ROI, then lookup the class 
            predictions= classifier.predict(roi)
            pred = classifier.predict(roi)[0]
            
            # Probabilities
            angry = predictions[0][0] * 100
            happy = predictions[0][1] * 100
            neutral = predictions[0][2] * 100
            sad = predictions[0][3] * 100
            surprise = predictions[0][4] * 100
            
            # label probabilities
            label_angry = "Angry: {0}%".format(round(angry,2))
            label_happy = "Happy: {0}%".format(round(happy,2))
            label_neutral = "Neutral: {0}%".format(round(neutral,2))
            label_sad = "Sad: {0}%".format(round(sad,2))
            label_surprise = "Surprise: {0}%".format(round(surprise,2))
                       
            # prediction label (actual)
            primary_label = (f"Primary emotion: {class_labels[pred.argmax()]}")
            inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")

            if primary_label != class_labels[2]:
                secondary_label = (f"Secondary emotion: {class_labels[2]}")
                if secondary_label != inverse_label:
                    inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")
            elif primary_label != class_labels[1]:
                secondary_label = (f"Secondary emotion: {class_labels[1]}")
                if secondary_label != inverse_label:
                    inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")
            elif primary_label != class_labels[0]:
                secondary_label = (f"Secondary emotion: {class_labels[0]}")
                if inverse_label != secondary_label:
                    inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")
            elif primary_label != class_labels[3]:
                secondary_label = (f"Secondary emotion: {class_labels[3]}")
                if inverse_label != secondary_label:
                    inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")
            elif primary_label != class_labels[4]:
                secondary_label = (f"Secondary emotion: {class_labels[4]}")
                if inverse_label != secondary_label:
                    inverse_label = (f"Inverse emotion: {class_labels[pred.argmin()]}")
            else:
                print('error')
                inverse_label = secondary_label

            # label coordinates 
            primary_label_position =(x,y-50)
            secondary_label_position =(x,y-30)
            inverse_label_position = (x,y-10)

            # rectangle for predicted neutral_emotion
            cv2.rectangle(frame,(x,y-65),
                         (x+200,y-5),
                         (30,30,30),cv2.FILLED)
           
            # probabilities data              
            cv2.putText(frame,label_angry,(10,100),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame,label_happy,(10,120),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame,label_neutral,(10,140),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame,label_sad,(10,160),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame,label_surprise,(10,180),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            
            # predicted neutral_emotion legend
            cv2.putText(frame,primary_label,(10,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(frame,secondary_label,(10,60),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(frame,inverse_label,(10,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)

            # put text prediction to frame
            cv2.putText(frame, primary_label, primary_label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame, secondary_label, secondary_label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv2.putText(frame, inverse_label, inverse_label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)                       
            
            # Visualize emotions
            emotion_neutral.append(float(neutral))
            emotion_angry.append(float(angry))
            emotion_happy.append(float(happy))
            emotion_sad.append(float(sad))
            emotion_surprise.append(float(surprise))
            t.append(i)

            ax.plot(t,emotion_angry,label='Angry')
            ax.plot(t,emotion_happy,label='Happy')
            ax.plot(t,emotion_neutral,label='Neutral')
            ax.plot(t,emotion_sad,label='Sad')
            ax.plot(t,emotion_surprise,label='Surprise')
            ax.set_xlabel('Time [Seconds]')
            ax.set_ylabel('Emotions Detected [Percent %]')

            handles,labels_ = plt.gca().get_legend_handles_labels()
            by_label= dict(zip(labels_,handles))

            ax.legend(by_label.values(),by_label.keys())
            fig.canvas.draw()

            time.sleep(0.1)
            i += 0.2

        else: 
            cv2.putText(frame,'No Face Found',primary_label_position,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
          
    #show the text on the output window
    cv2.imshow('Emotion Detector', frame)
    fig.show()
    # q key for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Display exit message
        print('Exiting Emotion Detector')
        break
    
# Close webcam and close main program
capture.release()
cv2.destroyAllWindows()
