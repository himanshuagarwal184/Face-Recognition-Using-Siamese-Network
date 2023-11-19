#import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


#import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Building App and Layout

class CamApp(App):

    def build(self):
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify",size_hint=(1,.1),on_press=self.verify)
        self.verification = Label(text="Verification uninitiated",size_hint=(1,.1))

        #adding items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Loading Model
        self.model = tf.keras.models.load_model('siameseModel.h5',custom_objects={'L1Dist':L1Dist})



        #setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)

        return layout
    
    def update(self,*args):

        ret,frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        #flip horizontal and convert to texture

        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.web_cam.texture=img_texture

    #load image from file and convert it to 100x100
    def preprocess(self,file_path):
        byte_img = tf.io.read_file(file_path)   #  Reading the image
        img = tf.io.decode_jpeg(byte_img)       #  Loading the image
        img = tf.image.resize(img,(100,100))    #  Resizing the image to 100x100x3
        img = img / 255.0                       #  scaling image to be between 0 and 1
        return img
    
    #verification function to verify 
    def verify(self,*args):
        detection_threshold=.9
        verification_threshold= .6
        save_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret,frame = self.capture.read()
        frame = frame[120:120+250,200:200+250,:]
        cv2.imwrite(save_path,frame)


    # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_image')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_image', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image'))) 
        verified = verification > verification_threshold
        
        self.verification.text='verified' if verified == True else 'Unverified' 

        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.5))
        Logger.info(np.sum(np.array(results)>0.8))
        Logger.info("________")
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

if __name__=='__main__':
    CamApp().run()