import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

class ObjectDetector:
    def __init__(self, model_path, label_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = load_model(model_path)
        with open(label_path, 'r') as f:
            self.labels = f.read().strip().split('\n')
    
    def preprocess_image(self, image):
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image / 255.0
    
    def detect_objects(self, image):
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        results = []

        for i, label in enumerate(self.labels):
            confidence = predictions[0][i]
            if confidence > self.conf_threshold:
                results.append((label, confidence))

        return results

class VideoStreamer:
    def __init__(self, source=0, object_detector=None):
        self.video_source = source
        self.object_detector = object_detector
        self.cap = cv2.VideoCapture(self.video_source)

    def start_stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            results = self.object_detector.detect_objects(frame)
            self.display_results(frame, results)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def display_results(self, frame, results):
        for label, confidence in results:
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30 + results.index((label, confidence)) * 30), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

def main():
    model_path = 'path/to/your/model.h5'
    label_path = 'path/to/your/labels.txt'
    object_detector = ObjectDetector(model_path, label_path)
    video_streamer = VideoStreamer(source=0, object_detector=object_detector)
    video_streamer.start_stream()

if __name__ == "__main__":
    main()

# Pretrained models and labels you would typically download or have locally.
# Model source: TensorFlow Model Zoo or custom trained model
# Labels: List of class names corresponding to the model's output
# Example:
# labels.txt
# person
# bicycle
# car
# motorcycle
# airplane
# bus
# train
# truck
# boat
# traffic light
# fire hydrant
# stop sign
# parking meter
# bench
# bird
# cat
# dog
# horse
# sheep
# cow
# elephant
# bear
# zebra
# giraffe
# backpack
# umbrella
# handbag
# tie
# suitcase
# frisbee
# skis
# snowboard
# sports ball
# kite
# baseball bat
# baseball glove
# skateboard
# surfboard
# tennis racket
# bottle
# wine glass
# cup
# fork
# knife
# spoon
# bowl
# banana
# apple
# sandwich
# orange
# broccoli
# carrot
# hot dog
# pizza
# donut
# cake
# chair
# couch
# potted plant
# bed
# dining table
# toilet
# TV
# laptop
# mouse
# remote
# keyboard
# cell phone
# microwave
# oven
# toaster
# sink
# refrigerator
# book
# clock
# vase
# scissors
# teddy bear
# hair drier
# toothbrush
# Method to load the model and labels need to be adjusted based on user requirements