######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 17

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
filename=""
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    # Open video file
    frame_counter = 0
    video = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('C:\\Users\\ozgur\\Desktop\\output.avi',fourcc, 5, (1280,720))

    while(video.isOpened()):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_counter += 1
        if frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            messagebox.showinfo("İşlem Başarılı", "Bulanıklaştırılmış video masaütüne başarıyla oluşturuldu.")
            video.release()
            cv2.destroyAllWindows()
        frame_expanded = np.expand_dims(frame, axis=0)
        width  = video.get(3) # float
        height = video.get(4) # float
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
            
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=1)

        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Object detector', frame)
        boxes = np.squeeze(boxes)
        max_boxes_to_draw = boxes.shape[0]
        print(len(boxes.shape))
        scores = np.squeeze(scores)
        min_score_thresh=.2
        count = 0
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                count = count + 1
                print ("This box is gonna get used", boxes[i])
        # Create ROI coordinates
        print("count",count)
        while count != -1:        
            a = boxes[count][0]*height
            b = boxes[count][1]*width
            c = boxes[count][2]*height
            d = boxes[count][3]*width
            topLeft = (int(b), int(a))
            bottomRight = (int(d), int(c))
            x, y = topLeft[0], topLeft[1]
            w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
            print(x,h,y,w,a)

                # Grab ROI with Numpy slicing and blur
            ROI = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(ROI, (195,195), 0) 

                # Insert ROI back into image
            frame[y:y+h, x:x+w] = blur
            count=count-1

        #cv2.imshow('blur', blur)
        cv2.imshow('image', frame)
        if ret == True:
            b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(b)
        else:
            break   
        # display input and output image
        #cv2.imshow("Gaussian Smoothing",np.hstack((frame, dst)))

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.geometry('750x500')
root.title("Object Detection")
button = tk.Button(root, text='Video Seçiniz', command=UploadAction)
button.pack()
root.mainloop()



