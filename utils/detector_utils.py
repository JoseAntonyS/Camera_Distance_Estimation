# Utilities for object detector.
"""
TO convert the detection box values to pixels refer :
 https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
"""
import numpy as np
# import sys
import tensorflow as tf
# import os
# from threading import Thread
# from datetime import datetime
import cv2
from utils import label_map_util
# from collections import defaultdict

detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph_face.pb'
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/face_label_map.pbtxt'

NUM_CLASSES = 1
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen inference graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        # od_graph_def = tf.GraphDef()
        od_graph_def = tf.compat.v1.GraphDef()
        # with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
        # sess = tf.Session(graph=detection_graph)
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


# Drawing bounding boxes and distances onto image
# def draw_box_on_image(num_faces_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
def draw_box_on_image(num_faces_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    # focalLength = 1200
    focalLength = 875
    # avg_width = 4.0
    # focalLength = 132
    avg_width = 10.16
    # To more easily differentiate distances and detected bounding boxes
    # color = None
    color0 = (255, 0, 0)
    color1 = (0, 255, 0)
    for i in range(num_faces_detect):

        if scores[i] > score_thresh:

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            dis = int(dist)
            if 150 > dis:
                cv2.rectangle(image_np, p1, p2, color0, 3, 1)
                cv2.putText(image_np, 'move farther',
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color0, 2)
            elif dis > 170:
                cv2.rectangle(image_np, p1, p2, color0, 3, 1)
                cv2.putText(image_np, 'come closer', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color0, 2)
            else:
                cv2.rectangle(image_np, p1, p2, color1, 3, 1)
                cv2.putText(image_np, 'You are in the right spot',
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color1, 2)


            cv2.putText(image_np, 'distance: ' + str("{0:.2f}".format(dis) + ' cm'),
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color0, 2)
            return dis


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
