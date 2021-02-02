# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 22:42:43 2020

@author: Jose
"""

import cv2
import datetime
import argparse
import time
# import imutils
# from imutils.video import VideoStream

from utils import detector_utils as detector_utils

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int, default=1,
                help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    # score_thresh = 0.60
    score_thresh = 0.60
    # Get stream from webcam and set parameters)
    # vs = VideoStream().start()
    vs = cv2.VideoCapture(0)

    # max number of faces we want to detect/track
    num_faces_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 1

    im_height, im_width = (None, None)

    try:
        dist_list = []
        while True:
            # Read Frame and process
            _, frame = vs.read()
            # frame = cv2.imread("org_nfside_1105.jpg")
            # frame = cv2.resize(frame, (1080, 1440))
            frame = cv2.resize(frame, (520, 540))
            if im_height is None:
                im_height, im_width = frame.shape[:2]
                print(im_height, im_width)

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except None:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # Draw bounding boxes and text

            dist = detector_utils.draw_box_on_image(
                num_faces_detect, score_thresh, scores, boxes, im_width, im_height, frame)
            if dist is None:
                pass
            elif dist < 150 or dist > 170:
                pass
            else:
                dist_list.append(dist)
                time.sleep(1.5)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()

            fps = num_frames / elapsed_time
            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.release()
                    time.sleep(1.5)
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))
        #
        # To check the most repeated value of the list
        count = 0
        ele = dist_list[0]
        for i in dist_list:
            current_element = dist_list.count(i)
            if current_element > count:
                count = current_element
                ele = i

        print(dist_list)
        # print(len(dist_list))
        print(ele)

    except KeyboardInterrupt and IndexError:
        print("No face found ")
        
