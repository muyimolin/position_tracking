#!/usr/bin/env python

__author__ = 'Jane Li'

import cv2
import cv2.cv as cv
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os.path
import sys

boxes = []

# def on_mouse(event, x, y, img, flags, params):
#     t = time()
#     if event == cv.CV_EVENT_LBUTTONDOWN:
#         print 'Start Mouse Position: '+ str(x) + ', ' + str(y)
#         sbox = [x, y]
#         boxes.append(sbox)
#     elif event == cv.CV_EVENT_LBUTTONUP:
#         print 'End Mouse Position: ' + str(x) + ', ' + str(y)
#         ebox = [x, y]
#         boxes.append(ebox)
#         print boxes
#         crop = img[boxes[-2][1]:boxes[-1][1],boxes[-2][0]:boxes[-1][0]]
#         cv2.imshow('crop',crop)
#         k = cv2.waitKey(0)
#         if ord('r') == k:
#             cv2.imwrite('Crop'+str(t)+'.jpg', crop)
#             print "Written to file"


class Circle_drawing():

    def __init__(self, mode):

        self.sample_folder = "/home/motion/ros_ws/src/position_tracking/scripts/sample/"
        self.frame_folder = "/home/motion/ros_ws/src/position_tracking/scripts/frame/"

        self.sample_name = "sample002.jpg"
        self.frame_name = "frame0000.jpg"

        sample_file = self.sample_folder + self.sample_name
        frame_file = self.frame_folder + self.frame_name
        print "frame name: ", frame_file

        if os.path.isfile(frame_file):
            self.img_frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            cv2.imshow("Frame Window", self.img_frame)
        else:
            print "Frame file doesn't exist"

        self.draw_mode = mode
        self.count = 0
        self.box = list()
        (self.img_height, self.img_width) = self.img_frame.shape[:2]
        print "img_height = ", self.img_height, "img_width = ", self.img_width
        self.img_crop = None
        self.img_crop_flag = False

    def draw_circle(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img_frame, (x, y), 10, (255, 0, 0), -1)

    def draw_rect(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print "Start Mouse Position: ", x,  ", ", y
            start_box = [x, y]
            self.box.append(start_box)
        elif event == cv2.EVENT_LBUTTONUP:
            print "End Mouse Position: ",  x,  ", ", y
            end_box = [x, y]
            self.box.append(end_box)
            print "selected box = ", self.box
            self.img_crop = self.img_frame[self.box[-2][1]:self.box[-1][1], self.box[-2][0]:self.box[-1][0]]
            self.img_crop_flag = True

    def display(self):
        if self.draw_mode == "circle":
            cv.SetMouseCallback('Frame Window', self.draw_circle)
        elif self.draw_mode == "rect":
            cv.SetMouseCallback('Frame Window', self.draw_rect)
        while True:
            self.count += 1
            cv2.imshow("Frame Window", self.img_frame)
            if self.draw_mode == "rect" and self.img_crop_flag is True:
                cv2.imshow("Crop Image", self.img_crop)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == 33:
                cv2.destroyAllWindows()
                exit(0)
            elif k == ord('s'):
                print "Save image ..."
                save_image_name = "file_save"
                cv2.imwrite(self.frame_folder + save_image_name + ".png", self.img_frame)
                if self.draw_mode == "rect" and self.img_crop_flag is True:
                    sample_name = raw_input("Please input name for the sample file:")
                    if sample_name == "":
                        print "Use default sample name"
                        sample_name = self.sample_name
                    cv2.imwrite(self.sample_folder + sample_name + ".jpg", self.img_crop)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    mode = "circle"
    mode_list = ["circle", "rect"]
    if len(sys.argv) > 1:
        if sys.argv[1] in mode_list:
            mode = sys.argv[1]
    else:
        print "choose default mode: circle - double click to draw circle in image ... "

    circ = Circle_drawing(mode)
    circ.display()
