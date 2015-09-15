#!/usr/bin/env python

__author__ = 'Jane Li'

import roslib, rospy
import sys, time, os
# import argparse
import cv2
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np


class image_converter:

    def __init__(self, quality=""):
        self.image_pub = rospy.Publisher("/RGB_img",Image)

        cv2.namedWindow("Image window", 1)
        self.bridge = CvBridge()
        if not quality:
            self.image_quality = "sd"
        else:
            self.image_quality = quality

        if self.image_quality == "qhd":
            self.image_height = 540;
            self.image_width = 960;
        elif self.image_quality == "hd":
            self.image_height = 1080;
            self.image_width = 1920;
        elif self.image_quality == "sd":
            self.image_height = 424;
            self.image_width = 512;

        self.image_topic = "/kinect2/" + self.image_quality + "/image_color_rect"
        self.cloud_topic = "/kinect2/" + self.image_quality + "/points"

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.callback_img)
        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.callback_cloud)

        self.previous_time = time.clock()
        self.current_time = time.clock()

        self.cv_image = np.zeros((self.image_height,self.image_width,3), np.uint8)

        self.cloud = PointCloud2()
        self.centroid_xyz = []
        self.command = None
        self.sample_list = list()
        self.hsv_threshold = [5, 75, 75]
        # detect_mode options: hue, hist
        self.detect_mode = "hue"
        self.disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.cloud_xyz = None
        sample_dir = "/home/motion/ros_ws/src/position_tracking/scripts/sample/"
        self.sample_name = os.listdir(sample_dir)
        self.hs_filter = list()
        self.hsv_str = ["hue", "sat", "val"]
        self.hsv_range = [180, 255, 255]
        self.color_MIN = list()
        self.color_MAX = list()

        # initilize position list for each sample
        self.n_sample = len(self.sample_name)
        for i in range(0, self.n_sample):
            cmd_str = "self.centroids_xyz_" + str(i) + "=[]"
            exec cmd_str
            print "Create position sequence: ", i

        for s in self.sample_name:
            sample_file = sample_dir + s
            print sample_file
            if os.path.isfile(sample_file):
                print "Load sample ..."
                img = cv2.imread(sample_file, cv2.IMREAD_COLOR)
                img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hist_sample_2D = cv2.calcHist([img_HSV],[0, 1], None, [180, 255], [0, 180, 0, 255])
                cv2.normalize(hist_sample_2D, hist_sample_2D, 0, 255, cv2.NORM_MINMAX)
                self.sample_list.append(hist_sample_2D)

                hist_max_bin = []
                for i in range(0, 3):
                    hist_sample = cv2.calcHist([img_HSV],[i], None, [self.hsv_range[i]], [0, self.hsv_range[i]])
                    hist = np.sum(hist_sample, axis=1)
                    hist_max = np.amax(hist)
                    hist_max_idx = np.argmax(hist)
                    print self.hsv_str, ": size = ", hist.shape, "max = ", hist_max, "max_idx = ", hist_max_idx
                    hist_max_bin.append(hist_max_idx)

                self.hs_filter.append(hist_max_bin)
                color_min = np.asarray(hist_max_bin) - np.asarray(self.hsv_threshold)
                color_max = np.asarray(hist_max_bin) + np.asarray(self.hsv_threshold)
                print color_min, color_max
                self.color_MIN.append(color_min)
                self.color_MAX.append(color_max)
                print hist_max_bin

    def callback_img(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        cv_image_HSV = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

        cv_mask_list = list()
        if len(self.hs_filter) > 0:
            for i, hs in enumerate(self.hs_filter):
                if self.detect_mode == "hue":
                    mask = cv2.inRange(cv_image_HSV, self.color_MIN[i], self.color_MAX[i])
                    cv_image_HSV_masked = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask)
                    cv2.imshow("Image window", cv_image_HSV_masked)
                    cv_mask_list.append(mask)
                    self.command = cv2.waitKey(1)

                elif self.detect_mode == "hist":
                    hist_sample = self.sample_list[i]
                    dst = cv2.calcBackProject([cv_image_HSV], [0,1], hist_sample, [0, 180, 0, 256], 1)
                    cv2.filter2D(dst, -1, self.disc, dst)
                    ret, thresh = cv2.threshold(dst, 127, 255, 0)
                    mask = cv2.merge((thresh, thresh, thresh))
                    cv_mask_list.append(mask)
                    cv_image_masked = cv2.bitwise_and(self.cv_image, mask)
                    res = np.vstack((self.cv_image, cv_image_masked))
                    cv2.imshow("Image window", res)
                    self.command = cv2.waitKey(1)

        if self.cloud_xyz is not None:
            for i, mask in enumerate(cv_mask_list):
                cloud_pt = list()
                #  size mask.shape = (424 = self.image_height, 512 = self.image_height)
                # len(mask[0] = 512), len(ind_y = 424)
                ind_x = np.where(mask == 255)[0]
                ind_y = np.where(mask == 255)[1]
            #     # print type(ind_x), "ind_x = ", ind_x.shape, "ind_y = ", ind_y.shape
                for i_pt in range(0, ind_x.shape[0]):
                    current_pt = self.cloud_xyz[(ind_x[i_pt])*self.image_width + (ind_y[i_pt])]
                    cloud_pt.append(current_pt)
                point_array = np.array(cloud_pt)
                point_array = point_array[~np.isnan(point_array).any(1)]
                current_centroid = np.mean(point_array, axis=0)
                print self.sample_name[i], current_centroid
                cmd_str = "self.centroids_xyz_" + str(i) + ".append(current_centroid)"
                exec cmd_str
            # self.centroid_xyz.append(current_centroid)

    def callback_cloud(self, data):
        self.cloud = data
        if self.cloud is not None:
            generator = pc2.read_points(self.cloud, skip_nans=False, field_names=("x", "y", "z"))
            self.cloud_xyz = list(generator)


def main(args):
    quality_list = ["hd", "qhd", "sd"]
    im_quality = ""

    if len(args) > 1:
        if args[1] not in quality_list:
            print "Choose default image quality: sd. \nImage quality options: hd, qhd, sd "
        else:
            im_quality = args[1]
    else:
        print "Choose default image quality: sd."

    ic = image_converter(im_quality)
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)