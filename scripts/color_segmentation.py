#!/usr/bin/env python

__author__ = 'Jane Li'

import roslib, rospy
import sys, time, os.path
import argparse
import cv2
import matplotlib.pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
import cofi.generators.color_generator as cg
import cofi.trackers.color_tracker as ct
import cofi.visualization.point_cloud as pcl_vis


try:
    import pcl
    has_pcl = True
except ImportError:
    has_pcl = False

# can go from 0 (hard colors) to 1 (soft colors)
COLOR_MARGIN = 0.52
COLOR_MARGIN_HS = 0.75

NUM_COLORS = 12
ok = True


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("/RGB_img",Image)

        cv2.namedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_quality = "sd"
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
        self.depth_topic = "/kinect2/" + self.image_quality + "/image_depth_rect"

        self.image_sub = rospy.Subscriber(self.image_topic,Image,self.callback_img)
        self.depth_sub = rospy.Subscriber(self.depth_topic,Image,self.callback_depth)

        self.previous_time = time.clock()
        self.current_time = time.clock()

        self.cv_image = np.zeros((self.image_height,self.image_width,3), np.uint8)
        self.depth_image = np.zeros((self.image_height,self.image_width,1), np.uint8)
        self.hue_filters = list()
        self.hs_filters = list()

        self.points_rgb = np.zeros((0, 4), dtype=np.float32)
        # the return value of this algorithm is a list of centroids for the detected blobs
        self.centroids_xyz = np.zeros((0, 3), dtype=np.float32)

        self.cx = 0.0
        self.cy = 0.0
        self.h = 0.0
        self.contour = []

        if os.path.isfile('markers_v1.json'):
            print "Loading marker information from markers_v1.json"
            hs_filters = ct.load_hs_filters('markers_v1.json', COLOR_MARGIN_HS)
            print hs_filters
        print "Computing markers information from ideal hue distribution"
        colors = cg.get_hsv_equispaced_hues(NUM_COLORS)
        for color in colors:
            h,_,_ = color
            threshold = COLOR_MARGIN*(360.0/NUM_COLORS)

            h_min = 2*h - threshold/2
            if h_min < 0:
                h_min += 360
            h_min /= 2

            h_max = 2*h + threshold/2
            if h_max > 360:
                h_max -= 360
            h_max /= 2

            self.hue_filters.append((int(round(h_min)), int(round(h_max)), int(round(h))))
        print self.hue_filters

    def callback_img(self,data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        (rows,cols,channels) = self.cv_image.shape
        print "row={}, col={}, channels={} \n".format(rows, cols, channels)

        if len(self.hs_filters) > 0:
            blobs = ct.detect_hs(self.cv_image, self.hs_filters)
        else:
            blobs = ct.detect_hues(self.cv_image, self.hue_filters)

        for idx, blob in enumerate(blobs):
            self.cx, self.cy, self.h, self.contour = blob
            bgr = cv2.cvtColor(np.array([[[self.h,255,255]]],np.uint8),cv2.COLOR_HSV2BGR)
            bgr = tuple(bgr.tolist()[0][0])
            cv2.circle(self.cv_image, (self.cx, self.cy), 7,  bgr,     -1)
            cv2.circle(self.cv_image, (self.cx, self.cy), 7, (250, 250, 250), 2)

        cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(20)

    def callback_depth(self,data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError, e:
            print e

        (rows,cols,channels) = self.depth_image.shape
        print "row={}, col={}, channels={} \n".format(rows, cols, channels)

        if rows > 50 and cols > 50:
            mask = np.zeros(self.cv_image.shape[0:2], np.uint8)
            cv2.drawContours(mask,[self.contour],0,255,-1)
            mask_px = np.transpose(np.nonzero(mask))

            # cloud_mask = inverse_uv[mask_px[:, 0], mask_px[:, 1], :]
            # valid_mask = (cloud_mask[:,0] >= 0) & (cloud_mask[:,1] >= 0) & (cloud_mask[:,0] <= 1) & (cloud_mask[:,1] <= 1)
            # cloud_mask = cloud_mask[valid_mask]
            #
            # if cloud_mask.shape[0] > 0:
            #     points = cloud[np.clip(cloud_mask[:, 0]*cloud.shape[0], 0, cloud.shape[0]-1).round().astype(np.int),
            #                    np.clip(cloud_mask[:, 1]*cloud.shape[1], 0, cloud.shape[1]-1).round().astype(np.int)]
            #
            #     rgb_values = np.array([pcl_vis.rgb_to_pcl_float(
            #         frame[mask_px[i, 0], mask_px[i, 1], 0],
            #         frame[mask_px[i, 0], mask_px[i, 1], 1],
            #         frame[mask_px[i, 0], mask_px[i, 1], 2]) for i in xrange(points.shape[0])]
            #                           , dtype=np.float32).reshape(points.shape[0], 1)
            #     #import IPython.core.debugger as pdb
            #     #pdb.Tracer()()
            #
            #     centroid = (points.sum(0)/points.shape[0])
            #
            #     if centroid[2] > 0:
            #         centroids_xyz = np.append(centroids_xyz, centroid.reshape(1,3), 0)
            #         points_rgb = np.append(points_rgb, np.append(points, rgb_values, 1), 0)
            #         cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 2)

        # cv2.imshow("Depth window", self.depth_image)
        cv2.waitKey(20)


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)