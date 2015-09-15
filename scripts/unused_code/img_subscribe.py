#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("/IMG_RGB",Image)

        cv2.namedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/kinect2/sd/image_color_rect",Image,self.callback_img)
        self.depth_sub = rospy.Subscriber("/kinect2/sd/image_depth_rect",Image,self.callback_depth)


    def callback_img(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        (rows,cols,channels) = cv_image.shape
        # print "[Color image] row = {}, col = {}, channels = {} \n".format(rows, cols, channels)
        if cols > 60 and rows > 60 :
            cv2.circle(cv_image, (50,50), 10, (255,0,0))
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(20)

        # try:
        #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError, e:
        #     print e

    def callback_depth(self,data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError, e:
            print e

        (rows,cols,channels) = cv_depth.shape
        print "[Depth image] row = {}, col = {}, channels = {} \n".format(rows, cols, channels)
        if cols > 60 and rows > 60 :
            cv2.circle(cv_depth, (50,50), 10, (0,0,0))
        cv2.imshow("Depth window", cv_depth)
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