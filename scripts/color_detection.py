#!/usr/bin/env python

__author__ = 'Jane Li'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path


if __name__ == '__main__':
    frame_folder = "/home/motion/ros_ws/src/position_tracking/data/image/"
    sample_folder = "/home/motion/ros_ws/src/position_tracking/scripts/sample/"
    file_name = "frame0000.jpg"
    img = cv2.imread(frame_folder+file_name,cv2.IMREAD_COLOR)

    cv2.imshow("Image window", img)

    (height, width) = img.shape[:2]
    # img_crop = img[int(height/2)-10:int(height/2)+60, int(width/2)-5:int(width/2)+10];

    sample_name = "sample002.jpg"
    img_crop = cv2.imread(sample_folder + sample_name, cv2.IMREAD_COLOR)
    # print "save sample"
    # cv2.imwrite(sample_folder + sample_name, img_crop)

    blank_image = np.zeros((height, width, 3), np.uint8)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    # cv2.imshow("HSV image", img_HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[int(height/2)-10:int(height/2)+60, int(width/2)-5:int(width/2)+10] = 255
    # cv2.imshow("Mask", mask)

    img_masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Masked img", img_masked)

    img_HSV_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV);
    img_crop_HSV = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV);


    # hist_masked = cv2.calcHist([img_HSV_masked], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    # plt.imshow(hist, interpolation='nearest')
    # plt.subplot(221), plt.imshow(img)
    # plt.subplot(222), plt.imshow(img_masked)
    # plt.subplot(223), plt.imshow(hist)
    # plt.subplot(224), plt.imshow(hist_masked)

    color = ["b", "g", "r"]
    value_range = [180, 255, 255]
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_HSV], [i], None, [value_range[i]], [0, value_range[i]])/float(img_HSV.size)
        hist_mask = cv2.calcHist([img_crop_HSV], [i], None, [value_range[i]], [0, value_range[i]])/float(img_crop_HSV.size)
        print "max", col, "=", np.argmax(hist_mask)
        plt.plot(hist, color=col)
        plt.plot(hist_mask, color=col, linewidth=3.0)
        plt.xlim([0, 256])
        # plt.ylim([0, 20000])
    plt.show()

    k = cv2.waitKey(0) & 0xFF
    while True:
        if k == 27 or k == 255:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            exit(0)
        elif k == ord('s'): # wait for 's' key to save and exit
            save_image_name = "file_save"
            cv2.imwrite(folder_name + save_image_name + ".png",img)
            cv2.destroyAllWindows()
            exit(0)
        elif k != ord('s'):
            print "pressed key = ", k
            k = cv2.waitKey(0) & 0xFF





