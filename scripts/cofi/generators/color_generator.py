__author__ = 'arocchi'

import cv2
import numpy as np

def get_bgr_equispaced_hues(num_colors = 12):
    """ Returns a list of bgr values which are equispaced in Hues
    """
    colors = list()
    for i in xrange(num_colors):
        hue = i*np.floor(360/num_colors)
        hue_cv2 = hue/2
        color_hsv = np.uint8([[[hue_cv2, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        colors.append(color_bgr.tolist()[0][0])
    return colors

def get_hsv_equispaced_hues(num_colors = 12):
    """ Returns a list of bgr values which are equispaced in Hues
    """
    colors = list()
    for i in xrange(num_colors):
        hue = i*np.floor(360/num_colors)
        hue_cv2 = hue/2
        colors.append([hue_cv2,255,255])
    return colors

if __name__ == "__main__":
    pass