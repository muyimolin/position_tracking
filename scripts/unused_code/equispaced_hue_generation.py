#!/usr/bin/env python
__author__ = 'Alessio Rocchi'

import cv2
import numpy as np
import cofi.generators.color_generator as cg

NUM_COLORS = 12
# A4 - printing 1 inch diameter circles = 100 pixel radius circles
IMG_WIDTH = 1654
IMG_HEIGHT = 2339

SPOT_RADIUS = 100

# could compute this automatically...
GRID_W = 3
GRID_H = 4

if __name__ == "__main__":
    blank_image = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), np.uint8)
    blank_image[:,:,:] = 255
    colors = cg.get_bgr_equispaced_hues(NUM_COLORS)
    for i in range(NUM_COLORS):
        color_bgr = colors[i]
        grid_x = i/GRID_W
        grid_y = i%GRID_W
        center_x = int((grid_x + 0.5 ) * IMG_WIDTH/GRID_W)
        center_y = int((grid_y + 0.5 ) * IMG_HEIGHT/GRID_H)
        center = (center_x, center_y)
        cv2.circle(blank_image,
                   center,
                   SPOT_RADIUS,
                   tuple(color_bgr),-1)

cv2.namedWindow("test")
cv2.imshow("test",blank_image)
while (cv2.waitKey() & 0xff) != ord('q'): pass
cv2.destroyAllWindows()

cv2.imwrite("img.jpg",blank_image)
