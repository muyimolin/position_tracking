#!/usr/bin/env python

__author__ = 'Alessio Rocchi'

import cv2
import numpy as np
from utils import hs_raw_to_cv

S_MIN = 80
S_MAX = 255

V_MIN = 80
V_MAX = 255

BLUR_SIZE = 3

MIN_AREA = 9
MAX_AREA = 150

SHOW_DEBUG_IMG = True

def load_hs_filters(filename, hue_scaling=1.0):
    """
    Loads a .json file structured in the following way (for n+1 color tracking):
     [ {'H':[h0_min,h0_max],'S':[s0_min,s0_max]},
       {'H':[h1_min,h1_max],'S':[s1_min,s1_max]},
       ...
       {'H':[hn_min,hn_max],'S':[sn_min,sn_max]}]
       The hue values range from 0 to 360, while to saturation values range from 0 to 100.
       The output values will be automatically transformed to opencv values, meaning the ranges
       will be 0-180 for hue, 0-255 for value
    :param filename: the json filename to load
    :return: a list of hs filters
    """
    import json
    with open(filename) as data_file:
        raw_data = json.load(data_file)

    for i in range(len(raw_data)):
        # getting old h_min, h_max values
        h_min, h_max = raw_data[i]['H']

        # computing new range, clipping it
        hue_range = h_max - h_min
        if hue_range < 0:
            hue_range += 360

        # computing new nominal value, clipping it
        if h_min > h_max:
            hue_nominal = (h_max + 360 + h_min)/2.0
            if hue_nominal > 360:
                hue_nominal -= 360
        else:
            hue_nominal = (h_max + h_min)/2.0

        # finding new values of h_min and h_max, clipping them
        h_min = hue_nominal - hue_range * hue_scaling
        if h_min < 0:
            h_min += 360
        h_max = hue_nominal + hue_range * hue_scaling
        if h_max > 360:
            h_max -= 360

        # replace old tuple with new
        raw_data[i]['H'] = (h_min, h_max)

    data_cv = hs_raw_to_cv(raw_data)
    return data_cv

def detect_hs(frame, hs_filters):
    """
    Performs constant box thresholding in HSV color space based on a list of hue and saturation thresholds
    :param frame: the input frame
    :param hs_filters: a list of dicts {'H':(hue_min,hue_max),'S':(saturation_min,saturation_max)}
    :return: a list of tuples (blob_center_x,blob_center_y,nominal_detected_hue, blob_contour)
    """

    # smooth frame
    frame = cv2.blur(frame, (BLUR_SIZE, BLUR_SIZE))

    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    centers = list()

    thresh2 = None
    for hs_filter in hs_filters:
        if len(hs_filter['H']) == 3:
            h_min, h_max, h = hs_filter['H']
        else:
            h_min, h_max = hs_filter['H']
            if h_min > h_max:
                h = (h_max + 180 + h_min)/2.0
                if h > 180:
                    h -= 180
            else:
                h = (h_max + h_min)/2.0
        s_min, s_max = hs_filter['S']

        if h_min > h_max:
            thresh_l = cv2.inRange(hsv,np.array((h_min, s_min, V_MIN)),     np.array((180,  s_max, V_MAX)))
            thresh_u = cv2.inRange(hsv,np.array((0,     s_min, V_MIN)),     np.array((h_max,s_max, V_MAX)))
            thresh = cv2.add(thresh_l, thresh_u)
        else:
            thresh = cv2.inRange(hsv,np.array((h_min,   s_min, V_MIN)),     np.array((h_max,    s_max, V_MAX)))

        if thresh2 is None and SHOW_DEBUG_IMG:
            thresh2 = np.zeros((thresh.shape[0],thresh.shape[1],3))

        # find contours in the threshold image
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # finding contour with maximum area and store it as best_cnt
        max_area = 0
        best_cnt = None

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # @TODO add density check
            if area > MIN_AREA and area < MAX_AREA and area > max_area:
                if SHOW_DEBUG_IMG: # drawing contours for candidate areas
                    cv2.drawContours(thresh2,contours,idx,(255,0,0),1)
                max_area = area
                best_cnt = cnt
        if best_cnt is not None:
            # finding centroids of best_cnt and draw a circle there
            M = cv2.moments(best_cnt)
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            centers.append((cx, cy, h, best_cnt))

            if SHOW_DEBUG_IMG: # drawing areas and contours for blobs
                mask = np.zeros(frame.shape[0:2], np.uint8)
                cv2.drawContours(mask,[best_cnt],0,255,-1)
                mask_px = np.transpose(np.nonzero(mask))
                thresh2[mask_px[:, 0], mask_px[:, 1]] = (255,255,255)
                cv2.drawContours(thresh2,[best_cnt],0,(0,255,0),1)


    if SHOW_DEBUG_IMG:
        cv2.imshow('thresholds',thresh2)

    return centers


def detect_hues(frame, hue_filters):
    """
    detects hues in an image based on a list of hue filter thresholds
    :param frame: the input frame
    :param hue_filters: a list of tuples (hue_min, hue_max)
    :return: a list of tuples (blob_center_x,blob_center_y,nominal_detected_hue)
    """
    hs_filters = [{'H':hue_filter,'S':(S_MIN,S_MAX)} for hue_filter in hue_filters]
    hs_centers = detect_hs(frame, hs_filters)
    #h_centers = [ center[0:4] for center in hs_centers]

    return hs_centers

