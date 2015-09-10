
__author__ = 'Alessio Rocchi'

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import cofi.generators.color_generator as cg
import cofi.trackers.color_tracker as ct
import cofi.visualization.point_cloud as pcl_vis
import argparse
import os.path

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', nargs='?', default='0', help="a local image file, a number representing a camera id, or an ip address")
    parser.add_argument("--realsense", help="use the Realsense",action='store_true')
    args = parser.parse_args()

    img_mode = False
    opencv_mode = False
    network_mode = False
    realsense_mode = False
    camera_index = 0

    realsense_streamer = None   # Realsense over the network
    realsense_grabber = None    # RealSense grabber
    cap = None                  # OpenCV grabber

    if args.realsense:
        import RealSense.best_fast_grabber as gr
        realsense_mode = True
        realsense_grabber = gr.best_fast_grabber()
    else:
        try:
            camera_index = int(args.img_name)
            opencv_mode = True
            cap = cv2.VideoCapture(camera_index)
        except:
            if len(args.img_name.split('.')) == 4 or args.img_name == 'localhost':
                network_mode = True
                import RealSense.RealSenseLib.client
                port = None
                ip = args.img_name.split(':')[0]
                if len(args.img_name.split(':')) == 2:
                    port = args.img_name.split(':')[1]
                if port is not None:
                    realsense_streamer = RealSense.RealSenseLib.RemoteRawCamera(ip, port)
                else:
                    realsense_streamer = RealSense.RealSenseLib.RemoteRawCamera(ip)
            else:
                img_mode = True



    hue_filters = list()
    hs_filters = list()

    if os.path.isfile('markers_v1.json'):
        print "Loading marker information from markers_v1.json"
        hs_filters = ct.load_hs_filters('markers_v1.json', COLOR_MARGIN_HS)
        print hs_filters
    #else:
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

        hue_filters.append((int(round(h_min)), int(round(h_max)), int(round(h))))

    print hue_filters

    while ok:
        if realsense_mode:
            (color_image, cloud, depth_uv, inverse_uv) = realsense_grabber.grab()
            frame = color_image.copy()
        elif img_mode:
            frame = cv2.imread(args.img_name)
        elif network_mode:
            (color_image, cloud, depth_uv, inverse_uv) = realsense_streamer.read()
            frame = color_image.copy()
        elif opencv_mode:
            # read the frames
            _,frame = cap.read()
        else:
            print 'Error: unknown mode'
            exit()

        start_time = time.clock()

        if len(hs_filters) > 0:
            blobs = ct.detect_hs(frame, hs_filters)
        else:
            blobs = ct.detect_hues(frame, hue_filters)

        if cloud is not None:
            from RealSense.RealSenseLib import uvtexture
            texture = uvtexture( frame,depth_uv )

        points_rgb = np.zeros((0, 4), dtype=np.float32)

        # the return value of this algorithm is a list of centroids for the detected blobs
        centroids_xyz = np.zeros((0, 3), dtype=np.float32)

        for idx, blob in enumerate(blobs):
            cx, cy, h, contour = blob
            bgr = cv2.cvtColor(np.array([[[h,255,255]]],np.uint8),cv2.COLOR_HSV2BGR)
            bgr = tuple(bgr.tolist()[0][0])
            cv2.circle(frame, (cx, cy), 7,  bgr,     -1)
            cv2.circle(frame, (cx, cy), 7, (150, 150, 150), 2)

            # we get all the cloud points whose inverse_uv lies in the blob area as defined by the contour
            if cloud is not None:
                mask = np.zeros(frame.shape[0:2], np.uint8)
                cv2.drawContours(mask,[contour],0,255,-1)
                mask_px = np.transpose(np.nonzero(mask))

                cloud_mask = inverse_uv[mask_px[:, 0], mask_px[:, 1], :]
                valid_mask = (cloud_mask[:,0] >= 0) & (cloud_mask[:,1] >= 0) & (cloud_mask[:,0] <= 1) & (cloud_mask[:,1] <= 1)
                cloud_mask = cloud_mask[valid_mask]

                if cloud_mask.shape[0] > 0:
                    points = cloud[np.clip(cloud_mask[:, 0]*cloud.shape[0], 0, cloud.shape[0]-1).round().astype(np.int),
                                   np.clip(cloud_mask[:, 1]*cloud.shape[1], 0, cloud.shape[1]-1).round().astype(np.int)]

                    rgb_values = np.array([pcl_vis.rgb_to_pcl_float(
                        frame[mask_px[i, 0], mask_px[i, 1], 0],
                        frame[mask_px[i, 0], mask_px[i, 1], 1],
                        frame[mask_px[i, 0], mask_px[i, 1], 2]) for i in xrange(points.shape[0])]
                                          , dtype=np.float32).reshape(points.shape[0], 1)
                    #import IPython.core.debugger as pdb
                    #pdb.Tracer()()

                    centroid = (points.sum(0)/points.shape[0])

                    if centroid[2] > 0:
                        centroids_xyz = np.append(centroids_xyz, centroid.reshape(1,3), 0)
                        points_rgb = np.append(points_rgb, np.append(points, rgb_values, 1), 0)
                        cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 2)


        # Show it, if key pressed is 'Esc', exit the loop
        cv2.imshow('frame', frame)

        end_time = time.clock()
        print "elapsed time", end_time - start_time

        if cv2.waitKey(33) == 27:
            break

        #pcl_vis.render_3d_scatter_with_rgb(points_rgb)
        #plt.show()
        print "centroids:", centroids_xyz

        if img_mode and (cv2.waitKey() & 0xff) == ord('q'):
            ok = False

    cv2.destroyAllWindows()