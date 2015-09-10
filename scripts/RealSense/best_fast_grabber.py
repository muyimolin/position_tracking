#!/usr/bin/env python

__author__ = 'Mark Draelos, Alessio Rocchi'

import logging, traceback
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s')
logging.getLogger('OpenGL').setLevel(99)
logging.getLogger('packet').setLevel(logging.WARNING)

import sys, numpy
from time import time

from RealSenseLib.PyRealSenseLib import Camera, StreamType, PixelFormat

class best_fast_grabber(object):
    """
    the best_fast_grabber class will use PyRealSenseLib in order to create the highest resolution best_fast_grabber for the camera
    frames. The grab function will return a color image, a cloud, a uv map, the inverse uv map, and a colorized cloud
    """
    def __init__(self, device_identifier=0, show=False):

        self.camera = self._setup_camera(device_identifier)

        # variables to store visualization
        self.app = None
        self.colormap = None
        self.color_window = None
        self.depth_window = None

        # store tic/toc info for timing and number grabs
        self.mark = 0.0
        self.n = 0

        self.show_ui = show

        pass

    def _setup_camera(self, device_identifier):
        """
        _setup_camera will setup a RealSense for grabbing image & depth data
        :param device_identifier: an integer specifying the device identifier
        :return: a Camera object configured for highest resolution grabbing @30fps, with depth information
        """

        # enumerate all devices
        try:
            devices = list(Camera.list(StreamType.COLOR | StreamType.DEPTH))
        except:
            logger.error('unable to enumerate devices')
            logger.error(traceback.format_exc())

        if devices:
            for (i, device) in enumerate(devices):
                logger.debug('found device {}: {} {}'.format(i, device, device.serial))
        else:
            logger.debug('found no devices')

        if isinstance(device_identifier, int):
            # select the device based on device index
            try:
                device = devices[device_identifier]
            except IndexError:
                logger.error('invalid device index: {}'.format(device_identifier))
                raise SystemExit
        else:
            # select the device based on serial number
            try:
                device = filter(lambda d: d.serial == device_identifier, devices)[0]
            except IndexError:
                logger.error('invalid device serial: {}'.format(device_identifier))
                raise SystemExit

        logger.info('using device {} {}'.format(device, device.serial))
        camera = Camera(device)

        depth_formats = camera.formats(StreamType.DEPTH)
        for df in depth_formats:
            logger.debug('found depth format {}'.format(df))

        # find the highest resolution 30 fps depth format
        depth_formats = filter(lambda df: df.fps == 30 and df.format == PixelFormat.DEPTH, depth_formats)
        if depth_formats:
            depth_format = max(depth_formats, key=lambda cf: cf.width * cf.height)
            logger.debug('using depth format {}'.format(depth_format))
        else:
            logger.error('unable to find suitable depth format')
            raise SystemExit

        color_formats = camera.formats(StreamType.COLOR)
        for cf in color_formats:
            logger.debug('found color format {}'.format(cf))

        # find a 30 fps color format of the same resolution
        color_formats = filter(lambda cf: cf.fps == 30 and cf.format == PixelFormat.RGB24 and cf.width == depth_format.width and cf.height == depth_format.height, color_formats)
        if color_formats:
            color_format = color_formats[0]
            logger.debug('using color format {}'.format(color_format))
        else:
            logger.error('unable to find color format matching {}'.format(depth_format))
            raise SystemExit

        try:
            camera.add(color_format, StreamType.COLOR)
            camera.add(depth_format, StreamType.DEPTH)
            camera.open()
        except Exception:
            logger.error('error opening camera')
            logger.error(traceback.format_exc())
            raise SystemExit
        else:
            logger.info('camera open')

        return camera

    def close(self):
        if self.camera is not None:
            self.camera.close()
        self.camera = None

    def grab(self, show=None):
        """
        Grabs from the configured RealSense Camera
        :return: a tuple (color_image, cloud, depth_uv, inverse_uv)
        """
        if self.show_ui or (show is not None and show is True):
            from RealSenseLib.ui.numpy_widget import NumpyWidget
            import matplotlib.cm
            from PySide.QtGui import QApplication

            if self.app is None:
                self.app = QApplication(sys.argv)
                self.color_window = NumpyWidget()
                self.color_window.show()
                self.depth_window = NumpyWidget()
                self.depth_window.show()

            self.colormap = numpy.float32(matplotlib.cm.jet(numpy.arange(1001) / 1000.0))

        self.mark = time()
        try:
            frame = self.camera.read()
        except RuntimeError as e:
            logger.error('error while capturing frame')
            logger.error(traceback.format_exc())
            raise e

        color_buffer = frame.color.open(PixelFormat.RGB24)
        color_image = numpy.frombuffer(color_buffer.data(), dtype=numpy.uint8).reshape((color_buffer.height, color_buffer.width, -1))
        # BGR -> RGB
        color_image = color_image[:,:,::-1]

        try:
            cloud_buffer = frame.computePointCloud()
        except RuntimeError as e:
            logger.warn('{} -> skipping frame'.format(e))
            return (color_image)
        cloud = numpy.frombuffer(cloud_buffer, dtype=numpy.float32).reshape((frame.depth.height, frame.depth.width, 3))

        try:
            depth_uv_buffer = frame.computeDepthUVMap()
        except RuntimeError as e:
            logger.warn('{} -> skipping frame'.format(e))
            return (color_image, cloud)
        depth_uv = numpy.frombuffer(depth_uv_buffer, dtype=numpy.float32).reshape((frame.depth.height, frame.depth.width, 2))

        try:
            color_uv_buffer = frame.computeColorUVMap()
        except RuntimeError as e:
            logger.warn('{} -> skipping frame'.format(e))
            return (color_image, cloud, depth_uv)
        inverse_uv = numpy.frombuffer(color_uv_buffer, dtype=numpy.float32).reshape((color_buffer.height, color_buffer.width, 2))

        ##############
        # Based on __init__.py in RealSenseLib - VERIFY this
        # # flip everything up/down based on camera mounting
        # color = color[::-1,:,:]
        # cloud = cloud[::-1,:,:]
        # if depth_uv is not None:
        #     depth_uv = depth_uv[::-1,:,:]
        #
        # # the point cloud and the depth UV map actually need to have their values changed
        # # because the Y spatial direction is reversed
        # cloud[:,:,1] *= -1
        # if depth_uv is not None:
        #     depth_uv[:,:,1] = 1 - depth_uv[:,:,1]
        #
        # # convert point cloud to meters
        # cloud /= 1000
        ###############

        ##############
        # depth_buffer = frame.depth.open(PixelFormat.DEPTH_F32)
        # depth_image = numpy.frombuffer(depth_buffer.data(), dype=numpy.float32).reshape((depth_buffer.height, depth_buffer.width))
        #
        # cloud[:,:,2] = depth_image
        # depth_colorized = uvtexture(color_image, depth_uv)
        ###############

        #color_image = numpy.empty((cloud.shape[0], cloud.shape[1], 3), dtype=numpy.uint8)
        #color_image[:,:] = [ 0, 255, 0 ]

        #depth_uv = numpy.empty((cloud.shape[0], cloud.shape[1], 2), dtype=numpy.float32)
        #depth_uv[:,:] = [ -1, -1 ]

        # update the view every 0.3s
        if show and (self.n % 10 == 0):
            self.color_window.update_frame(color_image)
            self.depth_window.update_frame(self.colormap[numpy.clip(cloud[:,:,2], 0, len(self.colormap) - 1).astype(numpy.int)])
            # process events for the windows
            self.app.processEvents()

        self.n += 1

        if self.n % 30 == 0:
            # print some debug stats
            fps = 30 / (time() - self.mark)
            self.mark = time()
            logger.debug('{:.1f} fps'.format(fps))

        return (color_image, cloud, depth_uv, inverse_uv)


def parse_args():
    try:
        device_identifier = int(sys.argv[1])
    except ValueError:
        device_identifier = sys.argv[1]
    except IndexError:
        device_identifier = 0

    return device_identifier

if __name__ == '__main__':
    try:
        device_identifier = parse_args()
    except Exception:
        print sys.argv[0], '[device index | serial]'
        raise SystemExit

    grabber = best_fast_grabber(device_identifier)

    while True:
        try:
            grabber.grab(show=True)
        except KeyboardInterrupt:
            break

    try:
        grabber.close()
    except Exception as e:
        logger.warn('ignored error while closing camera')
        logger.warn(traceback.format_exc())

    logger.info('exiting')


