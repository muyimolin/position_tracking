# cofi Suite

The cofi suite provides a simple system for fiducials tracking.
The assumptions of the algorithm are:
1) controlled lighting (& can hide distracting elements in the image or specify a ROI)
2) colored (of known color) fiducials



cofi_tracking:
takes care of the actual fiducials tracking. It can be used in 3 modes:
- in windows, it can be used with the option ```--realsense``` , and it will try to open a usb connection to a RealSense camera attached to the local computer.

```
cofi_tracking --realsense
```

- alternatively, an image name (for testing purposes), or an ip address can be specified, for example
In order to have this working, you should have a server (```RealSense/RealSenseLib/server.py```) running on a windows computer, and a client (```RealSense/RealSenseLib/client.py```) running on the client computer.
Given an example where the server is running on a pc with ip address ```192.168.0.3```, then the tracking can be run by typing:

```
cofi_tracking 192.168.0.3
```

If ```cofi_tracking``` finds a file named ```markers_v1.json``` it will load the constant threshold intervals corresponding to the colors to track. In a future version, this could be taken directly from an automatic procedure, where a piece of paper with colors to track is shown to the camera.
Also, on a future release, a method of adapting to varying illumination conditions might get implemented, where we shift the Hue and Saturation detection window according to the actual color with is tracked at any given time.
