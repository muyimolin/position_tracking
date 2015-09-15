
from __future__ import division

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import struct
from math import sqrt
import numpy as np
import colorsys

def f_addr_to_i(f):
    return struct.unpack('I', struct.pack('f', f))[0]
    
def i_addr_to_f(i):
    return struct.unpack('f', struct.pack('I', i))[0]

def rgb_to_pcl_float(r, g, b):
    i = r<<16 | g<<8 | b
    return i_addr_to_f(i)

def pcl_float_to_rgb(f):
    i = f_addr_to_i(f)
    r = i >> 16 & 0x0000ff
    g = i >> 8 & 0x0000ff
    b = i >> 0 & 0x0000ff
    return r,g,b

def render_3d_scatter(points, proportion=1, xlabel="x", ylabel="y", zlabel="z"):
    '''
    render 3d points. the points are represented by a list of lists (points) i.e. [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]]
    return the axis handler of the image (which, for example, can be used to change window limit by set_xlim, set_ylim, set_zlim)
    '''
    if len(points[0])==4:
        ax = render_3d_scatter_with_rgb(points, proportion, xlabel, ylabel, zlabel)
        return ax
    every = int(1/proportion)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([x for i,(x,_,_) in enumerate(points) if i%every==0], 
        [y for i,(_,y,_) in enumerate(points) if i%every==0], zs=[z for i,(_,_,z) in enumerate(points) if i%every==0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    return ax

def render_3d_scatter_with_rgb(points, proportion=1, xlabel="x", ylabel="y", zlabel="z"):
    '''
    render 3d points. the points are represented by a list of lists (points with rgb) i.e. [[x1,y1,z1,rgb1],[x2,y2,z2,rgb2],...,[xn,yn,zn,rgbn]]
    return the axis handler of the image (which, for example, can be used to change window limit by set_xlim, set_ylim, set_zlim)
    '''
    every = int(1/proportion)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb = [c for _,_,_,c in points]
    rgb_int = [struct.unpack('I', struct.pack('f', c))[0] for c in rgb]
    r = [c >> 16 & 0x0000ff for c in rgb_int]
    g = [c >> 8 & 0x0000ff for c in rgb_int]
    b = [c >> 0 & 0x0000ff for c in rgb_int]
    rgb = [[r[i]/255, g[i]/255, b[i]/255] for i in xrange(len(r))]
    x_selected = [x for i,(x,_,_,_) in enumerate(points) if i%every==0]
    y_selected = [y for i,(_,y,_,_) in enumerate(points) if i%every==0]
    z_selected = [z for i,(_,_,z,_) in enumerate(points) if i%every==0]
    rgb_selected = [c for i,c in enumerate(rgb) if i%every==0]
    ax.scatter(x_selected, y_selected, zs=z_selected, c=rgb_selected, linewidths=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    return ax

def brighten_point_cloud(xyzrgb):
    '''
    brighten a point cloud
    the s and v of the hsv color of each point are increased to maximum scale. the object is fully saturated and illuminated
    this function does not change the original point cloud but instead returns a new point cloud
    '''
    new_cloud = []
    for i in xrange(len(xyzrgb)):
        x,y,z,rgb = xyzrgb[i]
        r,g,b = pcl_float_to_rgb(rgb)
        r /= 255
        g /= 255
        b /= 255
        h,_,_ = colorsys.rgb_to_hsv(r,g,b)
        r,g,b = colorsys.hsv_to_rgb(h,1,1)
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
        rgb = rgb_to_pcl_float(r,g,b)
        new_cloud.append([x,y,z,rgb])


