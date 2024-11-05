import os
import cv2
import numpy as np
from get_dicom import *
from get_stl import *
from get_pcd import *
import open3d as o3d
import copy
import pandas as pd


def get_cylinder_param(nii_retransform) :
    # get cylinder point locations
    cylinder = np.array(np.where(nii_retransform == 1), dtype=float).T
    
    # fit the main axis
    output = cv2.fitLine(cylinder, distType = cv2.DIST_L2, param = 0, reps = 1e-2, aeps = 1e-2)
    center = np.array(output[3:]).T[0]
    direction = np.array(output[:3]).T[0]
    a = center
    b = direction + center
    
    # get bounding box
    x = cylinder[:, 0]
    y = cylinder[:, 1]
    z = cylinder[:, 2]
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    zmin = int(np.min(z))
    zmax = int(np.max(z))
    
    # get surface points of the cylinder
    nii_temp = copy.deepcopy(nii_retransform)
    for x in range(xmin, xmax + 1) :
        for y in range(ymin, ymax + 1) :
            min = zmax + 2
            max = 0
            for z in range(zmin, zmax + 1) :
                if nii_temp[x][y][z] == 1 :
                    if min == zmax + 2:
                        min = z
                    max = z
            for zit in range(min + 1, max) :
                nii_temp[x][y][zit] = 0
    cylinder_hollow = np.array(np.where(nii_temp == 1)).T
    
    # calculate distances of the points from the axis
    # in order to eliminate top and bottom faces
    distances = []
    for p in cylinder_hollow :
        num = (a[0] - p[0]) * (b[0] - a[0]) + (a[1] - p[1]) * (b[1] - a[1]) + (a[2] - p[2]) * (b[2] - a[2])
        den = (np.linalg.norm(b-a)) * (np.linalg.norm(b-a))
        k = -num / den
        vertical = [k * (b[0] - a[0]) + a[0], k * (b[1] - a[1]) + a[1], k * (b[2] - a[2]) + a[2]]
        distances.append(np.linalg.norm(vertical - center))
    dis_sort = np.sort(np.array(distances))
    threshold = dis_sort[round(len(dis_sort)* 0.95)]
    delete_index = []
    for i in range(len(distances)) :
        if distances[i] > threshold :
            delete_index.append(i)
    cylinder_hollow = np.delete(cylinder_hollow, delete_index, axis = 0)
    
    # get diameter (radius) of the cylinder
    distances = []
    for p in cylinder_hollow :
        cross = np.linalg.norm(np.cross(b-a, p-a))
        distances.append(cross / np.linalg.norm(b-a))
    radius = np.mean(distances) * 0.3 - 0.075
    
    # get surface points of the cylinder
    nii_temp = copy.deepcopy(nii_retransform)
    for z in range(zmin, zmax + 1) :
        for y in range(ymin, ymax + 1) :
            min = xmax + 2
            max = 0
            for x in range(xmin, xmax + 1) :
                if nii_temp[x][y][z] == 1 :
                    if min == xmax + 2:
                        min = x
                    max = x
            for xit in range(min + 1, max) :
                nii_temp[xit][y][z] = 0
    cylinder_hollow = np.array(np.where(nii_temp == 1)).T
    
    # get projection of the points onto the axis
    vertical = []
    for p in cylinder_hollow :
        num = (a[0] - p[0]) * (b[0] - a[0]) + (a[1] - p[1]) * (b[1] - a[1]) + (a[2] - p[2]) * (b[2] - a[2])
        den = (np.linalg.norm(b-a)) * (np.linalg.norm(b-a))
        k = -num / den
        vertical.append([k * (b[0] - a[0]) + a[0], k * (b[1] - a[1]) + a[1], k * (b[2] - a[2]) + a[2]])
    vertical = np.array(vertical)
    
    # # From numpy to Open3D
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vertical)
    # o3d.visualization.draw_geometries([pcd])
    # exit()
    
    # get length of the cylinder
    distances = []
    for p in vertical :
        distances.append(np.linalg.norm(p - center))
    distances = np.array(distances) * 0.3
    
    # get maximal and minimal interval as top and bottom faces
    score = pd.Series(distances)
    sel = pd.cut(score, [i * 0.1 for i in range(100)])
    eval = [sel.value_counts().index[0].left, 
            sel.value_counts().index[0].right, 
            sel.value_counts().index[1].left, 
            sel.value_counts().index[1].right]
    min_interval, max_interval = np.min(eval), np.max(eval)
    distances = distances[distances > min_interval]
    distances = distances[distances < max_interval]
    length = np.mean(distances) * 2
    
    # # visualize
    # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(np.vstack((vertical, cylinder)))
    # pcd.points = o3d.utility.Vector3dVector(cylinder_hollow)
    # o3d.visualization.draw_geometries([pcd])
    
    return center, direction, radius, length
