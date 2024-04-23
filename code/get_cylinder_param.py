import SimpleITK as sitk
import os
import cv2
import numpy as np
from get_dicom import *
from get_stl import *
from get_pcd import *
import open3d as o3d
import copy
import pandas as pd

def scan_path(path) :
    dirs = []
    files = []

    for item in os.scandir(path):
        if item.is_dir():
          dirs.append(item.path)

        elif item.is_file():
          files.append(item.path)
          
    return dirs, files

def get_mid(pcd) :
    
    cylinder = np.array(np.where(pcd == 1))
    midx = (np.min(cylinder[0]) + np.max(cylinder[0])) // 2
    midy = (np.min(cylinder[1]) + np.max(cylinder[1])) // 2
    midz = (np.min(cylinder[2]) + np.max(cylinder[2])) // 2
    
    return midx, midy, midz

def get_cylinder_param(nii_retransform) :
    # 取体素
    cylinder = np.array(np.where(nii_retransform == 1), dtype=float).T
    
    # 拟合中轴线
    output = cv2.fitLine(cylinder, distType = cv2.DIST_L2, param = 0, reps = 1e-2, aeps = 1e-2)
    center = np.array(output[3:]).T[0]
    direction = np.array(output[:3]).T[0]
    a = center
    b = direction + center
    
    # for i in range(600) :
    #     d = (i - 300 ) * 0.1
    #     cylinder = np.row_stack((cylinder, center+d*direction))
    
    # # From numpy to Open3D
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cylinder)
    # o3d.visualization.draw_geometries([pcd])
    # exit()
    
    # 计算包围盒
    x = cylinder[:, 0]
    y = cylinder[:, 1]
    z = cylinder[:, 2]
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    zmin = int(np.min(z))
    zmax = int(np.max(z))
    
    # 取侧面外围点
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
    
    # 计算外围点投影到中心距离，用于排除底面点
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
    
    # # From numpy to Open3D
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cylinder_hollow)
    # o3d.visualization.draw_geometries([pcd])
    # exit()
    
    # 计算直径
    distances = []
    for p in cylinder_hollow :
        cross = np.linalg.norm(np.cross(b-a, p-a))
        distances.append(cross / np.linalg.norm(b-a))
    radius = np.mean(distances) * 0.3 - 0.075
    
    # 取底面外围点
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
    
    # 计算投影
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
    
    # 计算长度
    distances = []
    for p in vertical :
        distances.append(np.linalg.norm(p - center))
    distances = np.array(distances) * 0.3
    
    # 统计各区间数量，取最高区间，即底面圆
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
    
    # # 点云可视化
    # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(np.vstack((vertical, cylinder)))
    # pcd.points = o3d.utility.Vector3dVector(cylinder_hollow)
    # o3d.visualization.draw_geometries([pcd])
    
    return center, direction, radius, length
    

if __name__ == '__main__':
    
    # 文件路径
    dicom_path = 'mandibular_tooth_implant'
    pcd_path = 'pcd'
    nii_path = 'postprocess'
    # nii_path = 'nii/label'
    dicom_dirs, _ = scan_path(dicom_path)
    _, pcd_dirs = scan_path(pcd_path)
    _, nii_dirs = scan_path(nii_path)
    store_path = './cross_section_predict'
    test_index = [4, 6, 26, 30]
    
    if len(dicom_dirs) != len(pcd_dirs):
        print("error!")
        exit()
    dicom_dirs = [dicom_dirs[i-1] for i in test_index]
    pcd_dirs = [pcd_dirs[i-1] for i in test_index]
    num = len(dicom_dirs)
    
    
    # 循环处理每一例患者
    for it in range(num) :
        
        # 读取文件
        dicom_dir, _ = scan_path(dicom_dirs[it])
        dicom_dir = dicom_dir[0]
        pcd_dir = pcd_dirs[it]
        nii_dir = nii_dirs[it]

        dicom = get_dicom(dicom_dir)
        dicom_size = dicom.shape
        pcd = get_pcd(pcd_dir, dicom_size)
        nii = sitk.ReadImage(nii_dir)
        nii = sitk.GetArrayFromImage(nii)
        
        # 恢复到原空间
        midx, midy, midz = get_mid(pcd)
        nii_retransform = np.zeros(dicom_size, dtype=np.uint8)
        nii_retransform[midx-36:midx+36, midy-36:midy+36, midz-36:midz+36] = nii
        
        # 翻转z轴（因为最初获取dicom时翻转了一次，在此需要恢复）
        dicom = dicom[::-1, :, :]
        pcd = pcd[::-1, :, :]
        nii_retransform = nii_retransform[::-1, :, :]
        
        # 计算圆柱体参数
        center, direction, radius, length = get_cylinder_param(nii_retransform)
        
        print(f"num:{it + 1}")
        print(f"diameter:{radius * 2} length:{length}")
        