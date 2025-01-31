import open3d as o3d
import numpy as np


def get_pcd(dir_path, dicom_size) :
    
    pcd = o3d.io.read_point_cloud(dir_path)
    cylinder = np.asarray(pcd.points)

    # 根据DICOM大小进行变换
    trans_cylinder = []
    for c in cylinder :
        trans_cylinder.append(np.array([
            dicom_size[0] * 0.5 - c[2] / 0.3,
            dicom_size[1] * 0.5 + c[1] / 0.3,
            dicom_size[2] * 0.5 + c[0] / 0.3]))
        
    trans_cylinder = np.rint(np.unique(trans_cylinder, axis=0))
    
    x = trans_cylinder[:, 0]
    y = trans_cylinder[:, 1]
    z = trans_cylinder[:, 2]
    
    trans_cylinder = np.delete(trans_cylinder, np.argwhere(x >= dicom_size[0]), axis = 0)
    trans_cylinder = np.delete(trans_cylinder, np.argwhere(y >= dicom_size[1]), axis = 0)
    trans_cylinder = np.delete(trans_cylinder, np.argwhere(z >= dicom_size[2]), axis = 0)
    
    x = trans_cylinder[:, 0]
    y = trans_cylinder[:, 1]
    z = trans_cylinder[:, 2]
    
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    zmin = int(np.min(z))
    zmax = int(np.max(z))
    
    # 构造三维空间
    stl_3d_array = np.zeros(dicom_size, dtype=np.uint8)
    for t_c in trans_cylinder :
        stl_3d_array[int(t_c[0])][int(t_c[1])][int(t_c[2])] = 1
    
    
    for x in range(xmin, xmax + 1) :
        for y in range(ymin, ymax + 1) :
            localmin = zmax + 2
            localmax = 0
            for z in range(zmin, zmax + 1) :
                if stl_3d_array[x][y][z] == 1 :
                    if localmin == zmax + 2:
                        localmin = z
                    localmax = z
            for zit in range(localmin, localmax + 1) :
                stl_3d_array[x][y][zit] = 1
                
    for x in range(xmin, xmax + 1) :
        for z in range(zmin, zmax + 1) :
            localmin = ymax + 2
            localmax = 0
            for y in range(ymin, ymax + 1) :
                if stl_3d_array[x][y][z] == 1 :
                    if localmin == ymax + 2:
                        localmin = y
                    localmax = y
            for yit in range(localmin, localmax + 1) :
                stl_3d_array[x][yit][z] = 1
                
    for z in range(zmin, zmax + 1) :
        for y in range(ymin, ymax + 1) :
            localmin = xmax + 2
            localmax = 0
            for x in range(xmin, xmax + 1) :
                if stl_3d_array[x][y][z] == 1 :
                    if localmin == xmax + 2:
                        localmin = x
                    localmax = x
            for xit in range(localmin, localmax + 1) :
                stl_3d_array[xit][y][z] = 1
    
    return stl_3d_array
