from stl import mesh
import open3d as o3d
import numpy as np
import random

# 根据文件路径，返回STL的三维数组，像素归一到[0, 255]
def get_stl(dir_path, dicom_size) :
    
    # 加载stl文件
    cylinder = mesh.Mesh.from_file(dir_path)
    # 输出stl信息
    # print(cylinder.vectors)
    
    # 获取所有点，去重
    cylinder = cylinder.vectors
    flatten = []
    for triangle in cylinder :
        flatten.append(list(np.array(triangle).reshape(1, -1)))
    cylinder = np.reshape(flatten, (-1, 3))
    cylinder = np.unique(cylinder, axis=0)
    
    # 根据DICOM大小进行变换(原大小)
    trans_cylinder = []
    for c in cylinder :
        trans_cylinder.append(np.array([
            dicom_size[0] * 0.5 - c[2] / 0.3,
            dicom_size[1] * 0.5 + c[1] / 0.3,
            dicom_size[2] * 0.5 - c[0] / 0.3]))
    # # 根据DICOM大小进行变换（截取后大小）
    # trans_cylinder = []
    # for c in cylinder :
    #     trans_cylinder.append(np.array([
    #         - c[2] / 0.3,
    #         dicom_size[1] * 0.5 + c[1] / 0.3,
    #         dicom_size[2] * 0.5 - c[0] / 0.3]))
    
    append = []
    num = len(trans_cylinder)
    for it in range(950):
        i = random.randint(0, num - 1)
        j = random.randint(0, num - 1)
        ran = random.random()
        append.append(trans_cylinder[i] * ran + trans_cylinder[j] * (1 - ran))
    trans_cylinder = np.append(trans_cylinder, append, axis = 0)
    
    num = len(trans_cylinder)
    for it in range(5000):
        i = random.randint(0, num - 1)
        j = random.randint(0, num - 1)
        ran = random.random()
        trans_cylinder = np.append(trans_cylinder, [trans_cylinder[i] * ran + trans_cylinder[j] * (1 - ran)], axis = 0)
        num += 1
    trans_cylinder = np.rint(np.unique(trans_cylinder, axis=0))

    # build a 3d space
    stl_3d_array = np.zeros(dicom_size, dtype=np.uint8)
    for t_c in trans_cylinder :
        stl_3d_array[int(t_c[0])][int(t_c[1])][int(t_c[2])] = 255
    
    return stl_3d_array
