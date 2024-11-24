import numpy as np
import open3d as o3d
import os
import cv2
from get_dicom import *

def stl_to_voxel_array_with_fill(stl_file_path, dicom_size,voxel_size=0.3):
    """
    将 STL 文件转化为三维稠密体素网格并完成表面填充和内部填充。
    Args:
        stl_file_path (str): STL 文件路径。
        dicom_size (tuple): 目标三维网格大小（x, y, z）。
        voxel_size (float): 体素尺寸。
    Returns:
        stl_3d_array (np.ndarray): 填充完成后的稠密三维体素网格。
    """
    # 加载 STL 文件
    cylinder = o3d.io.read_triangle_mesh(stl_file_path)
    # 获取网格的顶点
    original_vertices = np.asarray(cylinder.vertices)  # 提取原始顶点
    # 从三角网格表面均匀采样点
    sampled_points = cylinder.sample_points_uniformly(number_of_points=10000)
    sampled_points = np.asarray(sampled_points.points)  # 提取采样点的坐标
    # 合并原始顶点和采样点，并去重
    filled_cylinder = np.vstack((original_vertices, sampled_points))  # 合并
    filled_cylinder = np.unique(filled_cylinder, axis=0)  # 去重

    transformed_points = []
    for point in filled_cylinder:
        transformed_point = np.array([
            dicom_size[0] * 0.5 - point[2] / voxel_size,
            dicom_size[1] * 0.5 + point[1] / voxel_size,
            dicom_size[2] * 0.5 - point[0] / voxel_size
        ])
        transformed_points.append(transformed_point)

    transformed_points = np.array(transformed_points)  # 转为 NumPy 数组
    
    # 转换为体素网格坐标
    voxel_coords = np.floor(transformed_points).astype(int)
    
    # 初始化稠密体素网格
    stl_3d_array = np.zeros(dicom_size, dtype=np.uint8)
    for voxel in voxel_coords:
        stl_3d_array[voxel[0], voxel[1], voxel[2]] = 255
    # 获取非零点的边界
    x, y, z = np.where(stl_3d_array == 255)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()

    # 内部填充逻辑
    for x in range(xmin, xmax + 1):  # 遍历 x 轴范围内的每一个体素列
        for y in range(ymin, ymax + 1):  # 对每个 x 值，遍历 y 轴范围内的每一个体素列
            z_vals = np.where(stl_3d_array[x, y, :] == 255)[0]  # 找出在 (x, y) 平面上 z 轴方向上被填充为 1 的体素索引
            if len(z_vals) > 1:  # 如果在该列中，z 轴方向有多个被填充为 1 的体素
                stl_3d_array[x, y, z_vals[0]:z_vals[-1] + 1] = 255  # 将 z 轴方向上从第一个到最后一个索引之间的体素全部填充为 1

    for x in range(xmin, xmax + 1):
        for z in range(zmin, zmax + 1):
            y_vals = np.where(stl_3d_array[x, :, z] == 255)[0]
            if len(y_vals) > 1:
                stl_3d_array[x, y_vals[0]:y_vals[-1] + 1, z] = 255

    for z in range(zmin, zmax + 1):
        for y in range(ymin, ymax + 1):
            x_vals = np.where(stl_3d_array[:, y, z] == 255)[0]
            if len(x_vals) > 1:
                stl_3d_array[x_vals[0]:x_vals[-1] + 1, y, z] = 255

    return stl_3d_array

# #可视化体素网格
# def voxel_array_to_open3d(voxel_array, voxel_size=0.3):
#     voxel_coords = np.argwhere(voxel_array > 0)
#     voxel_points = voxel_coords * voxel_size
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(voxel_points)
#     pcd.paint_uniform_color([0.5, 0.5, 0.5])  
#     return pcd

# #生成种植体切片图像
# def save_slices(voxel_array, save_path, slice_size=512, upscale_factor=4):
#     os.makedirs(save_path, exist_ok=True)
#     for i in range(voxel_array.shape[0]):
#         # 将切片转为灰度图像
#         slice_img = voxel_array[i, :, :] * 1
#         # 使用插值增加分辨率
#         high_res_size = (slice_size * upscale_factor, slice_size * upscale_factor)
#         slice_img = cv2.resize(slice_img, high_res_size, interpolation=cv2.INTER_CUBIC)
#         # 保存为 PNG 图像
#         cv2.imwrite(os.path.join(save_path, f"slice_{i:04d}.png"), slice_img)
#     print(f"High-resolution slices saved to {save_path}")


# if __name__ == '__main__':
#     # 参数设置
#     stl_file = "../data1/001_张研_36_B.stl"  
#     voxel_size = 0.3  # 单体素的大小
#     dicom_dir='../data/data/01_mimics/01_mimics'
#     dicom = get_dicom(dicom_dir)
#     dicom_size=dicom.shape
    
#     save_path = "../slices1"  # 切片保存路径

#     # 1. 将 STL 文件转换为稠密体素网格
#     print("Generating voxel grid...")
#     filled_voxel_array = stl_to_voxel_array_with_fill(stl_file, dicom_size, voxel_size)

#     # 2. 可视化稠密体素网格
#     print("Visualizing voxel grid...")
#     pcd = voxel_array_to_open3d(filled_voxel_array, voxel_size)
#     o3d.visualization.draw_geometries([pcd])

#     # 3. 保存高分辨率切片图像
#     print("Saving high-resolution slices...")
#     save_slices(filled_voxel_array, save_path, slice_size=256, upscale_factor=4)

   
