import SimpleITK as sitk
import numpy as np

# 使用SITK获取DICOM的三维数组
def get_dcm_3d_array(dicom_dir) :
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    size = image.GetSize()
    # print("Image size:", size[0], size[1], size[2], type(image)) # height, depth, width
    dcm_3d_array = sitk.GetArrayFromImage(image)
    # print(dcm_3d_array.shape, dcm_3d_array.dtype)
    dcm_3d_array = dcm_3d_array[::-1, :, :]
    return dcm_3d_array

# 将单张CBCT图像根据窗位和窗宽进行转换
def window_transform_2d(dcm_2d_array, window_width, window_center, normal=False):
    """
    window_width: 窗位
    window_center: 窗宽
    """
    min_window = float(window_center) - 0.5 * float(window_width)
    new_2d_array = (dcm_2d_array - min_window) / float(window_width)
    new_2d_array[new_2d_array < 0] = 0
    new_2d_array[new_2d_array > 1] = 1
    if not normal:
        new_2d_array = (new_2d_array * 255).astype('uint8')
    return new_2d_array

# 将整个CBCT图像利用窗位和窗宽转换
def window_transform_3d(dcm_3d_array, window_width, window_center, low_slice_num=0, high_slice_num=0, normal=False) :
    
    if(high_slice_num == 0):
        high_slice_num = len(dcm_3d_array)
    
    for slice_num in range(low_slice_num, high_slice_num):
        dcm_3d_array[slice_num] = window_transform_2d(dcm_3d_array[slice_num], window_width, window_center, normal)

    return dcm_3d_array


# 根据文件路径，返回DICOM的三维数组，像素归一到[0, 255]
def get_dicom(dir_path) :
    
    # 获取CBCT图像所有slice并转化为numpy三维矩阵，分别是横断面从上往下、冠状面从前往后、矢状面从左往右
    dcm_3d_array = get_dcm_3d_array(dir_path)
    
    # 根据窗位调整
    dcm_3d_array = window_transform_3d(dcm_3d_array, window_width=4000, window_center=1000).astype(np.uint8)
    
    # # 三维矩阵的高度，宽度，深度
    # height, depth, width = dcm_3d_array.shape
    
    return dcm_3d_array