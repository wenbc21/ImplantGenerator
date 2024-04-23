import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
from get_pcd import *


if __name__ == '__main__':

    label_path = './data/nii/labeled/label'
    raw_path = './data/nii/labeled/raw'
    predict_path = './best_test'
    save_path = 'temptemp'

    label_dirs = [item.path for item in os.scandir(label_path) if item.is_file()]
    raw_dirs = [item.path for item in os.scandir(raw_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    label_dirs.sort()
    raw_dirs.sort()
    predict_dirs.sort()

    assert len(label_dirs) == len(raw_dirs) and len(predict_dirs) == len(raw_dirs), "file number not equal!"
    
    for it in range(len(raw_dirs)) :
        label = sitk.ReadImage(label_dirs[it])
        label = sitk.GetArrayFromImage(label)
        raw = sitk.ReadImage(raw_dirs[it])
        raw = sitk.GetArrayFromImage(raw)
        predict = sitk.ReadImage(predict_dirs[it])
        predict = sitk.GetArrayFromImage(predict)
        
        # 产生截面
        os.makedirs(f"{save_path}/label/{it}", exist_ok=True)
        os.makedirs(f"{save_path}/raw/{it}", exist_ok=True)
        os.makedirs(f"{save_path}/predict/{it}", exist_ok=True)
        for i in range(raw.shape[0]) :
            cross_sec_img = label[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
            cv2.imwrite(f"{save_path}/label/{it}/cross_section{i}.png", cross_sec_img * 255)  # 保存横断面
            
            cross_sec_img = raw[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
            cv2.imwrite(f"{save_path}/raw/{it}/cross_section{i}.png", cross_sec_img)  # 保存横断面

            cross_sec_img = predict[i, :, :]  # 从三维矩阵中找出横断面切片
            cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
            cv2.imwrite(f"{save_path}/predict/{it}/cross_section{i}.png", cross_sec_img * 255)  # 保存横断面
        
        print(f"{it+1} done!")