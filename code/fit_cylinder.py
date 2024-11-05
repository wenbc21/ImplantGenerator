import SimpleITK as sitk
import numpy as np
from get_dicom import *
from get_stl import *
from get_pcd import *
from get_cylinder_param import *
import pandas as pd


if __name__ == '__main__':
    
    # read raw data
    predict_path = './dataset/nnUNet_raw/Dataset737_Implant/labelsTs'
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    predict_dirs.sort()
    
    # fit cylinder for each item
    for it in range(len(predict_dirs)) :

        nii_dir_p = predict_dirs[it]
        
        nii_p = sitk.ReadImage(nii_dir_p)
        nii_p = sitk.GetArrayFromImage(nii_p)
        
        # compute cylinder parameters for predict and groundtruth
        center_upper_half, direction_upper_half = get_upper_axis(nii_p)
        center_p, direction_p, radius_p, length_p = get_cylinder_param(nii_p)
        radius_p /= 0.3
        length_p /= 0.3
        normal = pow(pow(length_p / 2,2) + pow(radius_p, 2), 0.5)
        print(nii_p.shape, center_p, direction_p, radius_p, length_p)
        
        # get centers of the top and bottom faces
        upper_center_p = center_p + length_p * direction_p * 0.5
        bottom_center_p = center_p - length_p * direction_p * 0.5
        bottom_center_p = center_upper_half - length_p * direction_upper_half * 0.25
        upper_center_p = center_upper_half - length_p * direction_upper_half * 0.75
        
        # # make cross section
        # os.makedirs(f"./retransform/{str(it+1).zfill(3)}", exist_ok=True)
        # for i in range(96) :
        #     cross_sec_img = nii_p[i, :, :]  # 从三维矩阵中找出横断面切片
        #     cross_sec_img = cross_sec_img.astype("uint8")  # 转换类型
        #     cv2.imwrite(f"./retransform/{str(it+1).zfill(3)}/cross_section{i}.png", cross_sec_img*255)  # 保存横断面
        
        # get cylinder point locations
        cylinder = np.array(np.where(nii_p == 1), dtype=float).T
        
        # calculate bounding box
        x = cylinder[:, 0]
        y = cylinder[:, 1]
        z = cylinder[:, 2]
        xmin = int(np.min(x))
        xmax = int(np.max(x))
        ymin = int(np.min(y))
        ymax = int(np.max(y))
        zmin = int(np.min(z))
        zmax = int(np.max(z))
        
        # build a standard cylinder
        nii_new = np.zeros_like(nii_p)
        for i in range(xmin-5, xmax + 6) :
            for j in range(ymin-5, ymax + 6) :
                for k in range(zmin-5, zmax + 6) :
                    a = np.array(bottom_center_p)
                    b = np.array(upper_center_p)
                    p = np.array([i, j, k], dtype=float)
                    ab = b - a
                    ap = p - a
                    S = np.linalg.norm(np.cross(ab, ap))
                    dis1 = S / length_p
                    dis2 = np.linalg.norm(p - center_p)
                    
                    if dis1 <= radius_p and dis2 <= normal :
                        nii_new[i,j,k]=1
        
        # save
        implant_new = sitk.GetImageFromArray(nii_new)
        sitk.WriteImage(implant_new, f"./data/data_trans/post_2/IMPLANT_{str(it+1).zfill(3)}.nii.gz")
        
        print(f"num: {it + 1}")