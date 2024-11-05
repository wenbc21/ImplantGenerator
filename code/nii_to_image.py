import SimpleITK as sitk
import os
from get_dicom import *
from get_stl import *
from get_pcd import *


if __name__ == '__main__':

    # data files
    label_path = './data/nii/labeled/label'
    raw_path = './data/nii/labeled/raw'
    predict_path = './best_test'
    save_path = 'temptemp'

    # read raw data
    label_dirs = [item.path for item in os.scandir(label_path) if item.is_file()]
    raw_dirs = [item.path for item in os.scandir(raw_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    label_dirs.sort()
    raw_dirs.sort()
    predict_dirs.sort()
    assert len(label_dirs) == len(raw_dirs) and len(predict_dirs) == len(raw_dirs), "file number not equal!"
    
    # get cross section for each item
    for it in range(len(raw_dirs)) :
        # get dicom file, implant file and predict file
        label = sitk.ReadImage(label_dirs[it])
        label = sitk.GetArrayFromImage(label)
        raw = sitk.ReadImage(raw_dirs[it])
        raw = sitk.GetArrayFromImage(raw)
        predict = sitk.ReadImage(predict_dirs[it])
        predict = sitk.GetArrayFromImage(predict)
        
        # get cross section
        os.makedirs(f"{save_path}/label/{it}", exist_ok=True)
        os.makedirs(f"{save_path}/raw/{it}", exist_ok=True)
        os.makedirs(f"{save_path}/predict/{it}", exist_ok=True)
        for i in range(raw.shape[0]) :
            cross_sec_img = label[i, :, :]
            cross_sec_img = cross_sec_img.astype("uint8")
            cv2.imwrite(f"{save_path}/label/{it}/cross_section{i}.png", cross_sec_img * 255)
            
            cross_sec_img = raw[i, :, :]
            cross_sec_img = cross_sec_img.astype("uint8")
            cv2.imwrite(f"{save_path}/raw/{it}/cross_section{i}.png", cross_sec_img)

            cross_sec_img = predict[i, :, :]
            cross_sec_img = cross_sec_img.astype("uint8")
            cv2.imwrite(f"{save_path}/predict/{it}/cross_section{i}.png", cross_sec_img * 255)
        
        print(f"{str(it+1).zfill(3)} done!")