import SimpleITK as sitk
import os
import numpy as np


if __name__ == '__main__':

    label_path = './../../nnUNet/dataset/nnUNet_raw/Dataset747_ROIUF/labelsTs'
    image_path = './../../nnUNet/dataset/nnUNet_raw/Dataset747_ROIUF/imagesTs'
    predict_path = './../../nnUNet/dataset/nnUNet_raw/Dataset747_ROIUF/predict'

    # read raw data
    label_dirs = [item.path for item in os.scandir(label_path) if item.is_file()]
    image_dirs = [item.path for item in os.scandir(image_path) if item.is_file()]
    predict_dirs = [item.path for item in os.scandir(predict_path) if item.is_file()]
    label_dirs.sort()
    image_dirs.sort()
    predict_dirs.sort()
    assert len(label_dirs) == len(image_dirs) and len(predict_dirs) == len(image_dirs), "file number not equal!"

    # get cross section for each item
    for it in range(len(image_dirs)) :
        # get dicom file, implant file and predict file
        label = sitk.ReadImage(label_dirs[it])
        label = sitk.GetArrayFromImage(label)
        image = sitk.ReadImage(image_dirs[it])
        image = sitk.GetArrayFromImage(image)
        predict = sitk.ReadImage(predict_dirs[it])
        predict = sitk.GetArrayFromImage(predict)

        if np.max(predict) == 0:
            print(it+1, "No prediction!")
            continue

        # get central position of cylinder
        label_cylinder = np.array(np.where(label == 1))
        label_midx, label_midy, label_midz = (np.min(label_cylinder, axis=1) + np.max(label_cylinder, axis=1)) // 2
        predict_cylinder = np.array(np.where(predict == 1))
        predict_midx, predict_midy, predict_midz = (np.min(predict_cylinder, axis=1) + np.max(predict_cylinder, axis=1)) // 2
        distance = ((label_midx - predict_midx) ** 2 + (label_midy - predict_midy) ** 2 + (label_midz - predict_midz) ** 2)**0.5

        print(it+1, \
            "    labeled pos.", label_midx, label_midy, label_midz, \
            "    predict pos.", predict_midx, predict_midy, predict_midz, \
            "    distance: ", distance)